# main.py
print("✅ main.py started") # First line to confirm script execution

import runpod
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler, ControlNetModel
from diffusers.utils import export_to_video
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from controlnet_aux import OpenposeDetector, MidasDetector
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np
import base64
import io
import requests
import os
import tempfile
import asyncio # Added for local testing with asyncio.run

print("✅ Container started. Script parsing begins.")

# ------------------------------ IP-Adapter Helper Class ----------------------------- #
class IPAdapterImageProj(torch.nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        self.image_proj_model = self.init_proj(state_dict)
    def init_proj(self, state_dict):
        image_proj_model = torch.nn.Linear(state_dict["image_proj"].shape[-1], state_dict["ip_adapter"].shape[0])
        image_proj_model.load_state_dict({"weight": state_dict["image_proj"], "bias": torch.zeros(state_dict["image_proj"].shape[0])})
        return image_proj_model
    def forward(self, image_embeds):
        return self.image_proj_model(image_embeds)

# --------------------------------- Globals for Lazy-Loading --------------------------------- #
pipe = None
image_encoder = None
image_proj_model = None
image_processor = None
openpose_detector = None
midas_detector = None

# ------------------------------ File Upload Utility ----------------------------- #
def upload_to_catbox(filepath):
    try:
        with open(filepath, 'rb') as f:
            files = {'fileToUpload': (os.path.basename(filepath), f)}
            data = {'reqtype': 'fileupload'}
            response = requests.post('https://litterbox.catbox.moe/resources/internals/api.php', files=files, data=data)
            response.raise_for_status()
            return response.text
    except Exception as e:
        return f"Error uploading: {str(e)}"

# --------------------------------- Job Handler ---------------------------------- #
async def generate_video(job): # Changed to async def
    print("📥 Job received:", job) # Log the incoming job payload
    global pipe, image_encoder, image_proj_model, image_processor, openpose_detector, midas_detector

    if pipe is None:
        print("⏳ Starting model loading...")

        if not torch.cuda.is_available(): # Check for CUDA availability
            raise RuntimeError("CUDA is not available! GPU required for this application.")

        HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")
        if HUGGING_FACE_TOKEN:
            print("🔐 Using HF token:", HUGGING_FACE_TOKEN[:8] + "...")
        else:
            print(⚠️ HF token not found. Downloads may fail for gated models.")

        base_model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
        motion_module_id = "guoyww/animatediff-motion-module-v3"
        ip_adapter_repo_id = "h94/IP-Adapter"

        openpose_controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16, token=HUGGING_FACE_TOKEN)
        depth_controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16, token=HUGGING_FACE_TOKEN)
        openpose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet", token=HUGGING_FACE_TOKEN)
        midas_detector = MidasDetector.from_pretrained("lllyasviel/ControlNet", token=HUGGING_FACE_TOKEN)

        pipe = AnimateDiffPipeline.from_pretrained(
            base_model_id,
            controlnet=[openpose_controlnet, depth_controlnet],
            torch_dtype=torch.float16,
            token=HUGGING_FACE_TOKEN
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.load_motion_module(motion_module_id, unet_additional_kwargs={"use_inflated_groupnorm": True}, token=HUGGING_FACE_TOKEN)
        pipe.enable_model_cpu_offload()

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            ip_adapter_repo_id, subfolder="models/image_encoder", torch_dtype=torch.float16, token=HUGGING_FACE_TOKEN
        ).to("cuda")
        image_processor = CLIPImageProcessor.from_pretrained(
            ip_adapter_repo_id, subfolder="models/image_encoder", token=HUGGING_FACE_TOKEN
        )
        ip_adapter_path = hf_hub_download(repo_id=ip_adapter_repo_id, filename="ip-adapter_sd15.bin", token=HUGGING_FACE_TOKEN)
        ip_adapter_weights = torch.load(ip_adapter_path, map_location="cpu")
        image_proj_model = IPAdapterImageProj(ip_adapter_weights).to("cuda")

        print("✅ Models loaded successfully.")

    job_input = job.get('input', {})
    base64_image = job_input.get('init_image')
    prompt = job_input.get('prompt', 'a couple kissing, beautiful, cinematic')

    num_frames = int(job_input.get('num_frames', 16))
    fps = int(job_input.get('fps', 8))
    ip_adapter_scale = min(max(float(job_input.get('ip_adapter_scale', 0.7)), 0.0), 1.5)
    openpose_scale = min(max(float(job_input.get('openpose_scale', 1.0)), 0.0), 1.5)
    depth_scale = min(max(float(job_input.get('depth_scale', 0.5)), 0.0), 1.5)

    if not base64_image:
        print("❌ 'init_image' missing in job input.")
        return {"error": "Missing 'init_image' base64 input."}

    try:
        init_image = Image.open(io.BytesIO(base64.b64decode(base64_image))).convert("RGB")
    except Exception as e:
        return {"error": f"Image decode error: {str(e)}"}

    print("🔍 Preprocessing image...")
    processed_image = image_processor(images=init_image, return_tensors="pt").pixel_values.to("cuda", dtype=torch.float16)
    clip_features = image_encoder(processed_image).image_embeds
    image_embeds = image_proj_model(clip_features)
    openpose_image = openpose_detector(init_image)
    depth_image = midas_detector(init_image)
    control_images = [openpose_image, depth_image]

    cross_attention_kwargs = {"ip_adapter_image_embeds": image_embeds, "scale": ip_adapter_scale}

    print("✨ Inference started...")
    output = pipe(
        prompt=prompt,
        negative_prompt="ugly, distorted, low quality, cropped, blurry",
        num_frames=num_frames,
        guidance_scale=7.5,
        num_inference_steps=20,
        image=control_images,
        controlnet_conditioning_scale=[openpose_scale, depth_scale],
        cross_attention_kwargs=cross_attention_kwargs
    )
    frames = output.frames[0]

    print("📼 Exporting video...")
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "kissify.mp4")
        export_to_video(frames, video_path, fps=fps)

        print("🚀 Uploading video...")
        url = upload_to_catbox(video_path)
        if "Error" in url:
            return {"error": url}
        return {"output": {"video_url": url}}

# Starts the job listener – REQUIRED!
if __name__ == "__main__": # Guard for direct execution
    print("🚀 Ready to receive jobs...")
    runpod.serverless.start({"handler": generate_video})

    # Optional: Local Test Mode (uncomment to use)
    # To use this, you would need to provide a valid base64 encoded image string
    # For example, you can convert a small image to base64 online and paste it here.
    # print("\n--- Running Local Test ---")
    # with open("test_image.jpg", "rb") as image_file:
    #     base64_test_image = base64.b64encode(image_file.read()).decode('utf-8')
    #
    # fake_job = {
    #     "input": {
    #         "init_image": base64_test_image, # Replace with a real base64 image string for testing
    #         "prompt": "a couple kissing under the moonlight, cinematic, romantic",
    #         "num_frames": 16,
    #         "fps": 8,
    #         "ip_adapter_scale": 0.8,
    #         "openpose_scale": 1.2,
    #         "depth_scale": 0.6
    #     }
    # }
    # try:
    #     # runpod.serverless.start handles the asyncio loop, so direct asyncio.run is for standalone testing without runpod.serverless
    #     # If you uncomment the local test, ensure runpod.serverless.start is commented out for that test run
    #     result = asyncio.run(generate_video(fake_job))
    #     print("Local test result:", result)
    # except Exception as e:
    #     print(f"Local test failed: {e}")