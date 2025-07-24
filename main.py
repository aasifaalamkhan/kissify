# main.py
print("✅ main.py started", flush=True) # First line to confirm script execution, with flush=True

import runpod
import torch
import traceback # Import traceback for detailed error logging
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
from threading import Thread # Added for health check server
from flask import Flask # Added for health check server (requires 'flask' in requirements.txt)

# Optimize cuDNN for consistent input shapes (like image sizes)
torch.backends.cudnn.benchmark = True

print("✅ Container started. Script parsing begins.", flush=True)

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
        print(f"❌ Error uploading to Catbox: {traceback.format_exc()}", flush=True) # Detailed log
        return f"Error uploading: {str(e)}"

# --------------------------------- Health Check Server ---------------------------------- #
def run_healthcheck_server():
    app = Flask(__name__)
    @app.route('/healthz')
    def health():
        return "ok"
    # Use 0.0.0.0 to make it accessible from outside the container
    app.run(host="0.0.0.0", port=3000, debug=False, use_reloader=False)

# --------------------------------- Job Handler ---------------------------------- #
def generate_video(job): # Changed back to def (synchronous)
    print("📥 Job received:", job, flush=True) # Log the incoming job payload
    global pipe, image_encoder, image_proj_model, image_processor, openpose_detector, midas_detector

    if pipe is None:
        print("⏳ Starting model loading...", flush=True)

        try: # Added try-except for model loading
            if not torch.cuda.is_available(): # Check for CUDA availability
                raise RuntimeError("CUDA is not available! GPU required for this application.")

            # Provide an empty string fallback for token
            HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN", "")
            if HUGGING_FACE_TOKEN:
                print("🔐 Using HF token:", HUGGING_FACE_TOKEN[:8] + "...", flush=True)
            else:
                # Corrected print statement with proper closing quote
                print("⚠️ HF token not found. Downloads may fail for gated models.", flush=True)

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

            print("✅ Models loaded successfully.", flush=True)
        except Exception as e:
            print(f"❌ Model load failed: {traceback.format_exc()}", flush=True) # Detailed log for model load errors
            return {"error": "Model load failed: " + str(e)}


    job_input = job.get('input', {})
    base64_image = job_input.get('init_image')
    prompt = job_input.get('prompt', 'a couple kissing, beautiful, cinematic')

    num_frames = int(job_input.get('num_frames', 16))
    fps = int(job_input.get('fps', 8))
    ip_adapter_scale = min(max(float(job_input.get('ip_adapter_scale', 0.7)), 0.0), 1.5)
    openpose_scale = min(max(float(job_input.get('openpose_scale', 1.0)), 0.0), 1.5)
    depth_scale = min(max(float(job_input.get('depth_scale', 0.5)), 0.0), 1.5)

    if not base64_image:
        print("❌ 'init_image' missing in job input.", flush=True)
        return {"error": "Missing 'init_image' base64 input."}

    try:
        init_image = Image.open(io.BytesIO(base64.b64decode(base64_image))).convert("RGB")
    except Exception as e:
        print(f"❌ Image decode error: {traceback.format_exc()}", flush=True) # Detailed log
        return {"error": f"Image decode error: {str(e)}"}

    print("🔍 Preprocessing image...", flush=True)
    processed_image = image_processor(images=init_image, return_tensors="pt").pixel_values.to("cuda", dtype=torch.float16)
    clip_features = image_encoder(processed_image).image_embeds
    image_embeds = image_proj_model(clip_features)
    openpose_image = openpose_detector(init_image)
    depth_image = midas_detector(init_image)
    control_images = [openpose_image, depth_image]

    cross_attention_kwargs = {"ip_adapter_image_embeds": image_embeds, "scale": ip_adapter_scale}

    print("✨ Inference started...", flush=True)
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

    print("📼 Exporting video...", flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "kissify.mp4")
        export_to_video(frames, video_path, fps=fps)

        print("🚀 Uploading video...", flush=True)
        url = upload_to_catbox(filepath=video_path) # Changed to use named argument for clarity
        if "Error" in url:
            return {"error": url}
        return {"output": {"video_url": url}}

# Entry point for the RunPod worker
if __name__ == "__main__": # Guard for direct execution
    # Start the health check server in a separate thread
    # This keeps the container alive and responsive to health checks
    Thread(target=run_healthcheck_server, daemon=True).start()

    try:
        print("🚀 Ready to receive jobs...", flush=True)
        runpod.serverless.start({"handler": generate_video})
    except Exception as e:
        # Catch and log any exceptions that prevent runpod.serverless.start from fully initializing
        print(f"❌ Server crashed during startup: {traceback.format_exc()}", flush=True)
        # It's crucial for the container to exit with a non-zero code if startup fails
        # This will be handled by RunPod's monitoring, which will eventually terminate the unhealthy worker
        # No explicit exit(1) needed here, as an uncaught exception will do that naturally.

    # Optional: Local Test Mode (uncomment to use and comment out runpod.serverless.start above)
    # To use this, you would need to provide a valid base64 encoded image string
    # For example, you can convert a small image to base64 online and paste it here.
    # import asyncio # No longer needed if generate_video is synchronous.
    # print("\n--- Running Local Test ---", flush=True)
    # try:
    #     # Example: Replace with actual base64 image content for testing
    #     # with open("path/to/your/local_test_image.jpg", "rb") as image_file:
    #     #     base64_test_image = base64.b64encode(image_file.read()).decode('utf-8')
    #     base64_test_image = "YOUR_BASE64_IMAGE_STRING_HERE"
    #
    #     fake_job = {
    #         "input": {
    #             "init_image": base64_test_image,
    #             "prompt": "a couple kissing under the moonlight, cinematic, romantic",
    #             "num_frames": 16,
    #             "fps": 8,
    #             "ip_adapter_scale": 0.8,
    #             "openpose_scale": 1.2,
    #             "depth_scale": 0.6
    #         }
    #     }
    #
    #     result = generate_video(fake_job) # Call synchronously for local testing
    #     print("Local test result:", result, flush=True)
    # except Exception as e:
    #     print(f"Local test failed: {traceback.format_exc()}", flush=True)