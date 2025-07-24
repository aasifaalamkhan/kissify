import runpod
import torch
import traceback
import os
import io
import base64
import requests
import tempfile
import numpy as np

from threading import Thread
from flask import Flask

from PIL import Image
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler, ControlNetModel
from diffusers.utils import export_to_video
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from controlnet_aux import OpenposeDetector, MidasDetector
from huggingface_hub import hf_hub_download

print("✅ main.py started: Initializing script execution.", flush=True)

# Optimize cuDNN for consistent input shapes (like image sizes)
torch.backends.cudnn.benchmark = True

# --- IP-Adapter Helper Class ---
class IPAdapterImageProj(torch.nn.Module):
    """
    A simple linear layer to project image embeddings for the IP-Adapter.
    """
    def __init__(self, state_dict):
        super().__init__()
        # Initialize a linear layer based on the shapes found in the state_dict
        self.image_proj_model = torch.nn.Linear(
            state_dict["image_proj"].shape[-1], state_dict["ip_adapter"].shape[0]
        )
        # Load the weights for the linear layer
        self.image_proj_model.load_state_dict(
            {"weight": state_dict["image_proj"], "bias": torch.zeros(state_dict["image_proj"].shape[0])}
        )

    def forward(self, image_embeds):
        """
        Forward pass for the image projection model.
        """
        return self.image_proj_model(image_embeds)

# --- Globals for Lazy-Loading Models ---
# These variables will hold our models and processors,
# initialized only when the first job arrives.
pipe = None
image_encoder = None
image_proj_model = None
image_processor = None
openpose_detector = None
midas_detector = None

# --- File Upload Utility ---
def upload_to_catbox(filepath: str) -> str:
    """
    Uploads a file to Catbox.moe for temporary hosting.

    Args:
        filepath (str): The path to the file to upload.

    Returns:
        str: The URL of the uploaded file or an error message.
    """
    try:
        with open(filepath, 'rb') as f:
            files = {'fileToUpload': (os.path.basename(filepath), f)}
            data = {'reqtype': 'fileupload'}
            response = requests.post('https://litterbox.catbox.moe/resources/internals/api.php', files=files, data=data)
            response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)
            return response.text
    except requests.exceptions.RequestException as e:
        print(f"❌ Network or HTTP error during Catbox upload: {e}", flush=True)
        return f"Error uploading: Network or HTTP issue."
    except Exception as e:
        print(f"❌ Unexpected error uploading to Catbox: {traceback.format_exc()}", flush=True)
        return f"Error uploading: {str(e)}"

# --- Health Check Server ---
def run_healthcheck_server():
    """
    Starts a Flask web server to respond to health check requests.
    This ensures RunPod knows the worker is alive and responsive.
    """
    app = Flask(__name__)

    @app.route('/healthz')
    def health():
        """Responds with 'ok' to health checks."""
        return "ok"

    # Run the Flask app, accessible from outside the container on port 3000.
    # debug=False and use_reloader=False are important for production deployment.
    app.run(host="0.0.0.0", port=3000, debug=False, use_reloader=False)

# --- Job Handler: Main Video Generation Logic ---
def generate_video(job: dict) -> dict:
    """
    Main job handler for RunPod. Generates a video based on the input image and parameters.

    Args:
        job (dict): The job payload from RunPod, containing 'input' parameters.

    Returns:
        dict: A dictionary containing the video URL or an error message.
    """
    print(f"📥 Job received. Processing job ID: {job.get('id', 'N/A')}", flush=True)
    global pipe, image_encoder, image_proj_model, image_processor, openpose_detector, midas_detector

    # Lazy-load models on the first job
    if pipe is None:
        print("⏳ Models not loaded. Beginning model loading process...", flush=True)
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available! A GPU is absolutely required for this application.")

            # Retrieve Hugging Face token from environment variables
            HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")
            if HUGGING_FACE_TOKEN:
                print("🔐 Hugging Face token detected.", flush=True)
            else:
                print("⚠️ Hugging Face token not found. Downloads for gated models may fail. "
                      "Set HUGGING_FACE_HUB_TOKEN environment variable.", flush=True)

            # Define model IDs
            base_model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
            motion_module_id = "guoyww/animatediff-motion-module-v3"
            ip_adapter_repo_id = "h94/IP-Adapter"

            # Load ControlNet models
            print("  Loading ControlNet models...", flush=True)
            openpose_controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16, token=HUGGING_FACE_TOKEN
            )
            depth_controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16, token=HUGGING_FACE_TOKEN
            )

            # Load ControlNet auxiliary detectors
            print("  Loading ControlNet auxiliary detectors...", flush=True)
            openpose_detector = OpenposeDetector.from_pretrained(
                "lllyasviel/ControlNet", token=HUGGING_FACE_TOKEN
            )
            midas_detector = MidasDetector.from_pretrained(
                "lllyasviel/ControlNet", token=HUGGING_FACE_TOKEN
            )

            # Load AnimateDiff pipeline
            print("  Loading AnimateDiff pipeline...", flush=True)
            pipe = AnimateDiffPipeline.from_pretrained(
                base_model_id,
                controlnet=[openpose_controlnet, depth_controlnet],
                torch_dtype=torch.float16,
                token=HUGGING_FACE_TOKEN
            )
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            pipe.load_motion_module(
                motion_module_id, unet_additional_kwargs={"use_inflated_groupnorm": True}, token=HUGGING_FACE_TOKEN
            )
            # Offload models to CPU when not in use to save GPU memory
            pipe.enable_model_cpu_offload()

            # Load IP-Adapter components
            print("  Loading IP-Adapter components...", flush=True)
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                ip_adapter_repo_id, subfolder="models/image_encoder", torch_dtype=torch.float16, token=HUGGING_FACE_TOKEN
            ).to("cuda") # Ensure image encoder is on CUDA
            image_processor = CLIPImageProcessor.from_pretrained(
                ip_adapter_repo_id, subfolder="models/image_encoder", token=HUGGING_FACE_TOKEN
            )
            ip_adapter_path = hf_hub_download(
                repo_id=ip_adapter_repo_id, filename="ip-adapter_sd15.bin", token=HUGGING_FACE_TOKEN
            )
            ip_adapter_weights = torch.load(ip_adapter_path, map_location="cpu")
            image_proj_model = IPAdapterImageProj(ip_adapter_weights).to("cuda") # Ensure projection model is on CUDA

            print("✅ All models loaded successfully to GPU.", flush=True)

        except Exception as e:
            error_message = f"❌ Model load failed: {traceback.format_exc()}"
            print(error_message, flush=True)
            return {"error": error_message}

    # --- Parse Job Input ---
    job_input = job.get('input', {})
    base64_image = job_input.get('init_image')
    prompt = job_input.get('prompt', 'a couple kissing, beautiful, cinematic')

    # Parameters with default values and validation
    num_frames = int(job_input.get('num_frames', 16))
    fps = int(job_input.get('fps', 8))
    # Ensure scales are within reasonable bounds (0.0 to 1.5)
    ip_adapter_scale = float(job_input.get('ip_adapter_scale', 0.7))
    ip_adapter_scale = min(max(ip_adapter_scale, 0.0), 1.5)
    openpose_scale = float(job_input.get('openpose_scale', 1.0))
    openpose_scale = min(max(openpose_scale, 0.0), 1.5)
    depth_scale = float(job_input.get('depth_scale', 0.5))
    depth_scale = min(max(depth_scale, 0.0), 1.5)

    if not base64_image:
        print("❌ 'init_image' (base64 encoded) is missing in job input.", flush=True)
        return {"error": "Missing 'init_image' base64 input. Please provide a base64 encoded image."}

    try:
        # Decode base64 image and convert to RGB
        init_image = Image.open(io.BytesIO(base64.b64decode(base64_image))).convert("RGB")
    except Exception as e:
        error_message = f"❌ Failed to decode or open 'init_image': {traceback.format_exc()}"
        print(error_message, flush=True)
        return {"error": error_message}

    # --- Image Preprocessing ---
    print("🔍 Preprocessing input image for model inference...", flush=True)
    try:
        # Process image for CLIP (IP-Adapter)
        processed_image = image_processor(images=init_image, return_tensors="pt").pixel_values.to("cuda", dtype=torch.float16)
        clip_features = image_encoder(processed_image).image_embeds
        image_embeds = image_proj_model(clip_features)

        # Generate ControlNet conditioning images
        openpose_image = openpose_detector(init_image)
        depth_image = midas_detector(init_image)
        control_images = [openpose_image, depth_image]

        cross_attention_kwargs = {"ip_adapter_image_embeds": image_embeds, "scale": ip_adapter_scale}

    except Exception as e:
        error_message = f"❌ Image preprocessing failed: {traceback.format_exc()}"
        print(error_message, flush=True)
        return {"error": error_message}

    # --- Video Inference ---
    print("✨ Starting video generation inference...", flush=True)
    try:
        output = pipe(
            prompt=prompt,
            negative_prompt="ugly, distorted, low quality, cropped, blurry, bad anatomy, bad quality",
            num_frames=num_frames,
            guidance_scale=7.5,
            num_inference_steps=20,
            image=control_images, # ControlNet conditioning images
            controlnet_conditioning_scale=[openpose_scale, depth_scale], # Scales for each ControlNet
            cross_attention_kwargs=cross_attention_kwargs # IP-Adapter conditioning
        )
        frames = output.frames[0] # Assuming we want the first (and likely only) generated video

    except torch.cuda.OutOfMemoryError:
        error_message = "❌ CUDA Out Of Memory error during inference. Try reducing num_frames or image resolution."
        print(error_message, flush=True)
        return {"error": error_message}
    except Exception as e:
        error_message = f"❌ Video inference failed: {traceback.format_exc()}"
        print(error_message, flush=True)
        return {"error": error_message}

    # --- Export and Upload Video ---
    print("📼 Exporting video and preparing for upload...", flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "kissify_video.mp4")
        try:
            export_to_video(frames, video_path, fps=fps)
        except Exception as e:
            error_message = f"❌ Failed to export video: {traceback.format_exc()}"
            print(error_message, flush=True)
            return {"error": error_message}

        print("🚀 Uploading generated video to Catbox...", flush=True)
        video_url = upload_to_catbox(filepath=video_path)
        if "Error" in video_url:
            return {"error": video_url}

        print("✅ Video generation complete and uploaded.", flush=True)
        return {"output": {"video_url": video_url}}

# --- Entry Point for RunPod Worker ---
if __name__ == "__main__":
    # Start the health check server in a separate daemon thread.
    # A daemon thread will automatically exit when the main program exits.
    # This is critical for RunPod's health monitoring.
    print("Starting health check server thread...", flush=True)
    Thread(target=run_healthcheck_server, daemon=True).start()

    try:
        print("🚀 RunPod worker is ready to receive jobs...", flush=True)
        # Start the RunPod serverless worker. The 'handler' function will be called
        # for each incoming job.
        runpod.serverless.start({"handler": generate_video})
    except Exception as e:
        # If RunPod serverless fails to start, log the error.
        # RunPod's platform monitoring will detect this failure.
        print(f"❌ RunPod serverless failed to start: {traceback.format_exc()}", flush=True)
        # It's good practice to let the container exit if the main service fails to initialize.
        # An uncaught exception here will cause the container to exit with a non-zero code.

    # --- Local Test Mode (Optional) ---
    # Uncomment the following block for local testing without RunPod.
    # Requires a base64 encoded image string for `base64_test_image`.
    # Make sure to comment out `runpod.serverless.start` if testing locally.

    # print("\n--- Running Local Test Mode (if uncommented) ---", flush=True)
    # try:
    #     # Replace with a real base64 image string for testing!
    #     # Example: base64_test_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=" # A tiny black pixel
    #     # To generate a base64 string from a file:
    #     # import base64
    #     # with open("path/to/your/image.jpg", "rb") as image_file:
    #     #     base64_test_image = base64.b64encode(image_file.read()).decode('utf-8')
    #
    #     # base64_test_image = "YOUR_BASE64_IMAGE_STRING_HERE" # <<< IMPORTANT: REPLACE THIS
    #
    #     # if base64_test_image == "YOUR_BASE64_IMAGE_STRING_HERE":
    #     #     print("⚠️ WARNING: Please replace 'YOUR_BASE64_IMAGE_STRING_HERE' with a valid base64 image for local testing.")
    #     # else:
    #     #     fake_job = {
    #     #         "id": "local-test-job",
    #     #         "input": {
    #     #             "init_image": base64_test_image,
    #     #             "prompt": "a couple kissing under the moonlight, cinematic, romantic",
    #     #             "num_frames": 16,
    #     #             "fps": 8,
    #     #             "ip_adapter_scale": 0.8,
    #     #             "openpose_scale": 1.2,
    #     #             "depth_scale": 0.6
    #     #         }
    #     #     }
    #     #     print("\n--- Starting local test job ---", flush=True)
    #     #     result = generate_video(fake_job) # Call synchronously for local testing
    #     #     print("Local test result:", result, flush=True)
    #
    # except Exception as e:
    #     print(f"❌ Local test failed: {traceback.format_exc()}", flush=True)