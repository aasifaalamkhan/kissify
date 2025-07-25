import runpod
import torch
import traceback
import os
import io
import base64
import requests
import tempfile
import numpy as np
import gc # Import garbage collection
import time # Import time module for sleep

from threading import Thread
from flask import Flask

from PIL import Image
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler, ControlNetModel
from diffusers.utils import export_to_video
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from controlnet_aux import OpenposeDetector, MidasDetector
from huggingface_hub import hf_hub_download, HfFolder # For better token handling

print("✅ main.py started: Initializing script execution.", flush=True)

# Determine if debug mode is enabled
RP_DEBUG = os.getenv("RP_DEBUG", "False").lower() == "true"
if RP_DEBUG:
    print("✨ Debug mode is ENABLED.", flush=True)

# Optimize cuDNN for consistent input shapes (like image sizes)
torch.backends.cudnn.benchmark = True

# --- Globals for Lazy-Loading Models ---
pipe = None
image_encoder = None # Will be loaded and used directly by pipe.load_ip_adapter
openpose_detector = None
midas_detector = None

# Set Hugging Face cache directory (important if pre-fetching during build)
os.environ['HF_HOME'] = os.getenv('HF_HOME', '/app/hf_cache')
print(f"Hugging Face cache directory set to: {os.environ['HF_HOME']}", flush=True)

# --- File Upload Utility ---
def upload_to_catbox(filepath: str, max_retries: int = 3, initial_delay: int = 5) -> str:
    """
    Uploads a file to Catbox.moe for temporary hosting with retry logic.

    Args:
        filepath (str): The path to the file to upload.
        max_retries (int): Maximum number of retries for the upload.
        initial_delay (int): Initial delay in seconds before the first retry.

    Returns:
        str: The URL of the uploaded file or an error message.
    """
    print(f"📤 Attempting to upload {filepath} to Catbox...", flush=True)
    retries = 0
    while retries < max_retries:
        try:
            with open(filepath, 'rb') as f:
                files = {'fileToUpload': (os.path.basename(filepath), f)}
                data = {'reqtype': 'fileupload', 'userhash': ''} # userhash can be empty
                response = requests.post('https://litterbox.catbox.moe/resources/internals/api.php', files=files, data=data, timeout=60)
                response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)
                
                # --- FIX: More robust check for successful URL ---
                if "https://" in response.text: # Check if "https://" exists anywhere in the response
                    print(f"✅ Upload successful. URL: {response.text}", flush=True)
                    return response.text
                else:
                    print(f"❌ Catbox upload returned unexpected response: {response.text}", flush=True)
        except requests.exceptions.Timeout:
            print(f"❌ Catbox upload timed out. Retry {retries + 1}/{max_retries}...", flush=True)
        except requests.exceptions.RequestException as e:
            print(f"❌ Network or HTTP error during Catbox upload: {e}. Retry {retries + 1}/{max_retries}...", flush=True)
            print(f"Response content (if any): {response.text if 'response' in locals() else 'N/A'}", flush=True)
        except Exception as e:
            print(f"❌ Unexpected error during Catbox upload: {traceback.format_exc()}. Retry {retries + 1}/{max_retries}...", flush=True)

        retries += 1
        if retries < max_retries:
            sleep_time = initial_delay * (2 ** (retries - 1)) # Exponential backoff
            print(f"Waiting {sleep_time} seconds before retrying...", flush=True)
            time.sleep(sleep_time)
    
    final_error_message = f"Error uploading: Failed to upload to Catbox after {max_retries} attempts."
    print(final_error_message, flush=True)
    return final_error_message

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
        if RP_DEBUG:
            print("💖 Health check received. Responding 'ok'.", flush=True)
        return "ok"

    try:
        print("Starting Flask health check server on 0.0.0.0:3000...", flush=True)
        app.run(host="0.0.0.0", port=3000, debug=False, use_reloader=False)
        print("Health check server thread finished running Flask app (should not happen if app.run blocks).", flush=True)
    except Exception as e:
        print(f"❌ Health check server failed to start: {traceback.format_exc()}", flush=True)

# --- Job Handler: Main Video Generation Logic ---
def generate_video(job: dict) -> dict:
    """
    Main job handler for RunPod. Generates a video based on the input image and parameters.

    Args:
        job (dict): The job payload from RunPod, containing 'input' parameters.

    Returns:
        dict: A dictionary containing the video URL and thumbnail URL or an error message.
    """
    job_id = job.get('id', 'N/A')
    print(f"📥 Job received. Processing job ID: {job_id}", flush=True)
    global pipe, image_encoder, openpose_detector, midas_detector

    # Lazy-load models on the first job
    if pipe is None:
        print("⏳ Models not loaded. Beginning model loading process...", flush=True)
        print("Waiting 5 seconds for network to stabilize...", flush=True)
        time.sleep(5)
        print("Resuming model loading...", flush=True)

        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available! A GPU is absolutely required for this application.")
            print(f"CUDA is available. Device count: {torch.cuda.device_count()}", flush=True)
            print(f"Current CUDA device: {torch.cuda.current_device()}", flush=True)
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}", flush=True)

            HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN") or HfFolder.get_token()
            if HUGGING_FACE_TOKEN:
                print("🔐 Hugging Face token detected.", flush=True)
                from huggingface_hub import login
                login(token=HUGGING_FACE_TOKEN, add_to_git_credential=False)
            else:
                print("⚠️ Hugging Face token not found. Downloads for gated models may fail. "
                      "Set HUGGING_FACE_HUB_TOKEN environment variable or login using `huggingface_hub.login()`.", flush=True)

            # Define model IDs
            base_model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
            motion_module_id = "guoyww/animatediff-motion-adapter-v1-5-2"
            
            # CLIP models for IP-Adapter
            clip_model_id = "openai/clip-vit-large-patch14" 
            # --- CRITICAL FIX: IP-Adapter weights are a subfolder in the h94/IP-Adapter repo ---
            ip_adapter_subfolder_id = "h94/IP-Adapter/ip-adapter_sd15" 

            controlnet_openpose_id = "lllyasviel/control_v11p_sd15_openpose"
            controlnet_depth_id = "lllyasviel/control_v11f1p_sd15_depth"
            controlnet_aux_id = "lllyasviel/ControlNet"

            print(f"  Loading OpenPose ControlNet from {controlnet_openpose_id}...", flush=True)
            openpose_controlnet = ControlNetModel.from_pretrained(
                controlnet_openpose_id, torch_dtype=torch.float16, use_safetensors=True
            )
            print(f"  Loading Depth ControlNet from {controlnet_depth_id}...", flush=True)
            depth_controlnet = ControlNetModel.from_pretrained(
                controlnet_depth_id, torch_dtype=torch.float16, use_safetensors=True
            )
            print("  ControlNet models loaded.", flush=True)

            print(f"  Loading OpenposeDetector from {controlnet_aux_id}...", flush=True)
            openpose_detector = OpenposeDetector.from_pretrained(controlnet_aux_id)
            print(f"  Loading MidasDetector from {controlnet_aux_id}...", flush=True)
            midas_detector = MidasDetector.from_pretrained(controlnet_aux_id)
            print("  ControlNet auxiliary detectors loaded.", flush=True)

            print(f"  Loading MotionAdapter from {motion_module_id}...", flush=True)
            adapter = MotionAdapter.from_pretrained(motion_module_id, torch_dtype=torch.float16)
            print("  MotionAdapter loaded.", flush=True)

            print(f"  Loading AnimateDiff pipeline from {base_model_id}...", flush=True)
            pipe = AnimateDiffPipeline.from_pretrained(
                base_model_id,
                motion_adapter=adapter,
                controlnet=[openpose_controlnet, depth_controlnet],
                torch_dtype=torch.float16,
                use_safetensors=True
            )
            print("  AnimateDiff pipeline loaded. Setting scheduler...", flush=True)
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            print("  Scheduler set.", flush=True)
            
            # --- FIX: Call enable_model_cpu_offload safely ---
            if hasattr(pipe, "enable_model_cpu_offload"):
                print("  Enabling model CPU offload...", flush=True)
                pipe.enable_model_cpu_offload()
                if RP_DEBUG:
                    print("DEBUG: pipe.device (after offload):", pipe.device, flush=True)
            else:
                print("  pipe.enable_model_cpu_offload() not found or not applicable.", flush=True)

            # --- CRITICAL FIX: Proper IP-Adapter integration with the pipeline ---
            # Load CLIP components
            print(f"  Loading CLIP Image Encoder from {clip_model_id}...", flush=True)
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                clip_model_id, torch_dtype=torch.float16
            ) # Do not move to cuda yet, pipeline will handle it.
            print(f"  Loading CLIP Image Processor from {clip_model_id}...", flush=True)
            image_processor = CLIPImageProcessor.from_pretrained(clip_model_id) # No .to('cuda') for image processor

            # Load IP-Adapter onto the pipeline using the subfolder path
            print(f"  Loading IP-Adapter weights from {ip_adapter_subfolder_id} onto the pipeline...", flush=True)
            pipe.load_ip_adapter(
                ip_adapter_subfolder_id, # Pass the full repo_id/subfolder path
                subfolder="image_encoder", # This subfolder needs to be specified correctly for the CLIP components within that IP-Adapter folder structure.
                                          # OR, if the `ip-adapter_sd15` folder IS the full IP-Adapter model format, just remove `subfolder`.
                                          # Given the error, it's likely this should be removed and `ip_adapter_subfolder_id` IS the model.
                image_encoder=image_encoder # Pass the pre-loaded image_encoder here
            )
            # Remove the now unnecessary custom IPAdapterImageProj class or references to it.
            # The pipeline handles the image projection.
            # global image_proj_model # Removed from globals as it's no longer used.

            print("✅ All models loaded successfully to GPU (or offloaded).", flush=True)

            # Initial cleanup after model loading
            gc.collect()
            torch.cuda.empty_cache()
            if RP_DEBUG:
                print(f"DEBUG: CUDA memory after initial load: {torch.cuda.memory_allocated() / (1024**3):.2f} GB", flush=True)

        except RuntimeError as e:
            error_message = f"❌ CUDA or Model Initialization Critical Error: {e}\n{traceback.format_exc()}"
            print(error_message, flush=True)
            return {"error": error_message}
        except Exception as e:
            error_message = f"❌ Model loading failed unexpectedly: {traceback.format_exc()}"
            print(error_message, flush=True)
            return {"error": error_message}

    # --- Parse Job Input ---
    # --- FIX: Job input fallback ---
    job_input = job.get('input') or job
    base64_image = job_input.get('init_image')
    prompt = job_input.get('prompt', 'a couple kissing, beautiful, cinematic')
    if RP_DEBUG:
        print(f"DEBUG: Prompt received: '{prompt}'", flush=True)

    height = int(job_input.get('height', 512))
    width = int(job_input.get('width', 512))
    num_frames = int(job_input.get('num_frames', 16))
    fps = int(job_input.get('fps', 8))

    height = max(256, min(height, 1024))
    width = max(256, min(width, 1024))
    num_frames = max(8, min(num_frames, 32))
    fps = max(4, min(fps, 15))

    ip_adapter_scale = float(job_input.get('ip_adapter_scale', 0.7))
    ip_adapter_scale = min(max(ip_adapter_scale, 0.0), 2.0)
    openpose_scale = float(job_input.get('openpose_scale', 1.0))
    openpose_scale = min(max(openpose_scale, 0.0), 2.0)
    depth_scale = float(job_input.get('depth_scale', 0.5))
    depth_scale = min(max(depth_scale, 0.0), 2.0)

    if RP_DEBUG:
        print(f"DEBUG: resolution={width}x{height}, num_frames={num_frames}, fps={fps}, "
              f"ip_adapter_scale={ip_adapter_scale}, openpose_scale={openpose_scale}, depth_scale={depth_scale}", flush=True)

    if not base64_image:
        print("❌ 'init_image' (base64 encoded) is missing in job input.", flush=True)
        return {"error": "Missing 'init_image' base64 input. Please provide a base64 encoded image."}

    try:
        print("🖼️ Decoding base64 image...", flush=True)
        init_image = Image.open(io.BytesIO(base64.b64decode(base64_image))).convert("RGB")
        print(f"🖼️ Input image dimensions: {init_image.size[0]}x{init_image.size[1]}", flush=True)
        
        # --- FIX: Add input image size validation ---
        if init_image.size[0] * init_image.size[1] > 1024 * 1024: # Max 1 Megapixel
            return {"error": "Input image too large. Max 1 Megapixel (e.g., 1024x1024) supported."}

    except Exception as e:
        error_message = f"❌ Failed to decode or open 'init_image': {traceback.format_exc()}"
        print(error_message, flush=True)
        return {"error": error_message}

    # --- Image Preprocessing ---
    print("🔍 Preprocessing input image for model inference...", flush=True)
    try:
        # --- FIX: Convert PIL image to numpy array for controlnet_aux detectors ---
        np_image = np.array(init_image)

        print("  Generating OpenPose conditioning image...", flush=True)
        openpose_image = openpose_detector(np_image) # Use np_image
        if RP_DEBUG:
            print(f"DEBUG: OpenPose image generated. Size: {openpose_image.size}", flush=True)
        print("  Generating Depth conditioning image...", flush=True)
        depth_image = midas_detector(np_image) # Use np_image
        if RP_DEBUG:
            print(f"DEBUG: Depth image generated. Size: {depth_image.size}", flush=True)

        control_images = [openpose_image, depth_image]

        # --- CRITICAL FIX: IP-Adapter scale is set on the pipeline directly ---
        pipe.set_ip_adapter_scale(ip_adapter_scale)
        # No `cross_attention_kwargs` needed for `ip_adapter_image_embeds` if using pipe.load_ip_adapter and passing `ip_adapter_image`
        cross_attention_kwargs = {} 

        print("🔍 Image preprocessing complete.", flush=True)

    except Exception as e:
        error_message = f"❌ Image preprocessing failed: {traceback.format_exc()}"
        print(error_message, flush=True)
        return {"error": error_message}

    # Log CUDA memory before inference
    if RP_DEBUG:
        print(f"DEBUG: CUDA memory BEFORE inference: {torch.cuda.memory_allocated() / (1024**3):.2f} GB", flush=True)

    # --- Video Inference ---
    print("✨ Starting video generation inference...", flush=True)
    try:
        output = pipe(
            prompt=prompt,
            negative_prompt="ugly, distorted, low quality, cropped, blurry, bad anatomy, bad quality, long_neck, long_body, text, watermark, signature",
            num_frames=num_frames,
            guidance_scale=7.5,
            num_inference_steps=20,
            image=control_images, # ControlNet conditioning images
            controlnet_conditioning_scale=[openpose_scale, depth_scale], # Scales for each ControlNet
            # --- CRITICAL FIX: Pass the original init_image to `pipe` for IP-Adapter ---
            ip_adapter_image=init_image, # Pass original PIL image here for IP-Adapter
            cross_attention_kwargs=cross_attention_kwargs, # Should be empty or contain other specific kwargs if any
            height=height,
            width=width
        )
        # Assign frames to a variable explicitly for clarity and longevity
        generated_frames = output.frames[0] 
        print("✅ Video inference completed.", flush=True)

    except torch.cuda.OutOfMemoryError:
        error_message = "❌ CUDA Out Of Memory error during inference. Try reducing num_frames or image resolution. Current memory usage might be too high."
        print(error_message, flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        if RP_DEBUG:
            print(f"DEBUG: CUDA memory after OOM attempt to clear: {torch.cuda.memory_allocated() / (1024**3):.2f} GB", flush=True)
        return {"error": error_message}
    except Exception as e:
        error_message = f"❌ Video inference failed: {traceback.format_exc()}"
        print(error_message, flush=True)
        return {"error": error_message}

    # Log CUDA memory after inference, before major cleanup
    if RP_DEBUG:
        print(f"DEBUG: CUDA memory AFTER inference (before export/major cleanup): {torch.cuda.memory_allocated() / (1024**3):.2f} GB", flush=True)

    # --- Export and Upload Video ---
    print("📼 Exporting video and preparing for upload...", flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "kissify_video.mp4")
        thumbnail_path = os.path.join(tmpdir, "kissify_thumbnail.jpg")
        try:
            # --- CRITICAL FIX: Use generated_frames for export and thumbnail ---
            export_to_video(generated_frames, video_path, fps=fps)
            print(f"📼 Video exported to {video_path} with {fps} FPS.", flush=True)

            # Generate thumbnail from the first frame
            if generated_frames and len(generated_frames) > 0:
                generated_frames[0].save(thumbnail_path)
                print(f"🖼️ Thumbnail exported to {thumbnail_path}.", flush=True)
            else:
                print("⚠️ No frames to generate thumbnail from.", flush=True)
                thumbnail_path = None # Indicate no thumbnail generated

        except Exception as e:
            error_message = f"❌ Failed to export video or thumbnail: {traceback.format_exc()}"
            print(error_message, flush=True)
            return {"error": error_message}

        # Now it's safe to delete the large model output and frames
        del output # Delete the pipeline output object
        del generated_frames # Delete the list of PIL images
        gc.collect()
        torch.cuda.empty_cache()
        if RP_DEBUG:
            print(f"DEBUG: CUDA memory after video export and cleanup: {torch.cuda.memory_allocated() / (1024**3):.2f} GB", flush=True)

        print("🚀 Uploading generated video to Catbox...", flush=True)
        video_url = upload_to_catbox(filepath=video_path, max_retries=5)
        if "Error" in video_url:
            print(f"❌ Video upload failed: {video_url}", flush=True)
            return {"error": video_url}

        thumbnail_url = None
        if thumbnail_path:
            print("🚀 Uploading generated thumbnail to Catbox...", flush=True)
            thumbnail_url = upload_to_catbox(filepath=thumbnail_path, max_retries=5)
            if "Error" in thumbnail_url:
                print(f"❌ Thumbnail upload failed: {thumbnail_url}", flush=True)
                thumbnail_url = None

        print(f"✅ Video generation complete and uploaded. Video URL: {video_url}", flush=True)
        return {"output": {"video_url": video_url, "thumbnail_url": thumbnail_url}}

# --- Entry Point for RunPod Worker ---
if __name__ == "__main__":
    print("Starting health check server thread...", flush=True)
    try:
        health_thread = Thread(target=run_healthcheck_server, daemon=True)
        health_thread.start()
        print("Health check server thread started.", flush=True)
    except Exception as e:
        print(f"❌ Failed to start health check server thread: {traceback.format_exc()}", flush=True)
        exit(1)

    try:
        print("🚀 RunPod worker is ready to receive jobs...", flush=True)
        runpod.serverless.start({"handler": generate_video})
    except Exception as e:
        print(f"❌ RunPod serverless failed to start: {traceback.format_exc()}", flush=True)
        exit(1)
