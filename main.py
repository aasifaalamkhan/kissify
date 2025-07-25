import runpod
import torch
import traceback
import os
import io
import base64
import requests
import tempfile
import numpy as np
import gc
import time

from threading import Thread, Lock
from flask import Flask

from PIL import Image
import diffusers
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler, ControlNetModel
from diffusers.utils import export_to_video
from transformers import CLIPVisionModelWithProjection # CLIPImageProcessor removed as per previous suggestion
from controlnet_aux import OpenposeDetector, MidasDetector
# from huggingface_hub import HfFolder, hf_hub_download, login as hf_login # Removed hf_login for runtime as pre-download handles it

print("✅ main.py started: Initializing script execution.", flush=True)

# Determine if debug mode is enabled
RP_DEBUG = os.getenv("RP_DEBUG", "False").lower() == "true"
if RP_DEBUG:
    print("✨ Debug mode is ENABLED.", flush=True)

# --- Log diffusers version for reproducibility ---
print(f"Diffusers version: {diffusers.__version__}", flush=True)

# Optimize cuDNN for consistent input shapes (like image sizes)
torch.backends.cudnn.benchmark = True

# --- Globals for Lazy-Loading Models ---
pipe = None
image_encoder = None # This global will store the CLIPVisionModelWithProjection
openpose_detector = None
midas_detector = None

# --- Thread-safe lazy-loading lock ---
model_load_lock = Lock()

# Set Hugging Face cache directory (important: must match Dockerfile's HF_HOME)
# This ENV VAR will be set by the Dockerfile, so this just confirms it.
os.environ['HF_HOME'] = os.getenv('HF_HOME', '/app/hf_cache')
print(f"Hugging Face cache directory set to: {os.environ['HF_HOME']}", flush=True)


# --- File Upload Utility ---
def upload_to_catbox(filepath: str, max_retries: int = 3, initial_delay: int = 5) -> dict:
    """
    Uploads a file to Catbox.moe for temporary hosting with retry logic.

    Args:
        filepath (str): The path to the file to upload.
        max_retries (int): Maximum number of retries for the upload.
        initial_delay (int): Initial delay in seconds before the first retry.

    Returns:
        dict: A dictionary containing 'url' on success or 'error' message on failure.
    """
    print(f"📤 Attempting to upload {filepath} to Catbox...", flush=True)
    retries = 0
    while retries < max_retries:
        try:
            with open(filepath, 'rb') as f:
                files = {'fileToUpload': (os.path.basename(filepath), f)}
                data = {'reqtype': 'fileupload', 'userhash': ''}
                response = requests.post('https://litterbox.catbox.moe/resources/internals/api.php', files=files, data=data, timeout=60)
                response.raise_for_status()

                if response.text.startswith("https://") and len(response.text.strip()) > len("https://"):
                    print(f"✅ Upload successful. URL: {response.text}", flush=True)
                    return {"url": response.text.strip()}
                else:
                    print(f"❌ Catbox upload returned unexpected response (Status: {response.status_code}): '{response.text.strip()}'. Retry {retries + 1}/{max_retries}...", flush=True)
        except requests.exceptions.Timeout:
            print(f"❌ Catbox upload timed out. Retry {retries + 1}/{max_retries}...", flush=True)
        except requests.exceptions.RequestException as e:
            response_content = response.text if 'response' in locals() and response.text else 'N/A'
            print(f"❌ Network or HTTP error during Catbox upload: {e}. Response: '{response_content.strip()}'. Retry {retries + 1}/{max_retries}...", flush=True)
        except Exception as e:
            print(f"❌ Unexpected error during Catbox upload: {traceback.format_exc()}. Retry {retries + 1}/{max_retries}...", flush=True)

        retries += 1
        if retries < max_retries:
            sleep_time = initial_delay * (2 ** (retries - 1))
            print(f"Waiting {sleep_time} seconds before retrying...", flush=True)
            time.sleep(sleep_time)
            
    final_error_message = f"Error uploading: Failed to upload {os.path.basename(filepath)} to Catbox after {max_retries} attempts."
    print(final_error_message, flush=True)
    return {"error": final_error_message}

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

    # --- Thread-safe lazy-loading ---
    with model_load_lock:
        if pipe is None:
            print("⏳ Models not loaded. Beginning model loading from local cache...", flush=True)

            try:
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is not available! A GPU is absolutely required for this application.")
                print(f"CUDA is available. Device count: {torch.cuda.device_count()}", flush=True)
                print(f"Current CUDA device: {torch.cuda.current_device()}", flush=True)
                print(f"CUDA device name: {torch.cuda.get_device_name(0)}", flush=True)

                # Define model IDs (used to load from local HF_HOME cache)
                base_model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
                motion_module_id = "guoyww/animatediff-motion-adapter-v1-5-2"
                clip_model_id = "openai/clip-vit-large-patch14"
                ip_adapter_repo_id = "h94/IP-Adapter"
                ip_adapter_model_subfolder = "models"
                ip_adapter_weight_filename = "ip-adapter_sd15.bin"
                controlnet_openpose_id = "lllyasviel/control_v11p_sd15_openpose"
                controlnet_depth_id = "lllyasviel/control_v11f1p_sd15_depth"
                controlnet_aux_repo_id = "lllyasviel/ControlNet"

                # No need for runtime hf_login here as models are pre-downloaded by Dockerfile
                # HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN") or HfFolder.get_token()
                # if HUGGING_FACE_TOKEN:
                #     print("🔐 Hugging Face token detected.", flush=True)
                #     hf_login(token=HUGGING_FACE_TOKEN, add_to_git_credential=False)
                # else:
                #     print("⚠️ Hugging Face token not found. This should be handled during Docker build.", flush=True)


                print(f"  Loading OpenPose ControlNet from {controlnet_openpose_id} (from cache)...", flush=True)
                openpose_controlnet = ControlNetModel.from_pretrained(
                    controlnet_openpose_id, torch_dtype=torch.float16, use_safetensors=True
                )
                print(f"  Loading Depth ControlNet from {controlnet_depth_id} (from cache)...", flush=True)
                depth_controlnet = ControlNetModel.from_pretrained(
                    controlnet_depth_id, torch_dtype=torch.float16, use_safetensors=True
                )
                print("  ControlNet models loaded.", flush=True)

                print(f"  Loading OpenposeDetector from {controlnet_aux_repo_id} (from cache)...", flush=True)
                openpose_detector = OpenposeDetector.from_pretrained(controlnet_aux_repo_id)
                print(f"  Loading MidasDetector from {controlnet_aux_repo_id} (from cache)...", flush=True)
                midas_detector = MidasDetector.from_pretrained(controlnet_aux_repo_id)
                print("  ControlNet auxiliary detectors loaded.", flush=True)

                print(f"  Loading MotionAdapter from {motion_module_id} (from cache)...", flush=True)
                adapter = MotionAdapter.from_pretrained(motion_module_id, torch_dtype=torch.float16)
                print("  MotionAdapter loaded.", flush=True)

                print(f"  Loading AnimateDiff pipeline from {base_model_id} (from cache)...", flush=True)
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
                
                # --- IP-Adapter integration ---
                print(f"  Loading CLIP Image Encoder from {clip_model_id} (from cache)...", flush=True)
                image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    clip_model_id, torch_dtype=torch.float32 # Load initially as float32 for potential precision needs
                )
                
                print(f"  Loading IP-Adapter weights: {ip_adapter_repo_id} (subfolder: {ip_adapter_model_subfolder}, filename: {ip_adapter_weight_filename}) (from cache)...", flush=True)
                pipe.load_ip_adapter(
                    ip_adapter_repo_id,
                    subfolder=ip_adapter_model_subfolder,
                    weight_name=ip_adapter_weight_filename,
                    image_encoder=image_encoder # Pass the pre-loaded image_encoder
                )

                # Move entire pipeline to GPU and float16
                if torch.cuda.is_available():
                    print(f"  Moving pipeline to GPU and float16...", flush=True)
                    pipe.to("cuda", dtype=torch.float16)

                    # Auto-Fix Logic for CLIP float32 Compatibility
                    try:
                        if hasattr(pipe, "image_encoder") and pipe.image_encoder is not None:
                            encoder_cls_name = pipe.image_encoder.__class__.__name__.lower()
                            if "clip" in encoder_cls_name:
                                if pipe.image_encoder.device.type == 'cpu':
                                    pipe.image_encoder.to("cuda", dtype=torch.float32)
                                else:
                                    pipe.image_encoder.to(dtype=torch.float32)
                                print("✅ image_encoder moved to float32 for compatibility (CLIP detected).", flush=True)
                    except Exception as e:
                        print(f"⚠️ Failed to move image_encoder to float32: {e}", flush=True)

                    if RP_DEBUG:
                        print(f"DEBUG: Pipeline UNet device: {pipe.unet.device}, VAE device: {pipe.vae.device}, Text Encoder device: {pipe.text_encoder.device}", flush=True)
                        print(f"DEBUG: Pipeline UNet dtype: {pipe.unet.dtype}, VAE dtype: {pipe.vae.dtype}, Text Encoder dtype: {pipe.text_encoder.dtype}", flush=True)
                        if hasattr(pipe, 'ip_adapter') and pipe.ip_adapter is not None and \
                           hasattr(pipe.ip_adapter, 'ip_adapter_modules') and len(pipe.ip_adapter.ip_adapter_modules) > 0:
                            print(f"DEBUG: IP-Adapter module (first) dtype: {pipe.ip_adapter.ip_adapter_modules[0].dtype}", flush=True)
                        print(f"DEBUG: Entire pipeline moved to GPU and (mostly) float16.", flush=True)
                else:
                    print("DEBUG: CUDA not available, pipeline remains on CPU.", flush=True)

                print("✅ All models loaded successfully from cache.", flush=True)

                gc.collect()
                torch.cuda.empty_cache()
                if RP_DEBUG:
                    print(f"DEBUG: CUDA memory after initial load: {torch.cuda.memory_allocated() / (1024**3):.2f} GB", flush=True)
                    hf_cache_size_bytes = sum(os.path.getsize(os.path.join(dirpath, filename)) for dirpath, dirnames, filenames in os.walk(os.environ['HF_HOME']) for filename in filenames)
                    print(f"DEBUG: Hugging Face cache size: {hf_cache_size_bytes / (1024**3):.2f} GB", flush=True)


            except RuntimeError as e:
                error_message = f"❌ CUDA or Model Initialization Critical Error: {e}\n{traceback.format_exc()}"
                print(error_message, flush=True)
                return {"error": error_message}
            except Exception as e:
                error_message = f"❌ Model loading failed unexpectedly: {traceback.format_exc()}"
                print(error_message, flush=True)
                return {"error": error_message}

    job_input = job.get('input') or job
    if not isinstance(job_input, dict):
        print("❌ Invalid job input structure. Expected a dictionary.", flush=True)
        return {"error": "Invalid job input structure. Expected a dictionary."}

    base64_image = job_input.get('init_image')
    prompt = job_input.get('prompt', 'a couple kissing, beautiful, cinematic')
    if RP_DEBUG:
        print(f"DEBUG: Prompt received: '{prompt}'", flush=True)

    height = int(job_input.get('height', 512))
    width = int(job_input.get('width', 512))
    num_frames = int(job_input.get('num_frames', 16))
    fps = int(job_input.get('fps', 8))

    # Clamp values to reasonable ranges
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
        
        # Add input image size validation
        if init_image.size[0] * init_image.size[1] > 1024 * 1024:
            return {"error": "Input image too large. Max 1 Megapixel (e.g., 1024x1024) supported."}

    except Exception as e:
        error_message = f"❌ Failed to decode or open 'init_image': {traceback.format_exc()}"
        print(error_message, flush=True)
        return {"error": error_message}

    # --- Image Preprocessing ---
    print("🔍 Preprocessing input image for model inference...", flush=True)
    try:
        np_image = np.array(init_image)

        print("  Generating OpenPose conditioning image...", flush=True)
        if torch.cuda.is_available():
            openpose_detector.to("cuda")
        openpose_image = openpose_detector(np_image)
        if torch.cuda.is_available():
            openpose_detector.to("cpu")
        if RP_DEBUG:
            print(f"DEBUG: OpenPose image generated. Type: {type(openpose_image)}, Size: {openpose_image.size[0]}x{openpose_image.size[1]}", flush=True)

        print("  Generating Depth conditioning image...", flush=True)
        if torch.cuda.is_available():
            midas_detector.to("cuda")
        depth_image = midas_detector(np_image)
        if torch.cuda.is_available():
            midas_detector.to("cpu")

        if isinstance(depth_image, np.ndarray):
            if RP_DEBUG:
                print(f"DEBUG: MidasDetector returned numpy array (shape: {depth_image.shape}). Converting to PIL Image.", flush=True)
            depth_image = Image.fromarray(depth_image)
        elif not isinstance(depth_image, Image.Image):
            raise TypeError(f"MidasDetector returned unexpected type: {type(depth_image)}. Expected PIL Image or numpy array.")

        if RP_DEBUG:
            print(f"DEBUG: Depth image generated. Type: {type(depth_image)}, Size: {depth_image.size[0]}x{depth_image.size[1]}", flush=True)
            
        control_images = [openpose_image, depth_image]

        pipe.set_ip_adapter_scale(ip_adapter_scale)
        cross_attention_kwargs = {}
        
        print("🔍 Image preprocessing complete.", flush=True)

    except Exception as e:
        error_message = f"❌ Image preprocessing failed: {traceback.format_exc()}"
        print(error_message, flush=True)
        return {"error": error_message}

    if RP_DEBUG:
        print(f"DEBUG: CUDA memory BEFORE inference: {torch.cuda.memory_allocated() / (1024**3):.2f} GB", flush=True)

    with torch.cuda.amp.autocast(dtype=torch.float16):
        print("✨ Starting video generation inference...", flush=True)
        try:
            output = pipe(
                prompt=prompt,
                negative_prompt="ugly, distorted, low quality, cropped, blurry, bad anatomy, bad quality, long_neck, long_body, text, watermark, signature",
                num_frames=num_frames,
                guidance_scale=7.5,
                num_inference_steps=20,
                control_image=control_images,
                controlnet_conditioning_scale=[openpose_scale, depth_scale],
                ip_adapter_image=init_image,
                cross_attention_kwargs=cross_attention_kwargs,
                height=height,
                width=width
            )
            if not output.frames or len(output.frames) == 0:
                print("❌ No frames were generated by the pipeline.", flush=True)
                return {"error": "No frames were generated by the pipeline. Inference may have failed silently."}
            
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

    if RP_DEBUG:
        print(f"DEBUG: CUDA memory AFTER inference (before export/major cleanup): {torch.cuda.memory_allocated() / (1024**3):.2f} GB", flush=True)

    # --- Export and Upload Video ---
    print("📼 Exporting video and preparing for upload...", flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "kissify_video.mp4")
        thumbnail_path = os.path.join(tmpdir, "kissify_thumbnail.jpg")
        try:
            export_to_video(generated_frames, video_path, fps=fps)
            print(f"📼 Video exported to {video_path} with {fps} FPS.", flush=True)

            if generated_frames and len(generated_frames) > 0:
                generated_frames[0].save(thumbnail_path)
                print(f"🖼️ Thumbnail exported to {thumbnail_path}.", flush=True)
            else:
                print("⚠️ No frames to generate thumbnail from.", flush=True)
                thumbnail_path = None

        except Exception as e:
            error_message = f"❌ Failed to export video or thumbnail: {traceback.format_exc()}"
            print(error_message, flush=True)
            return {"error": error_message}

        del output
        del generated_frames
        gc.collect()
        torch.cuda.empty_cache()
        if RP_DEBUG:
            print(f"DEBUG: CUDA memory after video export and cleanup: {torch.cuda.memory_allocated() / (1024**3):.2f} GB", flush=True)
            print(f"DEBUG: CUDA max memory allocated: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB", flush=True)

        upload_threads = []
        video_url = None
        thumbnail_url = None

        video_upload_result_container = []
        def video_uploader():
            video_upload_result_container.append(upload_to_catbox(filepath=video_path, max_retries=5))
        video_thread = Thread(target=video_uploader)
        upload_threads.append(video_thread)
        print("🚀 Starting video upload in a separate thread...", flush=True)
        video_thread.start()

        thumbnail_upload_result_container = []
        if thumbnail_path:
            def thumbnail_uploader():
                thumbnail_upload_result_container.append(upload_to_catbox(filepath=thumbnail_path, max_retries=5))
            thumbnail_thread = Thread(target=thumbnail_uploader)
            upload_threads.append(thumbnail_thread)
            print("🚀 Starting thumbnail upload in a separate thread...", flush=True)
            thumbnail_thread.start()

        for t in upload_threads:
            t.join()

        if video_upload_result_container:
            video_upload_output = video_upload_result_container[0]
            if "error" in video_upload_output:
                print(f"❌ Video upload failed: {video_upload_output['error']}", flush=True)
                return {"error": video_upload_output['error']}
            else:
                video_url = video_upload_output.get("url")

        if thumbnail_upload_result_container:
            thumbnail_upload_output = thumbnail_upload_result_container[0]
            if "error" in thumbnail_upload_output:
                print(f"❌ Thumbnail upload failed: {thumbnail_upload_output['error']}", flush=True)
                thumbnail_url = None
            else:
                thumbnail_url = thumbnail_upload_output.get("url")

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