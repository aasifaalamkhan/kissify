# download_models.py
import os
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, ControlNetModel
from transformers import CLIPVisionModelWithProjection
from controlnet_aux import OpenposeDetector, MidasDetector
from huggingface_hub import HfFolder, hf_hub_download, login as hf_login

print("🚀 download_models.py started: Pre-downloading models...", flush=True)

# Set Hugging Face cache directory (should match Dockerfile ENV HF_HOME)
os.environ['HF_HOME'] = os.getenv('HF_HOME', '/app/hf_cache')
print(f"Hugging Face cache directory set to: {os.environ['HF_HOME']}", flush=True)

# Log in to Hugging Face if a token is available (for gated models)
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN") or HfFolder.get_token()
if HUGGING_FACE_TOKEN:
    print("🔐 Hugging Face token detected for pre-download.", flush=True)
    hf_login(token=HUGGING_FACE_TOKEN, add_to_git_credential=False)
else:
    print("⚠️ Hugging Face token not found. Pre-downloads for gated models may fail.", flush=True)

# Define model IDs (same as in main.py)
base_model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
motion_module_id = "guoyww/animatediff-motion-adapter-v1-5-2"
clip_model_id = "openai/clip-vit-large-patch14"
ip_adapter_repo_id = "h94/IP-Adapter"
ip_adapter_model_subfolder = "models"
ip_adapter_weight_filename = "ip-adapter_sd15.bin"
controlnet_openpose_id = "lllyasviel/control_v11p_sd15_openpose"
controlnet_depth_id = "lllyasviel/control_v11f1p_sd15_depth"
controlnet_aux_repo_id = "lllyasviel/ControlNet" # For OpenposeDetector and MidasDetector

try:
    # Pre-download all necessary models
    print(f"Downloading base model: {base_model_id}", flush=True)
    AnimateDiffPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, use_safetensors=True)

    print(f"Downloading motion adapter: {motion_module_id}", flush=True)
    MotionAdapter.from_pretrained(motion_module_id, torch_dtype=torch.float16)

    print(f"Downloading OpenPose ControlNet: {controlnet_openpose_id}", flush=True)
    ControlNetModel.from_pretrained(controlnet_openpose_id, torch_dtype=torch.float16, use_safetensors=True)

    print(f"Downloading Depth ControlNet: {controlnet_depth_id}", flush=True)
    ControlNetModel.from_pretrained(controlnet_depth_id, torch_dtype=torch.float16, use_safetensors=True)

    print(f"Downloading CLIP Image Encoder: {clip_model_id}", flush=True)
    # Using float32 during download is safer, conversion can happen at runtime
    CLIPVisionModelWithProjection.from_pretrained(clip_model_id, torch_dtype=torch.float32)

    print(f"Downloading IP-Adapter weights from {ip_adapter_repo_id}/{ip_adapter_model_subfolder}/{ip_adapter_weight_filename}", flush=True)
    # For IP-Adapter weights, direct hf_hub_download is often best
    hf_hub_download(repo_id=ip_adapter_repo_id, subfolder=ip_adapter_model_subfolder, filename=ip_adapter_weight_filename)

    print(f"Downloading OpenposeDetector components from {controlnet_aux_repo_id}", flush=True)
    OpenposeDetector.from_pretrained(controlnet_aux_repo_id)

    print(f"Downloading MidasDetector components from {controlnet_aux_repo_id}", flush=True)
    MidasDetector.from_pretrained(controlnet_aux_repo_id)

    print("✅ All models pre-downloaded successfully.", flush=True)

except Exception as e:
    print(f"❌ Error during model pre-download: {e}", flush=True)
    import traceback
    traceback.print_exc()
    exit(1) # Exit with an error code to fail the build