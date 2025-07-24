import runpod
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_video
from ip_adapter.ip_adapter import IPAdapter # Correct import for the IP-Adapter class
from PIL import Image
import numpy as np
import base64
import io
import requests
import os
import tempfile

pipe = None
motion_adapter = None
ip_adapter = None

def upload_to_catbox(filepath):
    """ Uploads a file and returns a direct link to it. """
    try:
        with open(filepath, 'rb') as f:
            files = {'fileToUpload': (os.path.basename(filepath), f)}
            data = {'reqtype': 'fileupload'}
            response = requests.post('https://litterbox.catbox.moe/resources/internals/api.php', files=files, data=data)
            response.raise_for_status()
            return response.text
    except Exception as e:
        return f"Error uploading: {str(e)}"

def generate_video(job):
    """ The main handler function for the Runpod serverless worker. """
    global pipe, ip_adapter

    # Lazy-load models on first run
    if pipe is None:
        print("⏳ Loading models...")

        base_model = "SG161222/Realistic_Vision_V5.1_noVAE"
        adapter_id = "guoyww/animatediff-motion-adapter-v3"
        module_id = "guoyww/animatediff-motion-module-v3"
        ip_adapter_model_id = "h94/IP-Adapter"
        
        # Load AnimateDiff pipeline
        motion_adapter = MotionAdapter.from_pretrained(adapter_id, torch_dtype=torch.float16)
        scheduler = DDIMScheduler.from_pretrained(base_model, subfolder="scheduler")
        
        pipe = AnimateDiffPipeline.from_pretrained(
            base_model,
            motion_adapter=motion_adapter,
            torch_dtype=torch.float16,
            scheduler=scheduler
        )
        
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        pipe.load_motion_module(module_id, unet_additional_kwargs={"use_inflated_groupnorm": True})

        # Load and set up IP-Adapter
        ip_adapter = IPAdapter(pipe, os.path.join(ip_adapter_model_id, "models"), "ip-adapter_sd15.bin", "cpu")
        
        print("✅ Models loaded.")

    job_input = job.get('input', {})
    prompt = job_input.get('prompt', 'a couple kissing, cinematic, beautiful lighting')
    base64_image = job_input.get('init_image')

    if not base64_image:
        return {"error": "Missing 'init_image' base64 input."}

    # Decode image for identity guidance
    try:
        init_image = Image.open(io.BytesIO(base64.b64decode(base64_image))).convert("RGB")
    except Exception as e:
        return {"error": f"Image decode error: {str(e)}"}

    print("🎥 Generating frames with IPAdapter guidance...")
    
    # Generate with IP-Adapter conditioning
    frames = ip_adapter.generate(
        pil_image=init_image,
        prompt=prompt,
        num_frames=16,
        guidance_scale=7.5,
        num_inference_steps=20,
        scale=0.7
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "kissify.mp4")
        export_to_video(frames, video_path, fps=8)
        url = upload_to_catbox(video_path)
        if "Error" in url:
            return {"error": url}
        return {"output": {"video_url": url}}

# Start the serverless worker
runpod.serverless.start({"handler": generate_video})
