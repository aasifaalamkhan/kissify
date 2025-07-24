# main.py

import runpod
import torch
from diffusers import AnimateDiffPipeline, ControlNetModel, MotionAdapter
from diffusers.utils import export_to_video
from controlnet_aux import CannyDetector
from PIL import Image
import numpy as np
import base64
import io
import requests
import os
import tempfile

# --------------------------------- Model Setup -------------------------------- #
# This setup is more complex as it combines a realistic model, a motion adapter,
# and a ControlNet model to guide the generation from an image.

# Use a realistic base model that works well with ControlNet
BASE_MODEL_ID = "emilianJR/epiCRealism"
MOTION_ADAPTER_ID = "guoyww/animatediff-motion-adapter-v1-5-2"
CONTROLNET_MODEL_ID = "lllyasviel/sd-controlnet-canny"
TORCH_DTYPE = torch.float16

# Load all the necessary components
canny_detector = CannyDetector()
motion_adapter = MotionAdapter.from_pretrained(MOTION_ADAPTER_ID, torch_dtype=TORCH_DTYPE)
controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL_ID, torch_dtype=TORCH_DTYPE)

pipe = AnimateDiffPipeline.from_pretrained(
    BASE_MODEL_ID,
    motion_adapter=motion_adapter,
    controlnet=controlnet,
    torch_dtype=TORCH_DTYPE
)
pipe.enable_model_cpu_offload()


# ------------------------------ File Upload Utility ----------------------------- #
def upload_to_direct_linker(filepath):
    """ Uploads a file and returns a direct link to it. """
    try:
        with open(filepath, 'rb') as f:
            files = {'fileToUpload': (os.path.basename(filepath), f)}
            data = {'reqtype': 'fileupload', 'time': '24h'}
            response = requests.post('https://litterbox.catbox.moe/resources/internals/api.php', files=files, data=data)
            response.raise_for_status()
            direct_url = response.text
            print(f"Upload successful. Direct URL: {direct_url}")
            return direct_url
    except requests.exceptions.RequestException as e:
        print(f"Error uploading file: {e}")
        return str(e)


# --------------------------------- Job Handler ---------------------------------- #
def generate_video_from_image(job):
    """ The main handler function for the Runpod serverless worker. """
    job_input = job.get('input', {})
    prompt = job_input.get('prompt', 'a couple kissing, beautiful, cinematic, masterpiece')
    base64_image_str = job_input.get('init_image')

    if not base64_image_str:
        return {"error": "An initial image ('init_image') is required."}

    # 1. Decode the input image from base64
    try:
        image_bytes = base64.b64decode(base64_image_str)
        source_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return {"error": f"Failed to decode base64 image: {str(e)}"}

    # 2. Prepare the control image using Canny edge detection
    # This creates an "edge map" that guides the video generation.
    control_image = canny_detector(
        source_image,
        detect_resolution=384,
        image_resolution=512,
        low_threshold=100,
        high_threshold=200,
    )

    # 3. Generate video frames using the prompt and control image
    print("Generating video frames...")
    output = pipe(
        prompt=prompt,
        image=control_image,
        num_frames=16,
        guidance_scale=7.5,
        controlnet_conditioning_scale=0.7, # How strongly to follow the image
    )
    frames = output.frames[0]

    # 4. Save frames to a video file and upload it
    with tempfile.TemporaryDirectory() as output_dir:
        video_path = os.path.join(output_dir, "generated_video.mp4")
        export_to_video(frames, video_path, fps=8)
        direct_video_url = upload_to_direct_linker(video_path)

        # 5. Return the final URL in the expected format
        if "Error" in direct_video_url:
             return {"error": direct_video_url}
        
        return {"output": {"video_url": direct_video_url}}

# Start the serverless worker
runpod.serverless.start({"handler": generate_video_from_image})
