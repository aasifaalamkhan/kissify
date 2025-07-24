# main.py

import runpod
import torch
from diffusers import AnimateDiffPipeline
from diffusers.utils import export_to_video
import tempfile
import os
import requests # Use the requests library for uploading

# --- MODEL SETUP ---
model_id = "ByteDance/AnimateDiff-Lightning"
# For better performance on most GPUs, consider using float16
torch_dtype = torch.float16
variant = "fp16"

# Load AnimateDiff with memory optimization
pipe = AnimateDiffPipeline.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    variant=variant
)
pipe.enable_model_cpu_offload()


# --- UPLOAD FUNCTION ---
def upload_to_direct_linker(filepath):
    """
    Uploads a file to a service that provides a direct link.
    This uses litterbox.catbox.moe which keeps files for 24h.
    """
    try:
        with open(filepath, 'rb') as f:
            files = {'fileToUpload': (os.path.basename(filepath), f)}
            data = {'reqtype': 'fileupload', 'time': '24h'}
            response = requests.post('https://litterbox.catbox.moe/resources/internals/api.php', files=files, data=data)
            response.raise_for_status()  # This will throw an error for a bad response
            # The response text is the direct URL
            direct_url = response.text
            print(f"Upload successful. Direct URL: {direct_url}")
            return direct_url
    except requests.exceptions.RequestException as e:
        print(f"Error uploading file: {e}")
        return str(e)


# --- HANDLER FUNCTION ---
def generate_kissing_video(job):
    """
    The main handler for the Runpod serverless worker.
    """
    job_input = job.get('input', {})
    prompt = job_input.get('prompt', 'a couple kissing, photorealistic, 4k')
    
    # Generate video frames
    frames = pipe(prompt=prompt, num_inference_steps=4, num_frames=16).frames[0]

    # Use a temporary directory to save the video
    with tempfile.TemporaryDirectory() as output_dir:
        video_path = os.path.join(output_dir, "generated_video.mp4")
        
        # Export frames to a video file
        export_to_video(frames, video_path, fps=10)

        # Upload the video to get a direct link
        direct_video_url = upload_to_direct_linker(video_path)

        # Return the URL in the correct JSON format for the Flutter app
        return {
            "output": {
                "video_url": direct_video_url
            }
        }

# Start the serverless worker
runpod.serverless.start({"handler": generate_kissing_video})