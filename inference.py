# inference.py
#
# An optimized inference pipeline for generating video from an image using
# the Wan2.1-14B model and the 'kissing' LoRA adapter.
#
# Usage:
# python inference.py \
#   --image_path "path/to/your/image.jpg" \
#   --prompt "A man and a woman are embracing near a lake with mountains in the background. They are k144ing kissing, while still embracing each other." \
#   --output_path "output/kissing_video.mp4"

import argparse
import os
import torch
from PIL import Image
from diffusers import WanPipeline, UniPCMultistepScheduler
from diffusers.utils import export_to_video

def main(args):
    """
    Main function to set up the pipeline and run inference.
    """
    print("--- Step 1: Initializing Pipeline ---")
    
    # Define model and LoRA identifiers
    base_model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    lora_id = "Remade-AI/kissing"
    lora_filename = "kissing_30_epochs.safetensors"

    # Use float16 for VRAM efficiency on compatible GPUs
    torch_dtype = torch.float16

    # Load the base pipeline. The WanPipeline is specifically designed for this model family.
    pipe = WanPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
    )

    print(f"--- Step 2: Loading LoRA Adapter: {lora_id} ---")
    # Load the LoRA weights into the pipeline.
    # This modifies the attention layers of the base model.
    pipe.load_lora_weights(lora_id, weight_name=lora_filename)

    # Configure the scheduler with the recommended flow_shift parameter for this model.
    # This parameter is specific to the Wan model's motion generation.
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=args.flow_shift
    )

    print("--- Step 3: Configuring Optimizations and Device Placement ---")
    # Enable memory-saving optimizations if requested
    if args.enable_cpu_offload:
        print("Enabling Model CPU Offload for VRAM savings.")
        pipe.enable_model_cpu_offload()
    else:
        print("Moving pipeline to GPU.")
        pipe.to("cuda")

    if args.enable_vae_slicing:
        print("Enabling VAE Slicing for VRAM savings during decoding.")
        pipe.vae.enable_slicing()

    print("--- Step 4: Preparing Inputs ---")
    # Load and prepare the input image
    try:
        input_image = Image.open(args.image_path).convert("RGB")
        print(f"Loaded image from: {args.image_path}")
    except FileNotFoundError:
        print(f"Error: Input image not found at {args.image_path}")
        return

    # Resize image to be compatible with the model, maintaining aspect ratio
    original_width, original_height = input_image.size
    aspect_ratio = original_width / original_height
    if original_width > original_height:
        new_width = args.width
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = args.height
        new_width = int(new_height * aspect_ratio)
    
    # Ensure dimensions are divisible by 8 for the model
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8
    
    input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
    print(f"Resized image to: {new_width}x{new_height}")

    # Set a generator for reproducible results
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    # The LoRA strength is controlled via cross_attention_kwargs.
    # This is a more advanced way to control LoRA influence in diffusers.
    cross_attention_kwargs = {"scale": args.lora_scale}

    print("--- Step 5: Running Inference ---")
    # The pipeline call encapsulates the entire complex diffusion process.
    video_frames = pipe(
        prompt=args.prompt,
        image=input_image,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        num_frames=args.num_frames,
        width=new_width,
        height=new_height,
        generator=generator,
        cross_attention_kwargs=cross_attention_kwargs,
    ).frames

    print("--- Step 6: Saving Output ---")
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Export the generated frames to a video file.
    export_to_video(video_frames, args.output_path, fps=args.fps)
    print(f"Successfully saved video to: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run I2V inference with Wan2.1 and a LoRA.")
    
    # Input/Output Arguments
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation. Must include LoRA trigger.")
    parser.add_argument("--output_path", type=str, default="output/generated_video.mp4", help="Path to save the output MP4 video.")
    
    # Generation Parameter Arguments
    parser.add_argument("--negative_prompt", type=str, default="low quality, blurry, deformed, worst quality, low resolution", help="Concepts to avoid.")
    parser.add_argument("--steps", type=int, default=40, help="Number of inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="Guidance scale (prompt adherence).")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to generate (must be 4k+1).")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second for the output video.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--width", type=int, default=832, help="Target width for the video.")
    parser.add_argument("--height", type=int, default=480, help="Target height for the video.")

    # LoRA and Scheduler Arguments
    parser.add_argument("--lora_scale", type=float, default=1.0, help="Strength of the LoRA adapter.")
    parser.add_argument("--flow_shift", type=float, default=5.0, help="Flow shift parameter for the UniPC scheduler.")

    # Optimization Arguments
    parser.add_argument("--enable_cpu_offload", action='store_true', help="Enable CPU offloading to save VRAM.")
    parser.add_argument("--enable_vae_slicing", action='store_true', help="Enable VAE slicing to save VRAM during decoding.")

    args = parser.parse_args()
    main(args)