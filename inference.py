import argparse
import torch
from PIL import Image
from diffusers.utils import export_to_video
from diffusers import (
    I2VGenXLPipeline,
    I2VGenXLUNet,
    AutoencoderKL,
    UniPCMultistepScheduler,
)
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor


def main(args):
    """
    Main function to run the image-to-video inference pipeline.
    """
    # Define model paths
    model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    lora_model_id = "Remade-AI/kissing"
    lora_filename = "kissing_30_epochs.safetensors"

    # Use bfloat16 for memory efficiency
    dtype = torch.bfloat16
    print("Loading model components...")

    # 1. Load all necessary components from their subfolders
    # REMOVED variant="fp16" from the following lines
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        model_id, subfolder="image_encoder", torch_dtype=dtype
    )
    feature_extractor = CLIPImageProcessor.from_pretrained(
        model_id, subfolder="image_encoder"
    )
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    unet = I2VGenXLUNet.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype)
    scheduler = UniPCMultistepScheduler.from_pretrained(
        model_id, subfolder="scheduler"
    )

    # 2. Create the pipeline and pass all components explicitly
    pipe = I2VGenXLPipeline(
        vae=vae,
        image_encoder=image_encoder,
        feature_extractor=feature_extractor,
        unet=unet,
        scheduler=scheduler,
        text_encoder=None,
        tokenizer=None,
    )
    print("Pipeline created.")

    # 3. Apply LoRA weights
    print("Loading and fusing LoRA weights...")
    pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
    # Fuse LoRA for faster inference. Adjust lora_scale as needed.
    pipe.fuse_lora(lora_scale=args.lora_scale)
    print("LoRA fused.")

    # 4. Set up optimizations and move to GPU
    # Xformers for memory-efficient attention
    pipe.enable_xformers_memory_efficient_attention()
    # Offload model parts to CPU to save VRAM
    pipe.enable_model_cpu_offload()

    # 5. Prepare inputs
    generator = torch.manual_seed(args.seed)
    image = Image.open(args.image_path).convert("RGB")
    print(f"Input image '{args.image_path}' loaded.")

    # 6. Run inference
    print("Generating video...")
    video_frames = pipe(
        prompt=args.prompt,
        image=image,
        num_inference_steps=args.steps,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        generator=generator,
        height=args.height,
        width=args.width,
        flow_shift=args.flow_shift,
    ).frames[0]
    print("Video generation complete.")

    # 7. Save the output
    export_to_video(video_frames, args.output_path, fps=args.fps)
    print(f"Video saved to '{args.output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Image-to-Video inference with Wan2.1 and a LoRA."
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the input image."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The prompt describing the video, including LoRA triggers.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.mp4",
        help="Path to save the output video.",
    )
    parser.add_argument(
        "--steps", type=int, default=30, help="Number of inference steps."
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=6.5, help="Guidance scale."
    )
    parser.add_argument(
        "--num_frames", type=int, default=16, help="Number of frames to generate."
    )
    parser.add_argument("--fps", type=int, default=8, help="FPS of the output video.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--width", type=int, default=852, help="Width of the video.")
    parser.add_argument("--height", type=int, default=480, help="Height of the video.")
    parser.add_argument(
        "--lora_scale", type=float, default=0.7, help="LoRA fusion scale."
    )
    parser.add_argument(
        "--flow_shift", type=int, default=0, help="Flow shift value."
    )

    args = parser.parse_args()
    main(args)
