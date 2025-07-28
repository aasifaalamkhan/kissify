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
from diffusers.models import UNet2DModel
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def main(args):
    """
    Main function to run the image-to-video inference pipeline.
    """
    # --- Model & LoRA Identifiers ---
    model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    lora_model_id = "Remade-AI/kissing"
    lora_filename = "kissing_30_epochs.safetensors"
    
    # We still need to borrow the config to know the UNet's architecture
    config_source_model_id = "ali-vilab/i2vgen-xl"

    # Use bfloat16 for memory efficiency
    dtype = torch.bfloat16
    print("Loading model components...")

    # --- 1. Manually Load Standard Components ---

    # VAE
    vae = AutoencoderKL.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
    )

    # Image Encoder and Processor
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        model_id, subfolder="image_encoder", torch_dtype=dtype
    )
    feature_extractor = CLIPImageProcessor.from_pretrained(
        model_id, subfolder="image_processor"
    )

    # --- UNet (Definitive Fix for Unconventional Model) ---
    # The UNet weights are in the root 'wan21.safetensors' file, not a 'unet' folder.

    # 1. Borrow the config to create the empty UNet structure
    print("Borrowing UNet config to build model structure...")
    unet_config = UNet2DModel.load_config(config_source_model_id, subfolder="unet")
    unet = I2VGenXLUNet.from_config(unet_config)

    # 2. Download the main safetensors file from the model's root directory
    print("Downloading main weights file 'wan21.safetensors'...")
    unet_weights_path = hf_hub_download(
        repo_id=model_id,
        filename="wan21.safetensors",  # THIS IS THE CORRECT FILE
    )

    # 3. Load the entire state dictionary from the file
    print("Loading state dictionary from 'wan21.safetensors'...")
    state_dict = load_file(unet_weights_path, device="cpu")

    # 4. Load the weights into the UNet structure non-strictly.
    # `strict=False` is ESSENTIAL here. It tells the model to only load the
    # matching weights (for the UNet) and ignore all other weights in the file.
    print("Loading UNet-specific weights into the model...")
    unet.load_state_dict(state_dict, strict=False)
    
    # 5. Ensure the UNet is in the correct data type
    unet.to(dtype)
    print("UNet loaded successfully.")

    # Scheduler
    scheduler = UniPCMultistepScheduler.from_pretrained(
        model_id, subfolder="scheduler"
    )

    # --- 2. Assemble The Pipeline ---
    pipe = I2VGenXLPipeline(
        vae=vae,
        image_encoder=image_encoder,
        feature_extractor=feature_extractor,
        unet=unet,
        scheduler=scheduler,
        text_encoder=None,
        tokenizer=None,
    )
    print("Pipeline created successfully.")

    # --- 3. Apply LoRA Weights ---
    print("Loading and fusing LoRA weights...")
    pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
    pipe.fuse_lora(lora_scale=args.lora_scale)
    print("LoRA fused.")

    # --- 4. Optimizations & Device Placement ---
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    # --- 5. Prepare Inputs ---
    generator = torch.manual_seed(args.seed)
    image = Image.open(args.image_path).convert("RGB")
    print(f"Input image '{args.image_path}' loaded.")

    # --- 6. Run Inference ---
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

    # --- 7. Save Output ---
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
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
