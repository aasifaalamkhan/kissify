import argparse
import torch
import os
from PIL import Image
from diffusers.utils import export_to_video
from diffusers import (
    WanImageToVideoPipeline,
    WanTransformer3DModel,
    AutoencoderKLWan,
    UniPCMultistepScheduler,
)
from transformers import UMT5EncoderModel, CLIPVisionModel, T5Tokenizer, CLIPImageProcessor
from huggingface_hub import hf_hub_download

def main(args):
    """
    Main function to run the image-to-video inference pipeline using a robust,
    manual component assembly strategy for the Wan 2.1 model.
    """
    # --- 1. Configuration & Path Definitions ---
    # This script assumes you have downloaded the correct model files from the
    # 'Comfy-Org/Wan_2.1_ComfyUI_repackaged' Hugging Face repository and organized them locally.
    # The expected directory structure is:
    # ./models/
    #   ├── diffusion_models/
    #   │   └── wan2.1_i2v_480p_14B_fp16.safetensors
    #   ├── vae/
    #   │   └── wan_2.1_vae.safetensors
    #   ├── text_encoders/
    #   │   └── umt5_xxl_fp8_e4m3fn_scaled.safetensors
    #   └── clip_vision/
    #       └── clip_vision_h.safetensors

    model_base_path = "./models"
    transformer_path = os.path.join(model_base_path, "diffusion_models", "wan2.1_i2v_480p_14B_fp16.safetensors")
    vae_path = os.path.join(model_base_path, "vae", "wan_2.1_vae.safetensors")
    text_encoder_path = os.path.join(model_base_path, "text_encoders", "umt5_xxl_fp8_e4m3fn_scaled.safetensors")
    image_encoder_path = os.path.join(model_base_path, "clip_vision", "clip_vision_h.safetensors")

    # This ID is used only for loading peripheral components (tokenizer, scheduler)
    base_model_id_for_peripherals = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    
    # LoRA Configuration
    lora_repo_id = "Remade-AI/kissing"
    lora_filename = "kissing_30_epochs.safetensors"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 # Recommended for Ampere GPUs and newer

    print("--- Step 1: Manual Component Loading ---")

    # --- 2. Manually Load Each Pipeline Component ---

    # Transformer (the main diffusion model)
    print(f"Loading Transformer from local path: {transformer_path}")
    transformer = WanTransformer3DModel.from_pretrained(transformer_path, torch_dtype=dtype)

    # VAE (custom architecture)
    # The 'ignore_mismatched_sizes=True' flag is CRITICAL for loading this non-standard VAE.
    print(f"Loading VAE from local path: {vae_path}")
    vae = AutoencoderKLWan.from_pretrained(vae_path, torch_dtype=dtype, ignore_mismatched_sizes=True)

    # Text Encoder (UMT5)
    print(f"Loading Text Encoder from local path: {text_encoder_path}")
    text_encoder = UMT5EncoderModel.from_pretrained(text_encoder_path, torch_dtype=dtype)

    # Image Encoder (CLIP)
    print(f"Loading Image Encoder from local path: {image_encoder_path}")
    image_encoder = CLIPVisionModel.from_pretrained(image_encoder_path, torch_dtype=dtype)

    # Tokenizer and Image Processor (loaded from the peripheral repo)
    print(f"Loading Tokenizer from HF: {base_model_id_for_peripherals}")
    tokenizer = T5Tokenizer.from_pretrained(base_model_id_for_peripherals, subfolder="tokenizer")
    print(f"Loading Image Processor from HF: {base_model_id_for_peripherals}")
    image_processor = CLIPImageProcessor.from_pretrained(base_model_id_for_peripherals, subfolder="image_processor")
    
    # Scheduler (explicitly choosing UniPCMultistepScheduler for better results)
    print(f"Loading Scheduler from HF: {base_model_id_for_peripherals}")
    scheduler = UniPCMultistepScheduler.from_pretrained(base_model_id_for_peripherals, subfolder="scheduler")
    
    print("All components loaded successfully.")

    # --- 3. Assemble The Pipeline & Integrate LoRA ---
    print("\n--- Step 2: Assembling Pipeline and Integrating LoRA ---")
    
    pipe = WanImageToVideoPipeline(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        image_encoder=image_encoder,
        image_processor=image_processor,
    )
    
    # Download and load LoRA weights
    print(f"Downloading LoRA from {lora_repo_id}...")
    lora_path = hf_hub_download(repo_id=lora_repo_id, filename=lora_filename)
    print(f"Loading LoRA weights from {lora_path}...")
    pipe.load_lora_weights(lora_path)
    
    print("Pipeline assembled and LoRA loaded successfully.")

    # --- 4. Optimizations & Device Placement ---
    # Enable model offloading to manage VRAM usage; the pipeline will automatically
    # move components to the GPU as needed.
    pipe.enable_model_cpu_offload()

    # --- 5. Prepare Inputs ---
    print("\n--- Step 3: Preparing Inputs for Inference ---")
    generator = torch.manual_seed(args.seed)
    try:
        image = Image.open(args.image_path).convert("RGB")
        print(f"Input image '{args.image_path}' loaded.")
    except FileNotFoundError:
        print(f"ERROR: Input image not found at '{args.image_path}'.")
        print("Using a dummy black image for demonstration purposes.")
        image = Image.new('RGB', (args.width, args.height), color='black')

    # The LoRA trigger phrase "k144ing kissing" must be in the prompt.
    full_prompt = f"{args.prompt}, k144ing kissing"

    # --- 6. Run Inference ---
    print("\n--- Step 4: Generating Video ---")
    video_frames = pipe(
        prompt=full_prompt,
        image=image,
        num_inference_steps=args.steps,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        flow_shift=args.flow_shift,
        generator=generator,
        height=args.height,
        width=args.width,
        cross_attention_kwargs={"scale": args.lora_scale}, # Correct way to pass LoRA scale
    ).frames[0]
    print("Video generation complete.")

    # --- 7. Save Output ---
    print("\n--- Step 5: Saving Output ---")
    export_to_video(video_frames, args.output_path, fps=args.fps)
    print(f"Video saved successfully to '{args.output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Image-to-Video inference with Wan 2.1 and a LoRA using manual component assembly."
    )
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt describing the video (do not include LoRA triggers).")
    parser.add_argument("--output_path", type=str, default="output.mp4", help="Path to save the output video.")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="Guidance scale (6.0 recommended for this LoRA).")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to generate (model was trained on 81).")
    parser.add_argument("--fps", type=int, default=16, help="FPS of the output video.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--width", type=int, default=832, help="Width of the video.")
    parser.add_argument("--height", type=int, default=480, help="Height of the video.")
    parser.add_argument("--lora_scale", type=float, default=1.0, help="LoRA scale (1.0 recommended).")
    parser.add_argument("--flow_shift", type=int, default=5, help="Flow shift value (5.0 recommended for this LoRA).")

    args = parser.parse_args()
    main(args)
