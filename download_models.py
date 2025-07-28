import os
from huggingface_hub import hf_hub_download

def download_wan21_models():
    """
    Downloads the necessary model components for the Wan 2.1 Image-to-Video pipeline
    from the community-repackaged Hugging Face repository and organizes them into the
    correct local directory structure required by the inference script.
    """
    # The community-repackaged repository that contains the correct, stable files.
    repo_id = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"

    # Define the target directory. In RunPod, this will default to /workspace/models
    base_model_path = "./models"
    
    # A mapping of filenames to the subdirectories they should be saved in.
    # This structure is required by the manual assembly inference script.
    files_to_download = {
        "diffusion_models": "wan2.1_i2v_480p_14B_fp16.safetensors",
        "vae": "wan_2.1_vae.safetensors",
        "text_encoders": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "clip_vision": "clip_vision_h.safetensors",
    }

    print("--- Starting Download of Wan 2.1 Model Components ---")
    print(f"All files will be saved to the '{os.path.abspath(base_model_path)}' directory.")

    # Iterate over the mapping and download each file into its correct location.
    for subdir, filename in files_to_download.items():
        # Construct the full path for the subdirectory.
        download_dir = os.path.join(base_model_path, subdir)
        
        # Create the directory if it doesn't already exist.
        os.makedirs(download_dir, exist_ok=True)

        # Check if the file already exists to avoid re-downloading.
        local_file_path = os.path.join(download_dir, filename)
        if os.path.exists(local_file_path):
            print(f"\n[SKIPPED] File already exists: {local_file_path}")
            continue

        print(f"\nDownloading '{filename}' from repo '{repo_id}'...")
        print(f"Target directory: {download_dir}")

        try:
            # hf_hub_download will download the file and return its path.
            # We save it directly to the target directory.
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=download_dir,
                local_dir_use_symlinks=False, # Use False for better compatibility in containers
                resume_download=True,
            )
            print(f"Successfully downloaded '{filename}'.")
        except Exception as e:
            print(f"An error occurred while downloading {filename}: {e}")

    print("\n--- All model components downloaded and organized successfully! ---")
    print("You can now run the inference.py script.")


if __name__ == "__main__":
    # Before running, ensure you have the necessary library installed:
    # pip install huggingface_hub
    download_wan21_models()
