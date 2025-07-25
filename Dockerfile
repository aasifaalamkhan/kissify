# Use the official PyTorch image with CUDA 11.8 and Python 3.10
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel

# Set the working directory inside the container
WORKDIR /app

# Set environment variables for non-buffered Python output and CUDA home
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Set Hugging Face cache directory for both build and runtime
ENV HF_HOME=/app/hf_cache
# Create the cache directory if it doesn't exist
RUN mkdir -p ${HF_HOME}

# Install system dependencies:
RUN echo "--- Updating apt packages and installing system dependencies ---" && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        build-essential \
        python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "--- System dependencies installed ---"

# Copy the requirements file and install Python packages.
COPY requirements.txt .
RUN echo "--- Installing Python packages from requirements.txt ---" && \
    pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118 && \
    echo "--- Python packages installed ---"

# --- ACTIVATE AND REVISE: Pre-fetch Hugging Face models during build ---
# Use ARG for build-time secrets, then set ENV for runtime.
# This ARGs needs to be passed to docker build command or RunPod build settings.
ARG HUGGING_FACE_HUB_TOKEN
ENV HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}

RUN echo "--- Pre-fetching Hugging Face models ---" && \
    python -c " \
import os; \
from huggingface_hub import hf_hub_download, snapshot_download, login; \
# Login using the env var set by ARG or attempt to read existing token \
token = os.getenv('HUGGING_FACE_HUB_TOKEN'); \
if token: \
    print('Logging in to Hugging Face Hub...'); \
    login(token=token, add_to_git_credential=False); \
else: \
    print('HUGGING_FACE_HUB_TOKEN not set as build arg. Proceeding without explicit login. Gated models may fail.'); \
\
hf_cache_dir = os.getenv('HF_HOME', '/app/hf_cache'); \
print(f'Using HF_HOME: {hf_cache_dir}'); \
\
print('Pre-fetching SG161222/Realistic_Vision_V5.1_noVAE...'); \
snapshot_download(repo_id='SG161222/Realistic_Vision_V5.1_noVAE', allow_patterns=['*.safetensors', '*.json'], local_dir=os.path.join(hf_cache_dir, 'models--SG161222--Realistic_Vision_V5.1_noVAE'), resume_download=True, token=token); \
\
print('Pre-fetching guoyww/animatediff-motion-adapter-v1-5-2...'); \
snapshot_download(repo_id='guoyww/animatediff-motion-adapter-v1-5-2', allow_patterns=['*.safetensors', '*.json'], local_dir=os.path.join(hf_cache_dir, 'models--guoyww--animatediff-motion-adapter-v1-5-2'), resume_download=True, token=token); \
\
print('Pre-fetching lllyasviel/control_v11p_sd15_openpose...'); \
snapshot_download(repo_id='lllyasviel/control_v11p_sd15_openpose', allow_patterns=['*.safetensors', '*.json'], local_dir=os.path.join(hf_cache_dir, 'models--lllyasviel--control_v11p_sd15_openpose'), resume_download=True, token=token); \
\
print('Pre-fetching lllyasviel/control_v11f1p_sd15_depth...'); \
snapshot_download(repo_id='lllyasviel/control_v11f1p_sd15_depth', allow_patterns=['*.safetensors', '*.json'], local_dir=os.path.join(hf_cache_dir, 'models--lllyasviel--control_v11f1p_sd15_depth'), resume_download=True, token=token); \
\
print('Pre-fetching h94/IP-Adapter...'); \
# For IP-Adapter, snapshot_download for the repo, then verify specific files. \
# The `local_dir` here needs to match how HuggingFace Hub structures its cache: \
# it typically creates subdirectories like `models--h94--IP-Adapter` for the main repo. \
snapshot_download(repo_id='h94/IP-Adapter', allow_patterns=['models/*', 'ip-adapter_sd15.bin'], local_dir=os.path.join(hf_cache_dir, 'models--h94--IP-Adapter'), resume_download=True, token=token); \
\
print('Pre-fetching openai/clip-vit-large-patch14...'); \
snapshot_download(repo_id='openai/clip-vit-large-patch14', local_dir=os.path.join(hf_cache_dir, 'models--openai--clip-vit-large-patch14'), resume_download=True, token=token); \
\
print('Pre-fetching lllyasviel/ControlNet (for aux detectors)...'); \
snapshot_download(repo_id='lllyasviel/ControlNet', local_dir=os.path.join(hf_cache_dir, 'models--lllyasviel--ControlNet'), resume_download=True, token=token); \
\
print('Model pre-fetching complete.'); \
" && \
    echo "--- Pre-fetched models successfully ---"

# Copy your main application script into the container.
# This is crucial: it comes AFTER all dependencies and model downloads.
COPY main.py .

# Expose port 3000 for the health check server.
EXPOSE 3000

# This is the command that RunPod will execute when the container starts.
CMD ["python", "main.py"]