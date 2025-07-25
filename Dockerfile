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

# --- Pre-fetch Hugging Face models during build ---
# Use ARG for build-time secrets, then set ENV for runtime.
# This ARG needs to be passed to docker build command or RunPod build settings.
ARG HUGGING_FACE_HUB_TOKEN
ENV HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}

# Copy the model download script and execute it
# This resolves the SyntaxError by avoiding complex inline Python string escaping.
COPY download_models.py .
RUN echo "--- Pre-fetching Hugging Face models ---" && \
    python download_models.py && \
    echo "--- Pre-fetched models successfully ---"

# Copy your main application script into the container.
# This is crucial: it comes AFTER all dependencies and model downloads.
COPY main.py .

# Expose port 3000 for the health check server.
EXPOSE 3000

# This is the command that RunPod will execute when the container starts.
CMD ["python", "main.py"]