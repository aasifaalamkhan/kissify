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
# Use --yes for non-interactive installation.
# python3-pip is often needed explicitly for newer pip versions.
RUN echo "--- Updating apt packages and installing system dependencies ---" && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        build-essential \
        python3-dev \
        python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "--- System dependencies installed successfully ---"

# Copy the requirements file and install Python packages.
# Use RUN --mount=type=cache to cache pip downloads for faster rebuilds on some Docker versions
# and ensure --no-cache-dir is used when not caching for production images.
COPY requirements.txt .
RUN echo "--- Installing Python packages from requirements.txt ---" && \
    pip install --upgrade pip setuptools wheel && \
    # Using the extra-index-url for PyTorch related packages is crucial.
    # The --break-system-packages is sometimes needed in newer Python versions in Docker.
    pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118 && \
    echo "--- Python packages installed successfully ---"

# --- Pre-fetch Hugging Face models during build ---
# Use ARG for build-time secrets, then set ENV for runtime.
# This ARG needs to be passed to docker build command or RunPod build settings (e.g., --build-arg HUGGING_FACE_HUB_TOKEN=hf_...)
ARG HUGGING_FACE_HUB_TOKEN
ENV HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}

# Copy the model download script and execute it
# This resolves the SyntaxError by avoiding complex inline Python string escaping.
COPY download_models.py .
RUN echo "--- Running model pre-fetching script ---" && \
    python download_models.py || { echo "ERROR: Model pre-fetching failed. Check download_models.py logs."; exit 1; } && \
    echo "--- Model pre-fetching completed successfully ---"

# Copy your main application script into the container.
# This is crucial: it comes AFTER all dependencies and model downloads.
COPY main.py .

# Expose port 3000 for the health check server.
EXPOSE 3000

# This is the command that RunPod will execute when the container starts.
# Use exec form to ensure proper signal handling.
CMD ["python", "main.py"]