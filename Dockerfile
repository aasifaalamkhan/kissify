# Dockerfile

# ✅ Use the official PyTorch image with CUDA 11.8 and Python 3.10
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel

WORKDIR /app

# Install ffmpeg for video processing and git for package management
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg

# Copy requirements file and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your main worker script
COPY main.py .

# No CMD or ENTRYPOINT is needed for Runpod Serverless
