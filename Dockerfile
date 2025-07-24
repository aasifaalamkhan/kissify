# Dockerfile

# Use the official PyTorch image with CUDA 11.8 and Python 3.10
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel

WORKDIR /app

# Install ffmpeg for video processing and git for package management
# Clean up apt cache to reduce image size
RUN apt-get update && \
    apt-get install -y --no-install-recommends git ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your main worker script
COPY main.py .

# Expose port for health checks (optional but helpful for some platforms like RunPod)
EXPOSE 3000

# This is the crucial missing piece. It tells Runpod how to start your script.
CMD ["python", "main.py"]