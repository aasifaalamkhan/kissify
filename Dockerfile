# Use the official PyTorch image with CUDA 11.8 and Python 3.10
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies:
# - git for cloning repositories (like IP-Adapter if needed or other package management)
# - ffmpeg for video processing (used by diffusers export_to_video)
# We use --no-install-recommends to keep the image size smaller and clean up apt cache.
RUN apt-get update && \
    apt-get install -y --no-install-recommends git ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python packages.
# --no-cache-dir prevents pip from storing downloaded packages, reducing image size.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your main application script into the container.
COPY main.py .

# Expose port 3000 for the health check server.
# This is crucial for RunPod to determine if your worker is healthy.
EXPOSE 3000

# This is the command that RunPod will execute when the container starts.
# It starts your main.py script, which then initializes the RunPod worker and
# the health check server.
CMD ["python", "main.py"]