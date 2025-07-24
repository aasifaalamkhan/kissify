# Dockerfile

# ✅ Using a newer, available PyTorch 2.3 base image from Runpod
FROM runpod/pytorch:2.3.0-py3.10-cuda12.1.1-devel-20240606

WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the worker script
COPY main.py .
