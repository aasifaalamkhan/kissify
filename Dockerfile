# Dockerfile

# ✅ Using a stable, long-term supported PyTorch base image
FROM runpod/pytorch:2.2.0-py3.10-cuda11.8.0-devel

WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the worker script
COPY main.py .
