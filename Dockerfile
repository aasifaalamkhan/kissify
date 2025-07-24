# Dockerfile

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-20240412

WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the worker script
COPY main.py .

# The runpod library handles the entrypoint