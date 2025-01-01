FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevent tzdata questions
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install additional dependencies for YOLOv9 and VLM
RUN pip3 install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    ultralytics \
    transformers \
    pillow

# Create necessary directories
RUN mkdir -p /config /storage

# Copy application code
COPY . .

# Create models directory and download YOLOv9 weights
RUN mkdir -p models && \
    wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt -O models/yolov9.pt

# Environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Expose ports
EXPOSE 5000
EXPOSE 1935

# Start Visioncave
CMD ["python3", "-m", "visioncave"]
