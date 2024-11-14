# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3.10 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app
COPY requirements.txt /app

RUN pip3 install setuptools
RUN pip3 install -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

COPY . /app

# Clone and install CenterNet submodule
RUN git clone https://github.com/tteepe/CenterNet-pytorch-lightning.git lib \
    && cd lib \
    && git submodule update --init --recursive \
    && pip install -e .

# Fix import for deprecated import:
RUN sed -i 's/from DCN.dcn_v2 import DCN/from mmcv.ops.deform_conv import DeformConv2d as DCN/' lib/CenterNet/models/backbones/resnet_dcn.py
RUN sed -i 's/from DCN.dcn_v2 import DCN/from mmcv.ops.deform_conv import DeformConv2d as DCN/' lib/CenterNet/models/backbones/pose_dla_dcn.py
RUN sed -i 's/from collections import Callable/from collections.abc import Callable/' lib/CenterNet/transforms/sample.py

# Create mount point for data
RUN mkdir -p /data
# Create mount point for trained models
RUN mkdir -p /trained_models

# Define build arguments with a default value
ARG LR=0.0004
ARG BS=8
ARG EPOCHS=20

# Replace the problematic ENTRYPOINT with a shell script approach
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh && \
    sed -i 's/\r$//' /app/entrypoint.sh  # Remove Windows line endings if present
ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]

# Define volume for data mounting
VOLUME ["/data", "/trained_models"]
