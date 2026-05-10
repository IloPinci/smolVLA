FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/miniforge3/bin:${PATH}"

# Install system-level graphics and rendering libraries required by Genesis
RUN apt-get update && apt-get install -y \
    wget \
    git \
    libgl1-mesa-glx \
    libegl1 \
    libxrandr2 \
    libxext6 \
    libxcursor1 \
    libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/miniforge3 \
    && rm /tmp/miniforge.sh \
    && conda init bash

WORKDIR /workspace

# Reconstruct environment without lerobot
COPY environment.yml .
RUN conda env create -f environment.yml

# Transfer source code, including the local lerobot directory
COPY . .

# Install local lerobot into the isolated conda environment
RUN /opt/miniforge3/envs/lerobot/bin/pip install -e ./lerobot

RUN echo "conda activate lerobot" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

CMD ["python", "train.py"]
