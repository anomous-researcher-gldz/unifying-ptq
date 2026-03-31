# CUDA 11.8 runtime + cuDNN8 on Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Use bash for consistency
SHELL ["/bin/bash", "-lc"]

# Basic OS deps (add GUI deps for full OpenCV)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    wget curl ca-certificates build-essential git \
    libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 libx11-6 libxfixes3 libxtst6 \
    libgtk-3-0 libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create env
RUN conda create -y -n srtquant python=3.9 
ENV CONDA_DEFAULT_ENV=srtquant
ENV PATH=$CONDA_DIR/envs/srtquant/bin:$PATH
ENV CUDA_HOME=/usr/local/cuda-11.7
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Working directory
WORKDIR /workspace

# Copy requirements first
# Copy your project
COPY . /workspace

RUN pip install six
RUN pip install --no-cache-dir \
  torch==2.0.1+cu117 \
  torchvision==0.15.2+cu117 \
  torchaudio==2.0.2 \
  --index-url https://download.pytorch.org/whl/cu117

RUN pip install -r requirements.txt

RUN pip install -e . --no-build-isolation -v

RUN pip uninstall -y numpy opencv-python opencv-python-headless transformers tokenizers huggingface-hub causal-conv1d mamba-ssm 

RUN pip install opencv-python==4.9.0.80
RUN pip install numpy==1.24.3
RUN pip install transformers==4.37.1 tokenizers==0.15.1 huggingface-hub==0.20.3
RUN pip install causal_conv1d==1.0.0 --no-build-isolation
RUN pip install mamba_ssm==1.0.1 --no-build-isolation

RUN pip install -v --no-build-isolation causal_conv1d==1.0.0

RUN pip install -v --no-build-isolation mamba_ssm==1.0.1

# Default shell
CMD ["/bin/bash"]
