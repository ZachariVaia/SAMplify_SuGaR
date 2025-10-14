# Use *devel* so nvcc is available to build CUDA extensions
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# ---------- ENV ----------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    MPLBACKEND=TkAgg \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=8.6 \
    # defaults used by your runner/scripts
    IN_MNT=/data/in \
    OUT_MNT=/data/out \
    INDEX_ROOT=/data/out/__index_cache__

# ---------- System deps ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential ninja-build cmake \
    python3 python3-pip python3-dev python3-venv \
    ffmpeg libgl1 libglib2.0-0 \
    python3-tk tk x11-apps \
    libx11-xcb1 libxkbcommon-x11-0 libxrender1 libxext6 libxfixes3 libxi6 libxtst6 libsm6 \
 && rm -rf /var/lib/apt/lists/*

# ---------- Create non-root user (match host UID/GID if passed) ----------
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} user || true && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash user
# Workdirs & mounts with correct ownership
RUN mkdir -p /workspace ${IN_MNT} ${OUT_MNT} && chown -R user:user /workspace ${IN_MNT} ${OUT_MNT}

USER user
WORKDIR /home/user

# ---------- Python & Torch ----------
RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# Core deps (headless OpenCV, plotting, I/O, transformers utils, etc.)
RUN pip install --no-cache-dir \
    numpy pillow opencv-python-headless matplotlib tqdm \
    einops timm pycocotools imageio[ffmpeg] av pyyaml

# ---------- SAM2 ----------
RUN git clone https://github.com/facebookresearch/segment-anything-2.git
WORKDIR /home/user/segment-anything-2
# Build/Install the package against the pinned torch
RUN pip install --no-cache-dir -v --no-build-isolation -e .

# ---------- App workspace ----------
# We won’t COPY your repo here, because you bind-mount $PWD → /workspace.
# But we keep a neutral working dir so /workspace is the default.
RUN mkdir -p /home/user/app && mkdir -p /workspace && \
    ln -s /home/user/app /workspace/app || true
WORKDIR /workspace

# Helpful defaults (your run_command can override with -e)
ENV INPUT=${IN_MNT}/vazo \
    OUT=${OUT_MNT} \
    GUI=0 \
    FRAME_IDX=0 \
    OBJ_ID=1

CMD ["/bin/bash"]
