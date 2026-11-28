# syntax=docker/dockerfile:1

# Default to PyTorch 2.6 + CUDA 12.4 (PTX 8.4, compatible with NVIDIA Linux drivers >= 550
# across V100/T4/RTX 20xx-40xx/A100/H100). Override for Blackwell / R570+ hosts, e.g.:
#   --build-arg GPU_BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
ARG GPU_BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
ARG CPU_BASE_IMAGE=python:3.11-slim-bookworm
ARG WMA_EXTRAS=h5

# -----------------------------------------------------------------------------
# Lightweight CPU-only image (Slicer-free rigid/nonrigid ORG & vectorized CPU)
#   docker build --target cpu -t wma-cpu:0.4.0 .
# -----------------------------------------------------------------------------
FROM ${CPU_BASE_IMAGE} AS cpu

ARG WMA_EXTRAS
LABEL org.opencontainers.image.title="WhiteMatterAnalysis (CPU)" \
      org.opencontainers.image.description="WhiteMatterAnalysis streamline tractography and ORG parcellation (CPU image)" \
      org.opencontainers.image.source="https://github.com/xiaji/whitematteranalysis_gpu"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    WMA_ATLAS_CACHE=/tmp/wma-cache/atlas \
    MPLCONFIGDIR=/tmp/wma-cache/mpl

# Minimal shared libraries required by headless PyPI VTK wheels on slim Linux.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxcursor1 \
        libxext6 \
        libxfixes3 \
        libxi6 \
        libxrender1 \
        libxt6 \
    && rm -rf /var/lib/apt/lists/*

COPY . /tmp/wma-src
RUN if [ -n "${WMA_EXTRAS}" ]; then \
        python -m pip install --no-cache-dir "/tmp/wma-src[${WMA_EXTRAS}]"; \
    else \
        python -m pip install --no-cache-dir /tmp/wma-src; \
    fi \
    && rm -rf /tmp/wma-src

RUN wma --version \
    && python -c "import vtk, whitematteranalysis as wma; from whitematteranalysis.pipelines import ORGParcellationPipeline; print('WMA', wma.__version__, 'VTK', vtk.vtkVersion.GetVTKVersion())"

WORKDIR /workspace
RUN chmod 1777 /workspace

ENTRYPOINT ["wma"]
CMD ["--help"]

# -----------------------------------------------------------------------------
# Default CUDA + Triton GPU image (Slicer-free rigid/nonrigid ORG parcellation)
#   docker build -t wma-gpu:0.4.0 .
#   docker build --build-arg GPU_BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime -t wma-gpu:cu128 .
# -----------------------------------------------------------------------------
FROM ${GPU_BASE_IMAGE} AS gpu

ARG WMA_EXTRAS
LABEL org.opencontainers.image.title="WhiteMatterAnalysis GPU" \
      org.opencontainers.image.description="CUDA/Triton-accelerated WhiteMatterAnalysis tractography and ORG parcellation" \
      org.opencontainers.image.source="https://github.com/xiaji/whitematteranalysis_gpu"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    WMA_ATLAS_CACHE=/tmp/wma-cache/atlas \
    TRITON_CACHE_DIR=/tmp/wma-cache/triton \
    MPLCONFIGDIR=/tmp/wma-cache/mpl

# Minimal C toolchain for Triton JIT compilation plus headless VTK shared libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libc6-dev \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxcursor1 \
        libxext6 \
        libxfixes3 \
        libxi6 \
        libxrender1 \
        libxt6 \
    && rm -rf /var/lib/apt/lists/*

COPY . /tmp/wma-src
RUN if [ -n "${WMA_EXTRAS}" ]; then \
        python -m pip install --no-cache-dir "/tmp/wma-src[${WMA_EXTRAS}]"; \
    else \
        python -m pip install --no-cache-dir /tmp/wma-src; \
    fi \
    && rm -rf /tmp/wma-src

RUN gcc --version >/dev/null \
    && wma --version \
    && python -c "import torch, triton, vtk, whitematteranalysis as wma; from whitematteranalysis.pipelines import ORGParcellationPipeline; assert torch.version.cuda, 'CUDA PyTorch build missing'; print('WMA', wma.__version__, 'torch', torch.__version__, 'triton', triton.__version__, 'VTK', vtk.vtkVersion.GetVTKVersion())"

WORKDIR /workspace
RUN chmod 1777 /workspace

ENTRYPOINT ["wma"]
CMD ["--help"]
