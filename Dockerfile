# Sticker it — GPU masking worker image for Vast.ai serverless.
#
# EVERYTHING is baked in: Python deps (pinned), BiRefNet + SAM 2.1 model weights, and the
# worker code. A fresh Vast machine only pulls this image and runs — no pip installs, no
# HuggingFace downloads, no version drift. Boot-to-Ready becomes ~3 minutes, identical on
# every host.
#
# Built + published by .github/workflows/build-image.yml → ghcr.io/kartdavid/si-gpu-masking
#
# Notes:
# - python:3.10-slim base, no CUDA toolkit needed: torch pip wheels bundle their own CUDA
#   (the same trick the Modal deployment used — "it just works" on any NVIDIA host driver).
# - Vast wraps every image on the host with its own ssh/tmux layer (apt-based), so a slim
#   Debian base is fine.
# - The template must set USE_SYSTEM_PYTHON=true so Vast's start_server.sh uses this
#   image's Python directly instead of building a venv.

FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    # Tell NVIDIA's container runtime to inject the GPU driver (CUDA-base images set
    # these; a slim Python base must ask explicitly or torch sees no GPU)
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---- Python deps (the pinned requirements are the source of truth) ----
# torch first, pinned to 2.6.0: its standard PyPI wheel bundles CUDA 12.4, which runs on
# ANY host driver >= 12.4. (Latest torch bundles CUDA 13 and silently falls back to CPU
# on the older drivers most GPU-cloud hosts still run — that cost us a day.)
# NB: installed from plain PyPI on purpose — pip rejects download.pytorch.org's index
# over a typing-extensions metadata-name quirk.
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ---- Bake the models into the HF cache (BiRefNet ~1 GB + SAM 2.1 large ~1 GB) ----
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('ZhengPeng7/BiRefNet', ignore_patterns=['*.onnx', '*.bin']); \
snapshot_download('facebook/sam2.1-hiera-large', ignore_patterns=['*.pt'])"

# ---- Build-time sanity gate: actually LOAD both models on CPU. If a dependency bump ----
# ---- ever breaks model loading again, the BUILD fails — not the GPU fleet.          ----
RUN python -c "\
from transformers import AutoModelForImageSegmentation, Sam2Model, Sam2Processor; \
m = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True); \
s = Sam2Model.from_pretrained('facebook/sam2.1-hiera-large'); \
p = Sam2Processor.from_pretrained('facebook/sam2.1-hiera-large'); \
print('models load OK')"

# ---- Bake the worker code as a git clone; the template's on-start does a `git pull` ----
# ---- so small code fixes ship without rebuilding the image.                          ----
RUN git clone https://github.com/Kartdavid/si-vast-pyworker /workspace/vast-pyworker

# Default command = freshen the code (so pure code fixes ship with a pod RESTART, no
# image rebuild), then run the model server: API on port 18000, protected by API_KEY.
# Offline-safe: if the pull fails, the baked copy runs. Vast ignores this CMD (its
# wrapper's entrypoint + on-start drive the PyWorker instead).
CMD ["bash", "-c", "cd /workspace/vast-pyworker && (git pull --ff-only || true) && exec python server.py"]
