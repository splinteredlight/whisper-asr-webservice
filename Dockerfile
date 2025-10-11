FROM nvidia/cuda:12.6.3-base-ubuntu22.04

LABEL org.opencontainers.image.source="https://github.com/ahmetoner/whisper-asr-webservice"

ENV PYTHON_VERSION=3.10
ENV POETRY_VENV=/app/.venv
ENV PATH="${PATH}:${POETRY_VENV}/bin"
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV HF_HOME=/root/.cache/huggingface
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1

# --- System deps (build + audio) ---
RUN apt-get -qq update && apt-get -qq install -y --no-install-recommends \
    python3 python3-venv python3-pip python${PYTHON_VERSION}-dev \
    build-essential git curl ca-certificates \
    ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# --- Python base & Poetry ---
RUN python3 -m venv ${POETRY_VENV} \
 && ${POETRY_VENV}/bin/pip install -U pip setuptools wheel "poetry==1.8.3"

WORKDIR /app

# Copy lock context first for caching
COPY pyproject.toml poetry.lock ./

# Show what we're building with
RUN echo "----- pyproject in image -----" && sed -n '1,160p' pyproject.toml && echo "----------------------------------"

# Poetry should install into our venv (no extra venvs)
RUN poetry config virtualenvs.create false \
 && poetry config virtualenvs.in-project true

# Lock refresh to match this pyproject (no upgrades)
RUN poetry lock --no-update || poetry lock

# Install project deps from lock
RUN poetry install --no-interaction --no-ansi

# Install CUDA Torch wheels explicitly (use /whl/cpu for CPU builds)
RUN ${POETRY_VENV}/bin/pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.7.1+cu126 torchaudio==2.7.1+cu126

# NeMo diarization + helpers
RUN ${POETRY_VENV}/bin/pip install --no-cache-dir "nemo_toolkit[asr]" webdataset lhotse soundfile \
 && ${POETRY_VENV}/bin/pip install --no-cache-dir hydra-core omegaconf

# Copy the rest of the app
COPY . /app

EXPOSE 9000
ENTRYPOINT ["whisper-asr-webservice"]
