FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

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

# Create venv
RUN python3 -m venv ${POETRY_VENV} \
 && ${POETRY_VENV}/bin/pip install --upgrade pip setuptools wheel

# Put us in /app
WORKDIR /app

# Copy metadata first (better layer caching); ensure README.md exists
COPY pyproject.toml poetry.lock README.md /app/
# Copy the source code
COPY app /app/app

# Install your package so distribution metadata + console script are created
RUN ${POETRY_VENV}/bin/pip install --no-cache-dir -e /app

# (Optional but recommended) pin CUDA wheels after base install
RUN ${POETRY_VENV}/bin/pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.7.1+cu126 torchaudio==2.7.1+cu126 \
 && ${POETRY_VENV}/bin/pip install --no-cache-dir "nemo_toolkit[asr]" webdataset lhotse soundfile hydra-core omegaconf

EXPOSE 9000
ENTRYPOINT ["/app/.venv/bin/whisper-asr-webservice"]








## --- Python base & Poetry ---
#RUN python3 -m venv ${POETRY_VENV} \
# && ${POETRY_VENV}/bin/pip install -U pip setuptools wheel "poetry==1.8.3"
#
#WORKDIR /app
#
## Copy lock context first for caching
#COPY pyproject.toml poetry.lock ./
#
## Show what we're building with
#RUN echo "----- pyproject in image -----" && sed -n '1,160p' pyproject.toml && echo "----------------------------------"
#
## Poetry should install into our venv (no extra venvs)
#RUN poetry config virtualenvs.create false \
# && poetry config virtualenvs.in-project true
#
## Lock refresh to match this pyproject (no upgrades)
#RUN poetry lock --no-update || poetry lock
#
## Install project deps from lock
#RUN poetry install --no-interaction --no-ansi
#
## Install CUDA Torch wheels explicitly (use /whl/cpu for CPU builds)
#RUN ${POETRY_VENV}/bin/pip install --no-cache-dir \
#    --index-url https://download.pytorch.org/whl/cu126 \
#    torch==2.7.1+cu126 torchaudio==2.7.1+cu126
#
## NeMo diarization + helpers
#RUN ${POETRY_VENV}/bin/pip install --no-cache-dir "nemo_toolkit[asr]" webdataset lhotse soundfile \
# && ${POETRY_VENV}/bin/pip install --no-cache-dir hydra-core omegaconf
#
## Copy the rest of the app
#COPY . /app
#
## Install your package so the console script is created in /app/.venv/bin
#RUN ${POETRY_VENV}/bin/pip install --no-cache-dir -e /app
#
#EXPOSE 9000
#ENTRYPOINT ["/app/.venv/bin/whisper-asr-webservice"]
