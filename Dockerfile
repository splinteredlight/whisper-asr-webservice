FROM onerahmet/ffmpeg:n7.1 AS ffmpeg
FROM swaggerapi/swagger-ui:v5.9.1 AS swagger-ui
FROM python:3.10-bookworm

LABEL org.opencontainers.image.source="https://github.com/ahmetoner/whisper-asr-webservice"

ENV POETRY_VENV=/app/.venv
ENV PATH="${PATH}:${POETRY_VENV}/bin"

# --- Install Poetry ---
RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==2.1.3

WORKDIR /app
COPY . .
COPY --from=ffmpeg /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg
COPY --from=swagger-ui /usr/share/nginx/html/swagger-ui.css swagger-ui-assets/swagger-ui.css
COPY --from=swagger-ui /usr/share/nginx/html/swagger-ui-bundle.js swagger-ui-assets/swagger-ui-bundle.js

# ---- Poetry install (CPU by default; set POETRY_EXTRAS=gpu for CUDA wheels) ----
ARG POETRY_EXTRAS=cpu
RUN poetry config virtualenvs.in-project true \
 && poetry install --no-interaction --no-ansi --extras ${POETRY_EXTRAS}

# ---- Add NeMo diarization (CPU is fine; works with CUDA too if present) ----
# If you used the "no-soundfile" nemo_adapter I gave you, you only need nemo_toolkit:
RUN $POETRY_VENV/bin/pip install --no-cache-dir nemo_toolkit

# If you use the SoundFile-based adapter instead, uncomment these two lines:
# RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*
# RUN $POETRY_VENV/bin/pip install --no-cache-dir soundfile

EXPOSE 9000
ENTRYPOINT ["whisper-asr-webservice"]
