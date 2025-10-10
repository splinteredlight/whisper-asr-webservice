import importlib.metadata
import os
from os import path
from typing import Annotated, Optional, Union
from urllib.parse import quote

import click
import uvicorn
from fastapi import FastAPI, File, Query, UploadFile, applications
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from whisper import tokenizer

from app.config import CONFIG
from app.factory.asr_model_factory import ASRModelFactory
from app.utils import load_audio

asr_model = ASRModelFactory.create_asr_model()
# asr_model.load_model()

LANGUAGE_CODES = sorted(tokenizer.LANGUAGES.keys())

projectMetadata = importlib.metadata.metadata("whisper-asr-webservice")
app = FastAPI(
    title=projectMetadata["Name"].title().replace("-", " "),
    description=projectMetadata["Summary"],
    version=projectMetadata["Version"],
    contact={"url": projectMetadata["Home-page"]},
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    license_info={"name": "MIT License", "url": "https://github.com/ahmetoner/whisper-asr-webservice/blob/main/LICENCE"},
)

assets_path = os.getcwd() + "/swagger-ui-assets"
if path.exists(assets_path + "/swagger-ui.css") and path.exists(assets_path + "/swagger-ui-bundle.js"):
    app.mount("/assets", StaticFiles(directory=assets_path), name="static")

    def swagger_monkey_patch(*args, **kwargs):
        return get_swagger_ui_html(
            *args,
            **kwargs,
            swagger_favicon_url="",
            swagger_css_url="/assets/swagger-ui.css",
            swagger_js_url="/assets/swagger-ui-bundle.js",
        )

    applications.get_swagger_ui_html = swagger_monkey_patch


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return "/docs"


@app.post("/asr", tags=["Endpoints"])
async def asr(
    audio_file: UploadFile = File(...),  # noqa: B008
    encode: bool = Query(default=True, description="Encode audio first through ffmpeg"),
    task: Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
    language: Union[str, None] = Query(default=None, enum=LANGUAGE_CODES),
    initial_prompt: Union[str, None] = Query(default=None),

    vad_filter: Annotated[
        bool | None,
        Query(
            description="Enable the voice activity detection (VAD) to filter out parts of the audio without speech",
            include_in_schema=(True if CONFIG.ASR_ENGINE == "faster_whisper" else False),
        ),
    ] = False,
    word_timestamps: bool = Query(
        default=False,
        description="Word level timestamps",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "faster_whisper" else False),
    ),

    # ---- diarization controls ----
    diarize: bool = Query(
        default=False,
        description="Diarize the input",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),  # allow even without HF token (for NeMo)
    ),
    diarizer: str = Query(
        default="pyannote",
        regex="^(pyannote|nemo)$",
        description="Choose diarization backend",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),
    min_speakers: Union[int, None] = Query(
        default=None,
        description="Min speakers in this file",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),
    max_speakers: Union[int, None] = Query(
        default=None,
        description="Max speakers in this file",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),

    # ---- alignment caching policy (for WhisperX align model) ----
    align_cache: str = Query(
        default="none",
        regex="^(none|cpu|cuda)$",
        description="Alignment model caching: none (default), cpu (keep in RAM), or cuda (keep on GPU).",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),

    # (optional) simple diarization post-process knobs; safe defaults
    min_speech_duration: float = Query(
        default=0.35,
        description="Prune tiny speech islands (sec)",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),
    merge_gap: float = Query(
        default=0.30,
        description="Merge same-speaker turns if gap ≤ this (sec)",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),
    max_turn_sec: float = Query(
        default=10.0,
        description="Split turns longer than this (sec)",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),
    refine_embeddings: bool = Query(
        default=True,
        description="Refine long turns with CPU embeddings (helps separate similar voices)",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),

    output: Union[str, None] = Query(default="txt", enum=["txt", "vtt", "srt", "tsv", "json"]),
):
    audio = load_audio(audio_file.file, encode)

    opts = {
        "diarize": diarize,
        "diarizer": diarizer,                  # NEW
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "align_cache": align_cache,            # NEW

        # post-process knobs (engine will ignore if not used)
        "min_speech_duration": min_speech_duration,
        "merge_gap": merge_gap,
        "max_turn_sec": max_turn_sec,
        "refine_embeddings": refine_embeddings,
    }

    result = asr_model.transcribe(
        audio,
        task,
        language,
        initial_prompt,
        vad_filter,
        word_timestamps,
        opts,
        output,
    )
    return StreamingResponse(
        result,
        media_type="text/plain",
        headers={
            "Asr-Engine": CONFIG.ASR_ENGINE,
            "Content-Disposition": f'attachment; filename="{quote(audio_file.filename)}.{output}"',
        },
    )



@app.post("/detect-language", tags=["Endpoints"])
async def detect_language(
    audio_file: UploadFile = File(...),  # noqa: B008
    encode: bool = Query(default=True, description="Encode audio first through FFmpeg"),
):
    detected_lang_code, confidence = asr_model.language_detection(load_audio(audio_file.file, encode))
    return {
        "detected_language": tokenizer.LANGUAGES[detected_lang_code],
        "language_code": detected_lang_code,
        "confidence": confidence,
    }


@click.command()
@click.option(
    "-h",
    "--host",
    metavar="HOST",
    default="0.0.0.0",
    help="Host for the webservice (default: 0.0.0.0)",
)
@click.option(
    "-p",
    "--port",
    metavar="PORT",
    default=9000,
    help="Port for the webservice (default: 9000)",
)
@click.version_option(version=projectMetadata["Version"])
def start(host: str, port: Optional[int] = None):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start()
