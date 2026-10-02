import json
import os
import shutil
import tempfile
from dataclasses import asdict
from typing import BinaryIO, TextIO

import ffmpeg
import numpy as np
from faster_whisper.utils import format_timestamp

from app.config import CONFIG


class ResultWriter:
    extension: str

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def __call__(self, result: dict, audio_path: str):
        audio_basename = os.path.basename(audio_path)
        output_path = os.path.join(self.output_dir, audio_basename + "." + self.extension)

        with open(output_path, "w", encoding="utf-8") as f:
            self.write_result(result, file=f)

    def write_result(self, result: dict, file: TextIO):
        raise NotImplementedError


class WriteTXT(ResultWriter):
    extension: str = "txt"

    def write_result(self, result: dict, file: TextIO):
        for segment in result["segments"]:
            print(segment.text.strip(), file=file, flush=True)


class WriteVTT(ResultWriter):
    extension: str = "vtt"

    def write_result(self, result: dict, file: TextIO):
        print("WEBVTT\n", file=file)
        for segment in result["segments"]:
            print(
                f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n"
                f"{segment.text.strip().replace('-->', '->')}\n",
                file=file,
                flush=True,
            )


class WriteSRT(ResultWriter):
    extension: str = "srt"

    def write_result(self, result: dict, file: TextIO):
        for i, segment in enumerate(result["segments"], start=1):
            # write srt lines
            print(
                f"{i}\n"
                f"{format_timestamp(segment.start, always_include_hours=True, decimal_marker=',')} --> "
                f"{format_timestamp(segment.end, always_include_hours=True, decimal_marker=',')}\n"
                f"{segment.text.strip().replace('-->', '->')}\n",
                file=file,
                flush=True,
            )


class WriteTSV(ResultWriter):
    """
    Write a transcript to a file in TSV (tab-separated values) format containing lines like:
    <start time in integer milliseconds>\t<end time in integer milliseconds>\t<transcript text>

    Using integer milliseconds as start and end times means there's no chance of interference from
    an environment setting a language encoding that causes the decimal in a floating point number
    to appear as a comma; also is faster and more efficient to parse & store, e.g., in C++.
    """

    extension: str = "tsv"

    def write_result(self, result: dict, file: TextIO):
        print("start", "end", "text", sep="\t", file=file)
        for segment in result["segments"]:
            print(round(1000 * segment.start), file=file, end="\t")
            print(round(1000 * segment.end), file=file, end="\t")
            print(segment.text.strip().replace("\t", " "), file=file, flush=True)


class WriteJSON(ResultWriter):
    extension: str = "json"

    def write_result(self, result: dict, file: TextIO):
        if "segments" in result:
            result["segments"] = [asdict(segment) for segment in result["segments"]]
        json.dump(result, file)


def _audio_channel_count(path: str) -> int:
    """Channel count of the first audio stream, or 0 if ffprobe can't tell."""
    try:
        for stream in ffmpeg.probe(path).get("streams", []):
            if stream.get("codec_type") == "audio":
                return int(stream.get("channels") or 0)
    except Exception:
        pass
    return 0


def load_audio(file: BinaryIO, encode=True, sr: int = CONFIG.SAMPLE_RATE, keep_stereo: bool = False):
    """
    Open an audio file object and read as mono waveform, resampling as necessary.
    With ``keep_stereo=True`` a 2-channel input is returned as a (2, time) array
    instead of being downmixed (used for split-channel diarization).
    Modified from https://github.com/openai/whisper/blob/main/whisper/audio.py to accept a file object
    Parameters
    ----------
    file: BinaryIO
        The audio file like object
    encode: Boolean
        If true, encode audio stream to WAV before sending to whisper
    sr: int
        The sample rate to resample the audio if necessary
    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    if encode:
        # Decode from a temp file rather than stdin: MP4/MOV files whose moov
        # atom sits at the end (e.g. Facebook/phone downloads) can't be parsed
        # from a non-seekable pipe, so ffmpeg silently emits zero samples and
        # the empty array later crashes the VAD.
        with tempfile.NamedTemporaryFile(suffix=".media", delete=True) as tmp:
            shutil.copyfileobj(file, tmp)
            tmp.flush()
            channels = 1
            # Unknown channel count (0) also decodes as stereo: a mono source is
            # then duplicated onto both channels, which the independence check
            # downstream recognises and treats as mono.
            if keep_stereo and _audio_channel_count(tmp.name) in (0, 2):
                channels = 2
            try:
                out, _ = (
                    ffmpeg.input(tmp.name, threads=0)
                    .output("-", format="s16le", acodec="pcm_s16le", ac=channels, ar=sr)
                    .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True)
                )
            except ffmpeg.Error as e:
                raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    else:
        out = file.read()
        channels = 1

    audio = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    if audio.size == 0:
        raise RuntimeError("Failed to load audio: no decodable audio stream found in the uploaded file")
    if channels == 2:
        # interleaved L R L R ... -> (2, time), C-contiguous rows
        audio = np.ascontiguousarray(audio[: (audio.size // 2) * 2].reshape(-1, 2).T)
    return audio
