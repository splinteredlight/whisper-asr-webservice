import time
from io import StringIO
from threading import Thread
from typing import BinaryIO, Union

import numpy as np
import whisperx
from whisperx.audio import N_SAMPLES
from whisperx.diarize import DiarizationPipeline
from whisperx.utils import ResultWriter, SubtitlesWriter, WriteJSON, WriteSRT, WriteTSV, WriteTXT, WriteVTT

from app import split_channels
from app.asr_models.asr_model import ASRModel
from app.config import CONFIG

import gc, torch


class WhisperXASR(ASRModel):
    def __init__(self):
        super().__init__()
        self.model = {
            'whisperx': None,
            'diarize_model': None,
            'align_model': {}
        }

    def load_model(self):
        Thread(target=self.monitor_idleness, daemon=True).start()

    ########### --- new helpers --- ###################################
    def _load_whisper(self):
        if self.model['whisperx'] is None:
            asr_options = {"without_timestamps": False}
            self.model['whisperx'] = whisperx.load_model(
                CONFIG.MODEL_NAME,
                device=CONFIG.DEVICE,
                compute_type=CONFIG.MODEL_QUANTIZATION,
                asr_options=asr_options
            )

    def _release_whisper(self):
        if self.model['whisperx'] is not None:
            del self.model['whisperx']
            self.model['whisperx'] = None
            torch.cuda.empty_cache()
            gc.collect()

    def _load_diarizer(self):
        if self.model['diarize_model'] is None and CONFIG.HF_TOKEN != "":
            self.model['diarize_model'] = DiarizationPipeline(
                token=CONFIG.HF_TOKEN,  # renamed from use_auth_token in whisperx 3.8.x

                device=CONFIG.DEVICE  # keep GPU diarization; change to "cpu" if you ever want CPU
            )

    def _release_diarizer(self):
        if self.model['diarize_model'] is not None:
            del self.model['diarize_model']
            self.model['diarize_model'] = None
            torch.cuda.empty_cache()
            gc.collect()

    # --- Align model helpers -------------------------------------------------

    def _load_align_model(self, lang: str, device: str = None):
        """Lazy-load align model for a language into self.model['align_model'][lang]."""
        import whisperx
        dev = device or CONFIG.DEVICE
        with self.model_lock:
            if 'align_model' not in self.model:
                self.model['align_model'] = {}
            if lang not in self.model['align_model'] or self.model['align_model'][lang] is None:
                model_x, metadata = whisperx.load_align_model(language_code=lang, device=dev)
                self.model['align_model'][lang] = (model_x, metadata)

    def _release_align_model(self, lang: str = None):
        """Release one language or all align models and free CUDA."""
        import gc, torch
        with self.model_lock:
            if 'align_model' not in self.model:
                return
            if lang is None:
                # drop all
                for k in list(self.model['align_model'].keys()):
                    try:
                        m, md = self.model['align_model'][k]
                        # move off GPU if needed
                        try:
                            m.to("cpu")
                        except Exception:
                            pass
                    except Exception:
                        pass
                    self.model['align_model'][k] = None
                self.model['align_model'].clear()
            else:
                if lang in self.model['align_model'] and self.model['align_model'][lang] is not None:
                    try:
                        m, md = self.model['align_model'][lang]
                        try:
                            m.to("cpu")
                        except Exception:
                            pass
                    except Exception:
                        pass
                    self.model['align_model'][lang] = None
                    self.model['align_model'].pop(lang, None)
        # hard free
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    #####################################################################################
    def transcribe(
            self,
            audio,
            task: Union[str, None],
            language: Union[str, None],
            initial_prompt: Union[str, None],
            vad_filter: Union[bool, None],
            word_timestamps: Union[bool, None],
            options: Union[dict, None],
            output,
    ):
        self.last_activity_time = time.time()

        # ---- Split-channel input (see app/split_channels.py) ----
        # load_audio() hands us a (2, time) array only when SPLIT_CHANNEL_DIARIZATION
        # is on and the upload is stereo. ASR/alignment always use the mono downmix;
        # the per-channel signals are kept only if the channels are truly independent.
        mic_audio = remote_audio = None
        if isinstance(audio, np.ndarray) and audio.ndim == 2:
            stereo = audio
            audio = np.ascontiguousarray(stereo.mean(axis=0, dtype=np.float32))
            if options and options.get("diarize", False):
                independent, why = split_channels.channels_independent(stereo, CONFIG.SAMPLE_RATE)
                if independent:
                    mic_audio = np.ascontiguousarray(stereo[CONFIG.SPLIT_CHANNEL_MIC])
                    remote_audio = np.ascontiguousarray(stereo[1 - CONFIG.SPLIT_CHANNEL_MIC])
                    print(f"Split-channel diarization ON (mic=ch{CONFIG.SPLIT_CHANNEL_MIC}): {why}")
                else:
                    print(f"Split-channel diarization skipped, using mono: {why}")
            del stereo

        # 1) Load Whisper just-in-time
        with self.model_lock:
            if self.model is None:
                self.model = {'whisperx': None, 'diarize_model': None, 'align_model': {}}
            self._load_whisper()

        # ---- Transcription ----
        options_dict = {"task": task}
        if language:
            options_dict["language"] = language
        if initial_prompt:
            options_dict["initial_prompt"] = initial_prompt

        with self.model_lock:
            result = self.model['whisperx'].transcribe(audio, **options_dict)
        detected_lang = result.get("language", language or "en")

        # ---- ALIGNMENT (lazy load + caching policy) ----
        align_cache = (options or {}).get("align_cache", "none")  # "none" | "cpu" | "cuda"
        align_device = "cpu" if align_cache == "cpu" else CONFIG.DEVICE

        if result.get("segments"):
            # load lazily (on chosen device)
            self._load_align_model(detected_lang, device=align_device)
            model_x, metadata = self.model['align_model'][detected_lang]

            result = whisperx.align(
                result["segments"],
                model_x,
                metadata,
                audio,
                align_device,
                return_char_alignments=False
            )

            # handle caching policy
            if align_cache == "none":
                # drop immediately to reclaim VRAM
                self._release_align_model(detected_lang)
            elif align_cache == "cpu":
                # keep cached, but move to CPU to free VRAM
                try:
                    model_x.to("cpu")
                except Exception:
                    pass
                with self.model_lock:
                    self.model['align_model'][detected_lang] = (model_x, metadata)
                try:
                    import torch, gc
                    torch.cuda.empty_cache();
                    gc.collect()
                except Exception:
                    pass
            # else: "cuda" => keep as-is for reuse (fastest, uses VRAM)

        # 2) Whisper is no longer needed → free its VRAM before diarization
        with self.model_lock:
            self._release_whisper()

        # ---- Diarization (optional) ----
        if options and options.get("diarize", False) and CONFIG.HF_TOKEN != "":
            min_speakers = options.get("min_speakers", None)
            max_speakers = options.get("max_speakers", None)

            with self.model_lock:
                self._load_diarizer()

            # Keyword args are required: the 2nd positional parameter of
            # DiarizationPipeline.__call__ is num_speakers, so passing these
            # positionally forced the speaker count to exactly min_speakers.
            if remote_audio is not None:
                # The local speaker is not in the remote channel, so the caller's
                # speaker-count hints (which include them) shrink by one.
                r_min = max(1, min_speakers - 1) if min_speakers else None
                r_max = max(1, max_speakers - 1) if max_speakers else None
                diarize_segments = self.model['diarize_model'](
                    remote_audio, min_speakers=r_min, max_speakers=r_max
                )
                result = split_channels.assign_split_channel_speakers(
                    diarize_segments, result, mic_audio, remote_audio, CONFIG.SAMPLE_RATE
                )
            else:
                diarize_segments = self.model['diarize_model'](
                    audio, min_speakers=min_speakers, max_speakers=max_speakers
                )
                result = whisperx.assign_word_speakers(diarize_segments, result)

            # Free diarizer VRAM immediately after use
            with self.model_lock:
                self._release_diarizer()

        result["language"] = detected_lang

        # ---- Output writer (unchanged) ----
        output_file = StringIO()
        self.write_result(result, output_file, output)
        output_file.seek(0)
        return output_file

    def language_detection(self, audio):
        with self.model_lock:
            if self.model is None:
                self.model = {'whisperx': None, 'diarize_model': None, 'align_model': {}}
            self._load_whisper()
            if audio.shape[0] < N_SAMPLES:
                print("Warning: audio is shorter than 30s, language detection may be inaccurate.")
            results = self.model['whisperx'].model.detect_language(audio)
            language = results[0]
            language_probability = round(float(results[1]), 2)
            print(f"Detected language: {language} ({language_probability}) in first 30s of audio...")
            # optional: free right after detect
            self._release_whisper()
        return language, language_probability

    def write_result(self, result: dict, file: BinaryIO, output: Union[str, None]):
        default_options = {
            "max_line_width": CONFIG.SUBTITLE_MAX_LINE_WIDTH,
            "max_line_count": CONFIG.SUBTITLE_MAX_LINE_COUNT,
            "highlight_words": CONFIG.SUBTITLE_HIGHLIGHT_WORDS
        }

        if output == "srt":
            WriteSRT(SubtitlesWriter).write_result(result, file=file, options=default_options)
        elif output == "vtt":
            WriteVTT(SubtitlesWriter).write_result(result, file=file, options=default_options)
        elif output == "tsv":
            WriteTSV(ResultWriter).write_result(result, file=file, options=default_options)
        elif output == "json":
            WriteJSON(ResultWriter).write_result(result, file=file, options=default_options)
        else:
            WriteTXT(ResultWriter).write_result(result, file=file, options=default_options)