import time
from io import StringIO
from threading import Thread
from typing import BinaryIO, Union

import whisperx
from whisperx.audio import N_SAMPLES
from whisperx.diarize import DiarizationPipeline
from whisperx.utils import ResultWriter, SubtitlesWriter, WriteJSON, WriteSRT, WriteTSV, WriteTXT, WriteVTT

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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def _load_diarizer(self):
        if self.model['diarize_model'] is None and CONFIG.HF_TOKEN != "":
            self.model['diarize_model'] = DiarizationPipeline(
                use_auth_token=CONFIG.HF_TOKEN,
                device=CONFIG.DEVICE  # keep GPU diarization; change to "cpu" if you ever want CPU
            )

    def _release_diarizer(self):
        if self.model['diarize_model'] is not None:
            del self.model['diarize_model']
            self.model['diarize_model'] = None
            if torch.cuda.is_available():
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
    def assign_speakers_by_overlap(self, turns, result):
        """
        Assign speaker labels from diarization turns to words/segments in `result`.
        Works with both pyannote and NeMo outputs.
        """
        if not result or "segments" not in result:
            return result

        for seg in result["segments"]:
            words = seg.get("words") or []
            items = words if words else [seg]

            for it in items:
                s0 = float(it.get("start", seg["start"]))
                s1 = float(it.get("end", seg["end"]))
                best_spk, best_overlap = None, 0.0
                for t in turns:
                    a0, a1 = t["start"], t["end"]
                    olap = max(0.0, min(s1, a1) - max(s0, a0))
                    if olap > best_overlap:
                        best_overlap, best_spk = olap, t["speaker"]
                if best_spk is not None:
                    it["speaker"] = best_spk

            # majority vote for the segment label
            labels = [it.get("speaker") for it in items if it.get("speaker") is not None]
            if labels:
                seg["speaker"] = max(set(labels), key=labels.count)

        return result

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
        if options and options.get("diarize", False):
            min_speakers = options.get("min_speakers", None)
            max_speakers = options.get("max_speakers", None)
            which = (options.get("diarizer") or "pyannote").lower()

            if which == "pyannote":
                # keep your current pyannote path; HF token required for that model
                if CONFIG.HF_TOKEN == "":
                    raise RuntimeError(
                        "pyannote diarizer selected but HF token is empty. Use diarizer=nemo or set HF token.")
                with self.model_lock:
                    self._load_diarizer()
                diarize_segments = self.model['diarize_model'](audio, min_speakers, max_speakers)

                # (Optional) your existing post-processing here (merge/split tweaks)
                result = whisperx.assign_word_speakers(diarize_segments, result)

                with self.model_lock:
                    self._release_diarizer()

            elif which == "nemo":
                # NeMo path — no HF token needed
                try:
                    from .nemo_adapter import diarize_with_nemo
                except Exception as e:
                    raise RuntimeError(f"NeMo adapter import failed: {e}")

                # choose device for NeMo (CPU is fine; GPU faster if available)
                try:
                    import torch
                    nemo_device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    nemo_device = "cpu"

                diarize_segments = diarize_with_nemo(
                    audio=audio,
                    sample_rate=16000,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    device=nemo_device,
                )

                # Assign speakers to words/segments by overlap (works universally)
                result = self.assign_speakers_by_overlap(diarize_segments, result)

            else:
                raise ValueError(f"Unknown diarizer backend: {which}")

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
