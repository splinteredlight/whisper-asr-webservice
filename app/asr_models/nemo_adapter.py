# app/asr_models/nemo_adapter.py
from __future__ import annotations
from typing import List, Dict, Optional

def nemo_available() -> bool:
    try:
        import nemo  # noqa: F401
        return True
    except Exception:
        return False

# --- WebDataset shim so NeMo can import "nemo.utils.webdataset" safely ----
# Some NeMo builds import webdataset via a nested module path; expose it here.
try:
    import sys, types
    import webdataset as _wds
    if "nemo.utils" not in sys.modules:
        sys.modules["nemo.utils"] = types.ModuleType("nemo.utils")
    sys.modules["nemo.utils.webdataset"] = _wds
except Exception:
    # If this shim fails, NeMo might still work; fail later with a clear error.
    pass


def diarize_with_nemo(
    audio: "np.ndarray",           # mono float32 PCM
    sample_rate: int,              # e.g., 16000
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    device: str = "cpu",           # "cpu" or "cuda"
    diarizer_model: str = "diar_msdd_telephonic",
) -> List[Dict]:
    """
    Returns diarization turns as dicts:
        [{"start": float, "end": float, "speaker": "SPEAKER_00"}, ...]
    """
    if not nemo_available():
        raise RuntimeError("NeMo not installed. Try: pip install 'nemo_toolkit[asr]' soundfile webdataset lhotse")

    import os, tempfile, shutil
    try:
        import numpy as np
        import soundfile as sf
    except Exception as e:
        raise RuntimeError(f"Missing audio deps: {e}. Install 'soundfile' and system libsndfile.") from e

    # 1) write audio to a temp wav NeMo can read
    tmpdir = tempfile.mkdtemp(prefix="nemo_diar_")
    wav_path = os.path.join(tmpdir, "audio.wav")
    rttm_path = os.path.join(tmpdir, "out.rttm")
    sf.write(wav_path, audio, sample_rate, subtype="PCM_16")

    # 2) run NeMo MSDD diarizer and extract RTTM
    try:
        from nemo.collections.asr.models.msdd_models import NeuralDiarizer

        model = NeuralDiarizer.from_pretrained(
            model_name=diarizer_model,
            map_location=device
        )

        # If speaker count is known and fixed, pass it (stabilizes clustering)
        num_spk = min_speakers if (min_speakers is not None and min_speakers == max_speakers) else None

        pred = model.diarize([wav_path], num_speakers=num_spk)
        # Some versions return .rttm (string), others a dict => be defensive
        rttm_text = getattr(pred, "rttm", None) or getattr(pred, "rttm_str", None)
        if not rttm_text:
            # fall back: if a file path is returned or an object with .write_rttm(...)
            try:
                rttm_text = pred if isinstance(pred, str) else None
            except Exception:
                pass
        if not rttm_text:
            raise RuntimeError("NeMo diarizer did not produce RTTM text; version mismatch?")

        with open(rttm_path, "w") as f:
            f.write(rttm_text)

        turns = _parse_rttm_to_turns(rttm_path)
        return turns

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _parse_rttm_to_turns(path: str) -> List[Dict]:
    """
    RTTM line format (SPEAKER rows):
    SPEAKER <uri> <chan> <start> <dur> <ortho> <stype> <name> <conf> <slat>
    """
    turns: List[Dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("SPEAKER"):
                continue
            parts = line.split()
            start = float(parts[3])
            dur   = float(parts[4])
            spk   = parts[7]
            turns.append({"start": start, "end": start + dur, "speaker": spk})
    turns.sort(key=lambda d: (d["start"], d["end"]))
    return turns
