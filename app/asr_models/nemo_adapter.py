# app/asr_models/nemo_adapter.py
from __future__ import annotations
from typing import List, Dict, Optional, Any
import os, json, sys, types, tempfile
import numpy as np

def nemo_available() -> bool:
    try:
        import nemo  # noqa: F401
        return True
    except Exception:
        return False

# ---- WebDataset shim (some NeMo builds import nemo.utils.webdataset) ----
try:
    import webdataset as _wds
    if "nemo.utils" not in sys.modules:
        sys.modules["nemo.utils"] = types.ModuleType("nemo.utils")
    sys.modules["nemo.utils.webdataset"] = _wds
except Exception:
    pass


def _nemo_write_manifest(wav_path: str, manifest_path: str, num_spk: Optional[int] = None) -> None:
    """
    NeMo diarizer manifest: JSONL with keys:
      audio_filepath, offset, duration, label, text, num_speakers, rttm_filepath, uniq_id
    """
    rec = {
        "audio_filepath": wav_path,
        "offset": 0.0,
        "duration": None,  # let NeMo infer
        "label": "infer",
        "text": "-",
        "num_speakers": int(num_spk) if num_spk is not None else None,
        "rttm_filepath": None,
        "uniq_id": os.path.splitext(os.path.basename(wav_path))[0],
    }
    with open(manifest_path, "w") as f:
        f.write(json.dumps(rec) + "\n")


def _ensure_wav_path_from_array(audio_np: np.ndarray, sample_rate: int) -> str:
    """Write NumPy audio to a temp WAV (mono float32) and return its path."""
    import soundfile as sf
    # ensure shape [samples] or [samples, channels] -> mono
    if audio_np.ndim == 2 and audio_np.shape[1] > 1:
        audio_np = np.mean(audio_np, axis=1)
    audio_np = audio_np.astype("float32", copy=False)
    fd, tmp = tempfile.mkstemp(prefix="nemo_audio_", suffix=".wav")
    os.close(fd)
    sf.write(tmp, audio_np, samplerate=int(sample_rate), subtype="PCM_16")
    return tmp


def diarize_with_nemo(
    wav_path: Optional[str] = None,
    audio: Optional[Any] = None,               # may be NumPy array
    sample_rate: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    diarizer_model: str = "diar_msdd_telephonic",
    device: Optional[str] = None,              # allow override or env
    **kwargs,                                   # absorb extras (channels, etc.)
) -> List[Dict]:
    """
    Run NeMo MSDD diarizer and return turns:
      [{"start": float, "end": float, "speaker": "SPEAKER_00"}, ...]
    Accepts either `wav_path` (str) or in-memory `audio` (NumPy) + `sample_rate`.
    """
    if device is None:
        device = os.getenv("NEMO_DEVICE") or os.getenv("DIARIZER_DEVICE") or "cuda"

    # Normalize to a filesystem path
    tmp_file_to_cleanup: Optional[str] = None
    if wav_path is None and audio is not None:
        # Expecting a NumPy array
        if not isinstance(audio, np.ndarray):
            raise ValueError("diarize_with_nemo: `audio` must be a NumPy array if no `wav_path` is provided")
        if sample_rate is None:
            raise ValueError("diarize_with_nemo: `sample_rate` is required when passing `audio` array")
        wav_path = _ensure_wav_path_from_array(audio, sample_rate)
        tmp_file_to_cleanup = wav_path

    if wav_path is None or not isinstance(wav_path, str):
        raise ValueError("diarize_with_nemo: provide `wav_path` (str) or `audio` (NumPy) + `sample_rate`")

    if not nemo_available():
        raise RuntimeError("NeMo not installed. Try: pip install 'nemo_toolkit[asr]' soundfile webdataset lhotse")

    from tempfile import TemporaryDirectory
    from omegaconf import OmegaConf
    from nemo.collections.asr.models.msdd_models import NeuralDiarizer

    # Load model on requested device
    model = NeuralDiarizer.from_pretrained(model_name=diarizer_model, map_location=device)

    # Configure fixed speaker count (NeMo prefers clustering cfg)
    fixed = None
    if min_speakers is not None and max_speakers is not None and int(min_speakers) == int(max_speakers):
        fixed = int(min_speakers)
        try:
            if not hasattr(model, "cfg") or model.cfg is None:
                model.cfg = OmegaConf.create({})
            if not hasattr(model.cfg, "diarizer") or model.cfg.diarizer is None:
                model.cfg.diarizer = OmegaConf.create({})
            if not hasattr(model.cfg.diarizer, "clustering") or model.cfg.diarizer.clustering is None:
                model.cfg.diarizer.clustering = OmegaConf.create({})
            model.cfg.diarizer.clustering.min_num_speakers = fixed
            model.cfg.diarizer.clustering.max_num_speakers = fixed
        except Exception:
            pass  # non-fatal

    try:
        with TemporaryDirectory(prefix="nemo_diar_") as tmpd:
            manifest = os.path.join(tmpd, "manifest.json")
            out_dir = os.path.join(tmpd, "out")
            os.makedirs(out_dir, exist_ok=True)

            _nemo_write_manifest(wav_path, manifest, num_spk=fixed)

            # Newer NeMo API: set cfg and call with NO args
            # Newer NeMo API: set cfg and call with NO args
            pred = None
            if not hasattr(model, "cfg") or model.cfg is None:
                model.cfg = OmegaConf.create({})
            if not hasattr(model.cfg, "diarizer") or model.cfg.diarizer is None:
                model.cfg.diarizer = OmegaConf.create({})

            # Set both config fields (public) and private attrs some versions expect
            model.cfg.diarizer.manifest_path = manifest
            model.cfg.diarizer.out_dir = out_dir
            # Private/legacy fields NeMo accesses internally:
            setattr(model, "_manifest_filepath", manifest)
            setattr(model, "_out_dir", out_dir)
            setattr(model, "out_dir", out_dir)  # some codepaths look here

            # Always use the no-arg API on this NeMo version
            pred = model.diarize()

            # RTTM retrieval (from return object or generated files)
            rttm_text = getattr(pred, "rttm", None) or getattr(pred, "rttm_str", None)

            if not rttm_text:
                # NeMo commonly writes to <out_dir>/pred_rttms/<basename>.rttm
                candidates = []
                base = os.path.splitext(os.path.basename(wav_path))[0]
                pred_dir = os.path.join(out_dir, "pred_rttms")
                for p in (
                        os.path.join(out_dir, f"{base}.rttm"),
                        os.path.join(pred_dir, f"{base}.rttm"),
                ):
                    if os.path.exists(p):
                        candidates.append(p)

                # Fallback: any .rttm written under out_dir
                if not candidates and os.path.isdir(out_dir):
                    for root, _, files in os.walk(out_dir):
                        for fname in files:
                            if fname.lower().endswith(".rttm"):
                                candidates.append(os.path.join(root, fname))

                if candidates:
                    with open(candidates[0], "r") as f:
                        rttm_text = f.read()

            if not rttm_text:
                raise RuntimeError("NeMo diarizer did not produce RTTM text (version/config mismatch).")

            # Parse RTTM
            final_rttm = os.path.join(tmpd, "final.rttm")
            with open(final_rttm, "w") as f:
                f.write(rttm_text)

            return _parse_rttm_to_turns(final_rttm)
    finally:
        # clean up temp WAV if we created one
        if tmp_file_to_cleanup and os.path.exists(tmp_file_to_cleanup):
            try:
                os.remove(tmp_file_to_cleanup)
            except Exception:
                pass


def _parse_rttm_to_turns(path: str) -> List[Dict]:
    """
    RTTM SPEAKER line format:
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
            dur = float(parts[4])
            spk = parts[7]
            turns.append({"start": start, "end": start + dur, "speaker": spk})
    turns.sort(key=lambda d: (d["start"], d["end"]))
    return turns
