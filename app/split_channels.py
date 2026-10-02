"""Split-channel ("two-channel") diarization support.

A recording that keeps the local microphone on one channel and the remote
meeting audio (Zoom/Teams system or tab audio) on the other makes speaker
attribution for the local speaker deterministic:

* ASR + alignment run on the mono downmix, exactly as for any other file;
* the diarizer runs on the REMOTE channel only, so the local speaker never
  takes part in clustering and can never be merged with a remote voice;
* every word is attributed to the local speaker when the mic channel clearly
  dominates the remote channel over that word's time span; otherwise it keeps
  the label the remote-channel diarization gave it.

The local speaker is always emitted as ``SPEAKER_00``; remote speakers are
renumbered from ``SPEAKER_01`` upward.

A stereo file only takes this path when its two channels are *independent*
(see :func:`channels_independent`).  An ordinary stereo mix, where both
channels carry the same programme (an OBS recording, a podcast, a mono file
up-mixed to stereo), falls back to the normal mono pipeline.

Headphones are assumed on the mic side.  Remote voices bleeding into the mic
from loudspeakers do NOT trip the independence check (the mic's own speech
dominates its envelope); they are handled per word instead, because the mic
only claims a word when it beats the remote channel by ~6 dB, which bleed at
typical levels (-12 dB and below) never does.
"""
import re
from typing import Tuple

import numpy as np
import pandas as pd

import whisperx

MIC_LABEL = "SPEAKER_00"

FRAME_SEC = 0.01           # hop of the per-channel RMS envelope used per word
DOMINANCE_RATIO = 2.0      # ~6 dB: mic must beat remote by this to claim a word
MIC_FLOOR_RMS = 0.005      # ~-46 dBFS: below this the mic channel is "silent"
MIN_WORD_WINDOW = 0.10     # seconds; very short words are widened to this

INDEPENDENCE_FRAME_SEC = 0.05
ENVELOPE_CORR_MAX = 0.5    # above this the two channels carry the same programme
ACTIVE_FRAME_RMS = 0.003   # ~-50 dBFS: a frame above this counts as "active"
MIN_ACTIVE_FRACTION = 0.02 # each channel must be active at least this often

MIN_RUN_WORDS = 2          # a speaker run shorter than this (and than MIN_RUN_SEC)
MIN_RUN_SEC = 1.0          # is absorbed into its neighbour instead of split out

_LABEL_RE = re.compile(r"^SPEAKER_(\d+)$")


def frame_rms(x: np.ndarray, sr: int, frame_sec: float) -> np.ndarray:
    hop = max(1, int(sr * frame_sec))
    n = (len(x) // hop) * hop
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    frames = np.asarray(x[:n], dtype=np.float32).reshape(-1, hop)
    return np.sqrt(np.mean(frames * frames, axis=1))


def channels_independent(stereo: np.ndarray, sr: int) -> Tuple[bool, str]:
    """Decide whether a (2, time) array is a mic/remote split rather than a
    normal stereo mix.  Returns ``(independent, reason)``."""
    if stereo.ndim != 2 or stereo.shape[0] != 2:
        return False, "not a 2-channel array"
    env0 = frame_rms(stereo[0], sr, INDEPENDENCE_FRAME_SEC)
    env1 = frame_rms(stereo[1], sr, INDEPENDENCE_FRAME_SEC)
    if len(env0) < 40:  # < 2 s of audio
        return False, "too short to judge"
    active0 = float(np.mean(env0 > ACTIVE_FRAME_RMS))
    active1 = float(np.mean(env1 > ACTIVE_FRAME_RMS))
    if active0 < MIN_ACTIVE_FRACTION or active1 < MIN_ACTIVE_FRACTION:
        return False, f"a channel is (near-)silent (active fractions {active0:.3f}/{active1:.3f})"
    if float(np.std(env0)) == 0.0 or float(np.std(env1)) == 0.0:
        return False, "flat envelope"
    corr = float(np.corrcoef(env0, env1)[0, 1])
    if not np.isfinite(corr):
        return False, "undefined envelope correlation"
    if corr > ENVELOPE_CORR_MAX:
        return False, f"channels carry the same programme (envelope corr {corr:.2f})"
    return True, f"envelope corr {corr:.2f}, active fractions {active0:.2f}/{active1:.2f}"


class _Envelope:
    """Per-channel RMS envelope with O(1) window means via a cumulative sum."""

    def __init__(self, x: np.ndarray, sr: int, frame_sec: float = FRAME_SEC):
        self.hop = frame_sec
        env = frame_rms(x, sr, frame_sec)
        self.n = len(env)
        self.csum = np.concatenate([[0.0], np.cumsum(env, dtype=np.float64)])

    def mean(self, start: float, end: float) -> float:
        if self.n == 0:
            return 0.0
        i0 = int(start / self.hop)
        i1 = int(np.ceil(end / self.hop))
        i0 = min(max(i0, 0), self.n - 1)
        i1 = min(max(i1, i0 + 1), self.n)
        return float((self.csum[i1] - self.csum[i0]) / (i1 - i0))


def _mic_dominates(mic_env: _Envelope, rem_env: _Envelope, start: float, end: float) -> bool:
    if end - start < MIN_WORD_WINDOW:
        mid = (start + end) / 2.0
        start, end = mid - MIN_WORD_WINDOW / 2.0, mid + MIN_WORD_WINDOW / 2.0
    m = mic_env.mean(start, end)
    if m < MIC_FLOOR_RMS:
        return False
    return m > DOMINANCE_RATIO * rem_env.mean(start, end)


def _shift_label(label):
    """SPEAKER_03 -> SPEAKER_04 so that SPEAKER_00 is free for the mic."""
    m = _LABEL_RE.match(str(label))
    if not m:
        return label
    return f"SPEAKER_{int(m.group(1)) + 1:02d}"


def assign_split_channel_speakers(diarize_df: pd.DataFrame, result: dict,
                                  mic: np.ndarray, remote: np.ndarray, sr: int) -> dict:
    """Combine remote-channel diarization with mic-channel energy.

    ``diarize_df`` must come from diarizing ``remote`` alone (same timeline as
    the mono mix the transcript was made from).
    """
    df = diarize_df.copy()
    if len(df):
        df["speaker"] = df["speaker"].map(_shift_label)
    result = whisperx.assign_word_speakers(df, result)

    mic_env = _Envelope(mic, sr)
    rem_env = _Envelope(remote, sr)

    for seg in result.get("segments", []):
        words = seg.get("words", []) or []
        last = None
        for w in words:
            if "start" in w and "end" in w:
                if _mic_dominates(mic_env, rem_env, w["start"], w["end"]):
                    w["speaker"] = MIC_LABEL
                last = w.get("speaker", last)
            elif last:
                # tokens without timestamps (numerals etc.) follow their neighbour
                w["speaker"] = last

        # Segment label = speaker holding the most word time in it.
        majority = _majority_speaker(words)
        if majority:
            seg["speaker"] = majority
        elif _mic_dominates(mic_env, rem_env, seg.get("start", 0.0), seg.get("end", 0.0)):
            seg["speaker"] = MIC_LABEL

    result["segments"] = split_segments_at_speaker_changes(result.get("segments", []))
    return result


def _majority_speaker(words):
    totals = {}
    for w in words:
        spk = w.get("speaker")
        if spk and "start" in w and "end" in w:
            totals[spk] = totals.get(spk, 0.0) + max(w["end"] - w["start"], 0.02)
    return max(totals.items(), key=lambda kv: kv[1])[0] if totals else None


def _runs(words):
    """Group a segment's words into consecutive mic / non-mic runs.

    Only the mic-vs-remote boundary is trusted enough to cut on (it comes from
    channel energy, not from clustering).  Within a remote run the label is
    the majority remote speaker, exactly as whisperx labels whole segments, so
    remote-vs-remote attribution is no more fragmented than before.
    Timestamp-less tokens stay with the current run.
    """
    runs = []
    for w in words:
        is_mic = w.get("speaker") == MIC_LABEL
        if runs and (is_mic == runs[-1]["is_mic"] or "start" not in w):
            runs[-1]["words"].append(w)
        else:
            runs.append({"is_mic": is_mic, "words": [w]})
    for run in runs:
        run["speaker"] = MIC_LABEL if run["is_mic"] else _majority_speaker(run["words"])
    return runs


def _run_span(run):
    timed = [w for w in run["words"] if "start" in w and "end" in w]
    if not timed:
        return None, None
    return timed[0]["start"], timed[-1]["end"]


def split_segments_at_speaker_changes(segments):
    """Split a transcript segment wherever the word-level speaker changes.

    Whisper segments on pauses, not on speaker turns, so a quick hand-over
    without a pause lands both voices in one segment; the segment-level label
    (what Speakr displays) can then only be a majority vote.  With word-level
    speakers from the split-channel path we can cut cleanly.  Very short runs
    (a stray word) are absorbed into their neighbour rather than split out.
    """
    out = []
    for seg in segments:
        words = seg.get("words") or []
        runs = _runs(words)
        # absorb runs that are too short to stand on their own
        merged = []
        for run in runs:
            a, b = _run_span(run)
            timed = sum(1 for w in run["words"] if "start" in w)
            short = a is None or (timed < MIN_RUN_WORDS and (b - a) < MIN_RUN_SEC)
            if merged and (short or run["speaker"] == merged[-1]["speaker"]):
                merged[-1]["words"].extend(run["words"])
            elif short and len(runs) > 1 and not merged:
                merged.append({"speaker": None, "words": list(run["words"])})  # settled below
            else:
                merged.append({"speaker": run["speaker"], "words": list(run["words"])})
        # a leading short run takes the speaker of the run that follows it
        if len(merged) > 1 and merged[0]["speaker"] is None:
            merged[1]["words"] = merged[0]["words"] + merged[1]["words"]
            merged.pop(0)
        if len(merged) <= 1:
            out.append(seg)
            continue
        for i, run in enumerate(merged):
            a, b = _run_span(run)
            new = dict(seg)
            new["words"] = run["words"]
            new["start"] = seg.get("start", a) if i == 0 else a
            new["end"] = seg.get("end", b) if i == len(merged) - 1 else b
            new["text"] = " ".join(str(w.get("word", "")).strip() for w in run["words"]).strip()
            if run["speaker"] is not None:
                new["speaker"] = run["speaker"]
            else:
                new.pop("speaker", None)
            out.append(new)
    return out
