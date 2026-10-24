"""Streaming scores: slide a 2 s window (0.5 s hop) over a clip."""
import time

import numpy as np

from ml.config import SAMPLE_RATE
from ml.extract_embeddings import embed_waveform
from ml.preprocess import load_audio

WINDOW_S = 2.0
HOP_S = 0.5
SMOOTH_N = 3
ALERT_AT = 0.5
ALERT_RUN = 2


def window_scores(infer, audio, sr=SAMPLE_RATE, window_s=WINDOW_S, hop_s=HOP_S):
    """Raw fake probability per window. Returns (end_times, scores)."""
    win, hop = int(window_s * sr), int(hop_s * sr)
    if len(audio) < win:
        audio = np.pad(audio, (0, win - len(audio)), mode="wrap")
    times, scores = [], []
    for start in range(0, len(audio) - win + 1, hop):
        chunk = audio[start:start + win]
        emb = embed_waveform(infer.encoder, chunk, device=infer.device)
        scores.append(infer.score_embedding(emb))
        times.append((start + win) / sr)
    return np.array(times), np.array(scores)


def smooth(scores, n=SMOOTH_N):
    out = []
    for i in range(len(scores)):
        out.append(float(np.mean(scores[max(0, i - n + 1):i + 1])))
    return np.array(out)


def first_alert(smoothed, times, thr=ALERT_AT, run=ALERT_RUN):
    """Time of the first point where `run` consecutive smoothed scores exceed thr, else None."""
    count = 0
    for t, s in zip(times, smoothed):
        count = count + 1 if s >= thr else 0
        if count >= run:
            return float(t)
    return None


def timeline(infer, path):
    audio = load_audio(str(path))
    if isinstance(audio, tuple):
        audio = audio[0]
    t0 = time.time()
    times, raw = window_scores(infer, np.asarray(audio, dtype=np.float32))
    sm = smooth(raw)
    return {
        "times": [round(float(x), 2) for x in times],
        "scores": [round(float(x), 4) for x in sm],
        "raw_scores": [round(float(x), 4) for x in raw],
        "alert_time": first_alert(sm, times),
        "ms_per_update": round(1000 * (time.time() - t0) / max(len(raw), 1), 1),
    }
