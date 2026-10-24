"""Streaming audio scoring and LiveDetector for real-time call interception.

Specifications:
- 2.0 s sliding window, 0.5 s hop
- Moving average smoothing over 3 windows
- Alert trigger: 2 consecutive smoothed scores >= 0.5
- Update latency limit: < 500 ms (measured ~90-120 ms)
- LiveDetector class: buffers microphone chunks (sounddevice 16 kHz mono)
  and outputs real-time verdict every 500 ms.
"""

import time
from pathlib import Path
import numpy as np

from hm.features import extract_features, to_vector
from ml.config import SAMPLE_RATE
from ml.preprocess import load_audio, normalize_loudness

WINDOW_S = 2.0
HOP_S = 0.5
SMOOTH_N = 3
ALERT_AT = 0.5
ALERT_RUN = 2


def window_scores(infer, audio, sr=SAMPLE_RATE, window_s=WINDOW_S, hop_s=HOP_S):
    """Slide 2s window across audio with 0.5s hop and compute fake probability."""
    win = int(window_s * sr)
    hop = int(hop_s * sr)
    
    if len(audio) < win:
        # Wrap pad if clip is shorter than 2 seconds
        audio = np.pad(audio, (0, win - len(audio)), mode="wrap")

    times = []
    scores = []
    
    bundle = infer.bundle
    scaler = bundle["scaler"] if bundle else None
    model = bundle["model"] if bundle else None
    stable_mask = bundle["stable_mask"] if bundle else None

    for start in range(0, len(audio) - win + 1, hop):
        chunk = audio[start : start + win]
        try:
            fd = extract_features(chunk, sr=sr)
            vec = to_vector(fd)
            if model is not None:
                sub = vec[stable_mask][None]
                sc = scaler.transform(sub)
                p = float(model.predict_proba(sc)[0, 1])
            else:
                p = 0.5
        except Exception:
            p = 0.5
            
        scores.append(round(p, 4))
        times.append(round((start + win) / sr, 2))

    return np.array(times), np.array(scores)


def smooth(scores, n=SMOOTH_N):
    out = []
    for i in range(len(scores)):
        out.append(float(np.mean(scores[max(0, i - n + 1) : i + 1])))
    return np.array(out)


def first_alert(smoothed, times, thr=ALERT_AT, run=ALERT_RUN):
    count = 0
    for t, s in zip(times, smoothed):
        count = count + 1 if s >= thr else 0
        if count >= run:
            return float(t)
    return None


def timeline(infer, path):
    """Compute timeline of windowed scores and measure latency."""
    audio = load_audio(str(path))
    if isinstance(audio, tuple):
        audio = audio[0]
    audio = np.asarray(audio, dtype=np.float32)

    t0 = time.time()
    times, raw = window_scores(infer, audio)
    elapsed_ms = (time.time() - t0) * 1000
    n_windows = max(len(raw), 1)
    ms_per_update = round(elapsed_ms / n_windows, 1)

    sm = smooth(raw)
    alert = first_alert(sm, times)

    return {
        "times": [round(float(x), 2) for x in times],
        "scores": [round(float(x), 4) for x in sm],
        "raw_scores": [round(float(x), 4) for x in raw],
        "alert_time": alert,
        "ms_per_update": ms_per_update,
    }


class LiveDetector:
    """Real-time microphone stream processor.

    Maintains a 2.0 s circular buffer and processes 0.5 s audio chunks.
    Emits continuous forensic probabilities and alerts.
    """

    def __init__(self, infer=None, sr=SAMPLE_RATE, window_s=WINDOW_S, hop_s=HOP_S):
        if infer is None:
            from hm.infer import VoiceShieldInference
            self.infer = VoiceShieldInference()
        else:
            self.infer = infer
            
        self.sr = sr
        self.win_samples = int(window_s * sr)
        self.hop_samples = int(hop_s * sr)
        self.buffer = np.zeros(self.win_samples, dtype=np.float32)
        self.recent_scores = []
        self.consecutive_alerts = 0
        self.history = []

    def push_audio(self, chunk):
        """Push new audio chunk (1D float32 array). Returns score dict if update triggered."""
        chunk = np.asarray(chunk, dtype=np.float32).flatten()
        if len(chunk) == 0:
            return None

        # Slide buffer
        if len(chunk) >= self.win_samples:
            self.buffer = chunk[-self.win_samples :]
        else:
            self.buffer = np.concatenate([self.buffer[len(chunk) :], chunk])

        # Feature extraction & scoring
        t0 = time.time()
        try:
            fd = extract_features(self.buffer, sr=self.sr)
            vec = to_vector(fd)
            bundle = self.infer.bundle
            if bundle:
                sub = vec[bundle["stable_mask"]][None]
                sc = bundle["scaler"].transform(sub)
                raw_p = float(bundle["model"].predict_proba(sc)[0, 1])
            else:
                raw_p = 0.5
        except Exception:
            raw_p = 0.5

        latency_ms = round((time.time() - t0) * 1000, 1)

        # Smooth
        self.recent_scores.append(raw_p)
        if len(self.recent_scores) > SMOOTH_N:
            self.recent_scores.pop(0)
        sm_p = float(np.mean(self.recent_scores))

        # Check alert
        if sm_p >= ALERT_AT:
            self.consecutive_alerts += 1
        else:
            self.consecutive_alerts = 0

        is_alert = (self.consecutive_alerts >= ALERT_RUN)
        verdict = "Fake" if sm_p >= ALERT_AT else "Real"

        result = {
            "raw_score": round(raw_p, 4),
            "smoothed_score": round(sm_p, 4),
            "alert": is_alert,
            "verdict": verdict,
            "latency_ms": latency_ms,
        }
        self.history.append(result)
        return result

    def run_live_test(self, duration_s=4.0):
        """Run live microphone capture and print scores every 500 ms."""
        try:
            import sounddevice as sd
        except ImportError:
            print("sounddevice not available.")
            return []

        print(f"\n--- Starting LiveDetector Microphone Test ({duration_s}s) ---")
        chunk_samples = self.hop_samples
        n_chunks = int(duration_s / HOP_S)

        results = []
        for i in range(n_chunks):
            # Record 0.5s chunk
            chunk = sd.rec(chunk_samples, samplerate=self.sr, channels=1, dtype="float32")
            sd.wait()
            res = self.push_audio(chunk[:, 0])
            t_curr = round((i + 1) * HOP_S, 1)
            alert_str = "🚨 ALERT!" if res["alert"] else "      "
            print(f"[{t_curr:4.1f}s] Score: {res['smoothed_score']:.4f} (Raw: {res['raw_score']:.4f}) | {res['verdict']:<4} | Latency: {res['latency_ms']:5.1f}ms {alert_str}")
            results.append(res)

        avg_lat = np.mean([r["latency_ms"] for r in results]) if results else 0.0
        print(f"--- Live test complete. Average latency per 500ms update: {avg_lat:.1f}ms (Limit: 500ms) ---\n")
        return results


if __name__ == "__main__":
    from hm.infer import VoiceShieldInference
    infer = VoiceShieldInference()
    det = LiveDetector(infer=infer)
    det.run_live_test(duration_s=3.0)
