"""Diagnostic script for VoiceShield baseline.
Diagnoses why real microphone recordings and other inputs are flagged as fake.
"""
import os
import sys
import tempfile
from pathlib import Path
import soundfile as sf
import numpy as np
import torch
import librosa

# Make sure repo root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.inference import VoiceShieldInference
from ml.extract_embeddings import embed_file, embed_waveform
from ml.preprocess import preprocess_audio, standardize, fix_length
from ml.augment import degrade

def main():
    print("Loading VoiceShieldInference...")
    infer = VoiceShieldInference(device="cpu")
    mean = infer.mean.numpy()
    std = infer.std.numpy()

    test_dir = ROOT / "data" / "diag_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare synthetic signals
    sr = 16000
    dur = 4.0
    t = np.arange(int(sr * dur)) / sr

    # Tone 440 Hz
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    tone_path = test_dir / "tone_440hz.wav"
    sf.write(str(tone_path), tone, sr)

    # White Noise
    rng = np.random.RandomState(42)
    noise = (0.05 * rng.randn(len(t))).astype(np.float32)
    noise_path = test_dir / "white_noise.wav"
    sf.write(str(noise_path), noise, sr)

    # Pink Noise
    pink = np.cumsum(rng.randn(len(t))).astype(np.float32)
    pink = pink - pink.mean()
    pink = (0.05 * pink / (pink.std() + 1e-6)).astype(np.float32)
    pink_path = test_dir / "pink_noise.wav"
    sf.write(str(pink_path), pink, sr)

    # 2. Select dataset clips
    data_dir = ROOT / "data" / "raw" / "release_in_the_wild"
    splits_csv = ROOT / "data" / "processed" / "splits.csv"
    import pandas as pd
    df = pd.read_csv(splits_csv)

    real_train_files = df[(df["split"] == "train") & (df["label"] == "bona-fide")]["file"].iloc[:3].tolist()
    real_test_files = df[(df["split"] == "test") & (df["label"] == "bona-fide")]["file"].iloc[:3].tolist()
    fake_test_files = df[(df["split"] == "test") & (df["label"] == "spoof")]["file"].iloc[:3].tolist()

    clips = []
    for f in real_train_files:
        clips.append(("Real Train: " + f, data_dir / f))
    for f in real_test_files:
        clips.append(("Real Test: " + f, data_dir / f))
    for f in fake_test_files:
        clips.append(("Fake Test: " + f, data_dir / f))

    clips.append(("Synth: Pure Tone 440Hz", tone_path))
    clips.append(("Synth: White Noise", noise_path))
    clips.append(("Synth: Pink Noise", pink_path))

    # 3. User voice recordings from Downloads
    import glob
    wa_files = sorted(glob.glob("C:/Users/omen/Downloads/WhatsApp Audio*"))
    user_clips = []
    # Clip 1: 3.4s clip
    if len(wa_files) > 0:
        p1 = wa_files[0]
        y, s = librosa.load(p1, sr=sr)
        u1 = test_dir / "user_voice_1.wav"
        sf.write(str(u1), y, sr)
        user_clips.append(("User Voice 1 (WhatsApp)", u1))

    # Clip 2: 5.7s clip
    for f in wa_files:
        if "22.14.26" in f and f.endswith(".mp3"):
            y, s = librosa.load(f, sr=sr)
            u2 = test_dir / "user_voice_2.wav"
            sf.write(str(u2), y, sr)
            user_clips.append(("User Voice 2 (WhatsApp)", u2))
            break

    # Clip 3: 10.6s clip
    for f in wa_files:
        if "22.17.36" in f and f.endswith(".mp3"):
            y, s = librosa.load(f, sr=sr)
            u3 = test_dir / "user_voice_3.wav"
            sf.write(str(u3), y, sr)
            user_clips.append(("User Voice 3 (WhatsApp)", u3))
            break

    # Clip 4 & 5: slices from 96s clip
    for f in wa_files:
        if "22.14.25" in f:
            y, s = librosa.load(f, sr=sr)
            u4 = test_dir / "user_voice_4.wav"
            u5 = test_dir / "user_voice_5.wav"
            sf.write(str(u4), y[int(5*sr):int(9*sr)], sr)
            sf.write(str(u5), y[int(15*sr):int(19*sr)], sr)
            user_clips.append(("User Voice 4 (WhatsApp 4s)", u4))
            user_clips.append(("User Voice 5 (WhatsApp 4s)", u5))
            break

    # Also record 1 live mic clip using sounddevice if possible
    try:
        import sounddevice as sd
        print("Recording 3 seconds from laptop microphone for live test...")
        rec = sd.rec(int(3.0 * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        u_live = test_dir / "user_live_mic.wav"
        sf.write(str(u_live), rec[:, 0], sr)
        user_clips.append(("User Live Laptop Mic (3s)", u_live))
    except Exception as e:
        print("Could not record live mic:", e)

    clips.extend(user_clips)

    print("\n" + "="*85)
    print(f"{'Sample Name':<32} | {'Fake Prob':<10} | {'Verdict':<8} | {'Conf':<6} | {'Mean |z|':<9} | {'Max |z|':<8}")
    print("="*85)

    results = []
    for name, path in clips:
        try:
            emb = embed_file(infer.encoder, str(path), device="cpu", strict=True)
            z = (emb - mean) / std
            mean_abs_z = float(np.mean(np.abs(z)))
            max_abs_z = float(np.max(np.abs(z)))
            res = infer.predict(str(path))
            p = res["fake_probability"]
            v = res["prediction"]
            c = res["confidence"]
            print(f"{name:<32} | {p:<10.4f} | {v:<8} | {c:<6.2f} | {mean_abs_z:<9.2f} | {max_abs_z:<8.2f}")
            results.append((name, path, p, v, c, mean_abs_z, max_abs_z))
        except Exception as e:
            print(f"{name:<32} | ERROR: {e}")

    # 4. Degradation experiment on a known real clip
    print("\n" + "="*85)
    print("DEGRADATION ANALYSIS ON A REAL CLIP (Which degradation flips it?)")
    print("="*85)
    # Pick a real sample that is >= 2s
    real_sample_path = str(data_dir / "44.wav")
    a, _ = librosa.load(real_sample_path, sr=sr, mono=True)
    
    degradations = ["clean", "8k", "mulaw", "amr", "noise"]
    rng_deg = np.random.RandomState(42)
    for deg in degradations:
        if deg == "clean":
            deg_a = a
        else:
            deg_a = degrade(a, deg, rng_deg, fixed=True)
        deg_path = test_dir / f"real_test_deg_{deg}.wav"
        sf.write(str(deg_path), deg_a, sr)
        emb = embed_file(infer.encoder, str(deg_path), device="cpu", strict=True)
        z = (emb - mean) / std
        mean_abs_z = float(np.mean(np.abs(z)))
        res = infer.predict(str(deg_path))
        print(f"Degradation: {deg:<8} -> Fake Prob: {res['fake_probability']:.4f} | Verdict: {res['prediction']:<8} | Mean |z|: {mean_abs_z:.2f}")

    # 5. Test Backend Integration Comparison
    print("\n" + "="*85)
    print("BACKEND COMPARISON (Direct VoiceShieldInference vs FastAPI POST /analyze)")
    print("="*85)
    from fastapi.testclient import TestClient
    from backend.main import app
    client = TestClient(app)

    test_samples = [
        ("Real Test 44.wav", data_dir / "44.wav"),
        ("Fake Test 10.wav", data_dir / "10.wav"),
        ("User Live Mic", test_dir / "user_live_mic.wav"),
        ("User Voice 2", test_dir / "user_voice_2.wav"),
        ("User Voice 3", test_dir / "user_voice_3.wav"),
    ]
    for label, pth in test_samples:
        direct = infer.predict(str(pth))
        with open(str(pth), "rb") as f:
            resp = client.post("/analyze", files={"file": (pth.name, f.read(), "audio/wav")})
        if resp.status_code == 200:
            b_json = resp.json()
            b_pred = b_json.get("prediction")
            b_conf = b_json.get("confidence")
            b_fake_p = b_json.get("details", {}).get("fake_probability")
            match = (direct["prediction"] == b_pred and abs(direct["confidence"] - b_conf) < 1e-3)
            print(f"{label:<25} | Direct: {direct['prediction']:<6} ({direct['fake_probability']:.4f}) | Backend: {b_pred:<6} ({b_fake_p:.4f}) | Exact Match: {match}")
        else:
            print(f"{label:<25} | Backend Error: {resp.status_code} {resp.text}")

if __name__ == "__main__":
    main()
