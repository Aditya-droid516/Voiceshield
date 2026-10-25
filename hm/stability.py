"""Feature Stability Analysis under Phone-Call and Acoustic Degradations.

Evaluates how each hand-crafted feature shifts under:
- 8 kHz resampling
- G.711 mu-law compression
- AMR-NB compression
- Background noise (10 dB SNR)
- Microphone & room acoustic simulation (reverb + mic EQ + ambient noise)

Identifies which acoustic cues are stable across phone channels vs fragile,
enabling robust classifier training and transparent forensic reporting.
"""

import json
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from scipy import signal
from tqdm import tqdm

from hm.features import extract_features, FEATURE_NAMES, FEATURE_GROUPS, to_vector
from ml.augment import d_8k, d_codec, d_noise
from ml.preprocess import preprocess_audio

ROOT = Path(__file__).resolve().parent.parent


def d_mic_simulation(y, sr=16000, seed=42):
    """Simulate laptop / phone microphone in a typical room:

    - Room impulse response (early reflections + exponential decay, T60 ~ 0.15s)
    - Microphone frequency response (bandpass 150 Hz - 7000 Hz, slight low-mid resonance)
    - Mild ambient room noise (25 dB SNR)
    """
    rng = np.random.RandomState(seed)
    
    # 1. Room impulse response (reverb)
    rir_len = int(0.12 * sr)
    t = np.arange(rir_len) / sr
    decay = np.exp(-t / 0.03)  # decay constant
    rir = rng.randn(rir_len) * decay
    rir[0] = 1.0  # direct path
    rir = rir / np.sqrt(np.sum(rir ** 2) + 1e-9)
    y_rev = signal.fftconvolve(y, rir, mode="same")
    
    # 2. Mic EQ: 2nd order Butterworth bandpass 150 Hz - 7000 Hz
    b, a = signal.butter(2, [150 / (sr / 2), 7000 / (sr / 2)], btype="bandpass")
    y_eq = signal.lfilter(b, a, y_rev)
    
    # 3. Ambient room noise (25 dB SNR)
    snr_db = 25.0
    p_sig = np.mean(y_eq ** 2) + 1e-10
    n = rng.randn(len(y_eq))
    # low-pass filtered room air noise
    n_filt = np.convolve(n, np.ones(16) / 16, mode="same")
    p_n = np.mean(n_filt ** 2) + 1e-12
    n_filt = n_filt * np.sqrt(p_sig / (10 ** (snr_db / 10)) / p_n)
    y_mic = y_eq + n_filt
    
    return np.clip(y_mic, -1.0, 1.0).astype(np.float32)


def evaluate_stability(audio_paths, n_samples=40, out_json=None):
    """Compute normalized drift for all features across 5 degradation conditions."""
    sample_paths = list(audio_paths)[:n_samples]
    n_feats = len(FEATURE_NAMES)
    rng = np.random.RandomState(0)

    clean_vectors = []
    deg_vectors = {
        "8k": [],
        "mulaw": [],
        "amr": [],
        "noise": [],
        "mic": [],
    }

    print(f"Running stability evaluation on {len(sample_paths)} clips across 5 degradation conditions...")
    for p in tqdm(sample_paths, desc="Stability evaluation"):
        try:
            y, sr = preprocess_audio(p, strict=False)
            if len(y) < 16000:
                continue
            
            # Clean
            f_clean = to_vector(extract_features(y, sr))
            
            # Degraded
            y_8k = d_8k(y, rng, fixed=True)
            y_mulaw = d_codec(y, rng, fixed=True, codec="mulaw")
            y_amr = d_codec(y, rng, fixed=True, codec="amr")
            y_noise = d_noise(y, rng, fixed=True)
            y_mic = d_mic_simulation(y, sr=sr, seed=rng.randint(100000))
            
            f_8k = to_vector(extract_features(y_8k, sr))
            f_mulaw = to_vector(extract_features(y_mulaw, sr))
            f_amr = to_vector(extract_features(y_amr, sr))
            f_noise = to_vector(extract_features(y_noise, sr))
            f_mic = to_vector(extract_features(y_mic, sr))

            clean_vectors.append(f_clean)
            deg_vectors["8k"].append(f_8k)
            deg_vectors["mulaw"].append(f_mulaw)
            deg_vectors["amr"].append(f_amr)
            deg_vectors["noise"].append(f_noise)
            deg_vectors["mic"].append(f_mic)
        except Exception as e:
            continue

    clean_mat = np.array(clean_vectors)  # [N, n_feats]
    stds = np.std(clean_mat, axis=0) + 1e-6

    feature_drifts = {}
    condition_drifts = {c: [] for c in deg_vectors}

    for j, fname in enumerate(FEATURE_NAMES):
        shifts = []
        for cond, mat_list in deg_vectors.items():
            deg_mat = np.array(mat_list)
            # Relative mean absolute difference normalized by feature std
            drift = float(np.mean(np.abs(deg_mat[:, j] - clean_mat[:, j])) / stds[j])
            condition_drifts[cond].append(drift)
            shifts.append(drift)
        feature_drifts[fname] = {
            "mean_drift": float(np.mean(shifts)),
            "max_drift": float(np.max(shifts)),
            "drift_by_cond": {cond: float(condition_drifts[cond][-1]) for cond in deg_vectors},
        }

    # Group summary
    group_summary = {}
    for grp_name, f_names in FEATURE_GROUPS.items():
        grp_drifts = [feature_drifts[fn]["mean_drift"] for fn in f_names if fn in feature_drifts]
        group_summary[grp_name] = {
            "mean_drift": float(np.mean(grp_drifts)) if grp_drifts else 0.0,
            "n_features": len(f_names),
        }

    # Stability cutoff: features with mean drift < 1.5 standard deviations
    # Highly unstable features (> 1.5 std drift across phone channels) are flagged
    drift_threshold = 1.5
    stable_features = [fn for fn, d in feature_drifts.items() if d["mean_drift"] < drift_threshold]
    unstable_features = [fn for fn, d in feature_drifts.items() if d["mean_drift"] >= drift_threshold]

    report = {
        "n_samples": len(clean_vectors),
        "total_features": n_feats,
        "n_stable": len(stable_features),
        "n_unstable": len(unstable_features),
        "drift_threshold": drift_threshold,
        "group_summary": group_summary,
        "stable_features": stable_features,
        "unstable_features": unstable_features,
        "feature_drifts": feature_drifts,
    }

    if out_json:
        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved stability report to {out_json}")

    return report


if __name__ == "__main__":
    import pandas as pd
    splits_csv = ROOT / "data" / "processed" / "splits.csv"
    data_dir = ROOT / "data" / "raw" / "release_in_the_wild"
    df = pd.read_csv(splits_csv)
    # Pick 25 real and 25 fake dev clips
    dev_real = df[(df["split"] == "dev") & (df["label"] == "bona-fide")]["file"].iloc[:25].tolist()
    dev_fake = df[(df["split"] == "dev") & (df["label"] == "spoof")]["file"].iloc[:25].tolist()
    paths = [str(data_dir / f) for f in dev_real + dev_fake]

    rep = evaluate_stability(paths, n_samples=40, out_json=ROOT / "hm" / "stability_report.json")
    print("\n" + "=" * 60)
    print("FEATURE STABILITY ACROSS PHONE-CALL DEGRADATIONS")
    print("=" * 60)
    print(f"Total features evaluated: {rep['total_features']}")
    print(f"Stable features: {rep['n_stable']} | Unstable (>1.5 std shift): {rep['n_unstable']}")
    print("\nGroup Mean Relative Drift (Lower = More Robust to Phone Codecs & Mics):")
    for grp, s in sorted(rep["group_summary"].items(), key=lambda x: x[1]["mean_drift"]):
        print(f"  {grp:<20}: drift {s['mean_drift']:.3f} ({s['n_features']} features)")
    
    print("\nTop 10 Most Unstable Features (Do NOT survive phone codecs):")
    sorted_unstable = sorted(rep["feature_drifts"].items(), key=lambda x: x[1]["mean_drift"], reverse=True)[:10]
    for fn, d in sorted_unstable:
        print(f"  {fn:<32}: drift {d['mean_drift']:.3f} (max {d['max_drift']:.3f} on {max(d['drift_by_cond'], key=d['drift_by_cond'].get)})")
