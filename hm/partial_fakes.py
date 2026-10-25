"""Partial-Fake Generation and Synthetic Segment Localization.

Specifications:
- 100 spliced clips (real+fake, fake+real, real+fake+real) from the SAME test speaker
- Loudness matching across pieces so the detector cannot key on energy jumps
- Smooth 20ms cross-fades at boundaries to prevent artificial click artifacts
- Ground-truth binary labels at 0.5 s resolution
- Control set of 30 real+real splices (should be mostly unflagged)
- Measures Precision, Recall, and F1 at 0.5 s resolution
"""

import json
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

from ml.config import SAMPLE_RATE
from ml.preprocess import preprocess_audio, trim_edge_silence, normalize_loudness

ROOT = Path(__file__).resolve().parent.parent
PARTIAL_DIR = ROOT / "data" / "processed" / "partial_fakes"


def crossfade_splice(pieces, crossfade_ms=20, sr=SAMPLE_RATE):
    """Concatenate audio pieces with equalized RMS loudness and smooth cross-fades."""
    fade_len = int((crossfade_ms / 1000.0) * sr)
    fade_in = np.linspace(0.0, 1.0, fade_len)
    fade_out = np.linspace(1.0, 0.0, fade_len)

    # 1. Equalize RMS loudness
    target_rms = 0.05
    eq_pieces = []
    for p in pieces:
        rms = np.sqrt(np.mean(p ** 2)) + 1e-8
        scaled = np.clip(p * (target_rms / rms), -1.0, 1.0).astype(np.float32)
        eq_pieces.append(scaled)

    # 2. Cross-fade join
    out = eq_pieces[0]
    for nxt in eq_pieces[1:]:
        if len(out) < fade_len or len(nxt) < fade_len:
            out = np.concatenate([out, nxt])
            continue
        # Overlap fade_out of end with fade_in of next
        overlap = out[-fade_len:] * fade_out + nxt[:fade_len] * fade_in
        out = np.concatenate([out[:-fade_len], overlap, nxt[fade_len:]])

    return np.clip(out, -1.0, 1.0).astype(np.float32)


def generate_partial_dataset(n_fakes=100, n_controls=30, test_speaker="Bernie Sanders", seed=42):
    """Generate 100 spliced partial fakes and 30 real-real controls from the SAME test speaker."""
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    splits_csv = ROOT / "data" / "processed" / "splits.csv"
    data_dir = ROOT / "data" / "raw" / "release_in_the_wild"
    df = pd.read_csv(splits_csv)

    spk_df = df[(df["split"] == "test") & (df["speaker"] == test_speaker)]
    real_files = spk_df[spk_df["label"] == "bona-fide"]["file"].tolist()
    fake_files = spk_df[spk_df["label"] == "spoof"]["file"].tolist()

    print(f"Generating partial fakes from test speaker '{test_speaker}': {len(real_files)} real, {len(fake_files)} fake available.")
    rng = np.random.RandomState(seed)

    records = []

    # 1. Generate 100 partial fakes
    # Patterns: 35 real+fake, 35 fake+real, 30 real+fake+real
    patterns = ["rf"] * 35 + ["fr"] * 35 + ["rfr"] * 30
    rng.shuffle(patterns)

    for i, pat in enumerate(tqdm(patterns, desc="Building partial fakes")):
        if pat == "rf":
            r_file = str(data_dir / rng.choice(real_files))
            f_file = str(data_dir / rng.choice(fake_files))
            y_r, _ = preprocess_audio(r_file, strict=False)
            y_f, _ = preprocess_audio(f_file, strict=False)
            # Take 2-3s of each
            p_r = y_r[: int(rng.uniform(2.0, 3.0) * SAMPLE_RATE)]
            p_f = y_f[: int(rng.uniform(2.0, 3.0) * SAMPLE_RATE)]
            spliced = crossfade_splice([p_r, p_f])
            t_split = len(p_r) / SAMPLE_RATE
            fake_intervals = [(t_split, len(spliced) / SAMPLE_RATE)]
        elif pat == "fr":
            f_file = str(data_dir / rng.choice(fake_files))
            r_file = str(data_dir / rng.choice(real_files))
            y_f, _ = preprocess_audio(f_file, strict=False)
            y_r, _ = preprocess_audio(r_file, strict=False)
            p_f = y_f[: int(rng.uniform(2.0, 3.0) * SAMPLE_RATE)]
            p_r = y_r[: int(rng.uniform(2.0, 3.0) * SAMPLE_RATE)]
            spliced = crossfade_splice([p_f, p_r])
            t_split = len(p_f) / SAMPLE_RATE
            fake_intervals = [(0.0, t_split)]
        else:  # rfr
            r_file1 = str(data_dir / rng.choice(real_files))
            f_file = str(data_dir / rng.choice(fake_files))
            r_file2 = str(data_dir / rng.choice(real_files))
            y_r1, _ = preprocess_audio(r_file1, strict=False)
            y_f, _ = preprocess_audio(f_file, strict=False)
            y_r2, _ = preprocess_audio(r_file2, strict=False)
            p_r1 = y_r1[: int(rng.uniform(1.5, 2.0) * SAMPLE_RATE)]
            p_f = y_f[: int(rng.uniform(2.0, 2.5) * SAMPLE_RATE)]
            p_r2 = y_r2[: int(rng.uniform(1.5, 2.0) * SAMPLE_RATE)]
            spliced = crossfade_splice([p_r1, p_f, p_r2])
            t1 = len(p_r1) / SAMPLE_RATE
            t2 = t1 + (len(p_f) / SAMPLE_RATE)
            fake_intervals = [(t1, t2)]

        out_name = f"partial_fake_{i:03d}_{pat}.wav"
        out_path = PARTIAL_DIR / out_name
        sf.write(str(out_path), spliced, SAMPLE_RATE)

        records.append({
            "path": str(out_path),
            "type": "partial_fake",
            "pattern": pat,
            "duration": len(spliced) / SAMPLE_RATE,
            "fake_intervals": fake_intervals,
        })

    # 2. Generate 30 control real+real splices
    for i in range(n_controls):
        r1 = str(data_dir / rng.choice(real_files))
        r2 = str(data_dir / rng.choice(real_files))
        y_r1, _ = preprocess_audio(r1, strict=False)
        y_r2, _ = preprocess_audio(r2, strict=False)
        p_r1 = y_r1[: int(rng.uniform(2.0, 3.0) * SAMPLE_RATE)]
        p_r2 = y_r2[: int(rng.uniform(2.0, 3.0) * SAMPLE_RATE)]
        spliced = crossfade_splice([p_r1, p_r2])

        out_name = f"control_real_{i:03d}_rr.wav"
        out_path = PARTIAL_DIR / out_name
        sf.write(str(out_path), spliced, SAMPLE_RATE)

        records.append({
            "path": str(out_path),
            "type": "control_real",
            "pattern": "rr",
            "duration": len(spliced) / SAMPLE_RATE,
            "fake_intervals": [],
        })

    manifest_path = PARTIAL_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} partial-fake and control audio clips with metadata to {manifest_path}")
    return records


def evaluate_partial_localization(infer, manifest_path=None, thr=0.50):
    """Evaluate sliding window detector at 0.5 s resolution on partial fakes."""
    if manifest_path is None:
        manifest_path = PARTIAL_DIR / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"\nEvaluating synthetic segment localization on {len(manifest)} clips at 0.5s resolution (Threshold: {thr:.2f})...")
    
    y_true_all = []
    y_pred_all = []
    scores_all = []
    control_flagged_windows = 0
    control_total_windows = 0

    for item in tqdm(manifest, desc="Evaluating timeline localization"):
        path = item["path"]
        fake_intervals = item["fake_intervals"]
        res = infer.predict_timeline(path)
        times = res["times"]        # end times of 2s windows
        scores = res["scores"]      # smoothed scores

        for t_end, s in zip(times, scores):
            t_start = max(0.0, t_end - 2.0)
            
            # Ground truth: does window [t_start, t_end] overlap with any fake interval by >= 0.5s?
            is_fake = False
            for f_start, f_end in fake_intervals:
                overlap = max(0.0, min(t_end, f_end) - max(t_start, f_start))
                if overlap >= 0.5:
                    is_fake = True
                    break

            label = 1 if is_fake else 0
            pred = 1 if s >= thr else 0

            y_true_all.append(label)
            y_pred_all.append(pred)
            scores_all.append(s)

            if item["type"] == "control_real":
                control_total_windows += 1
                if pred == 1:
                    control_flagged_windows += 1

    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    control_fp_rate = float(control_flagged_windows / max(control_total_windows, 1))

    report = {
        "threshold": thr,
        "total_windows_evaluated": len(y_true),
        "synthetic_windows": int(np.sum(y_true == 1)),
        "genuine_windows": int(np.sum(y_true == 0)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "control_clean_windows": control_total_windows,
        "control_flagged_windows": control_flagged_windows,
        "control_false_alarm_rate": round(control_fp_rate, 4),
    }

    print("\n" + "=" * 65)
    print("SYNTHETIC SEGMENT LOCALIZATION RESULTS (0.5s Resolution)")
    print("=" * 65)
    print(f"Total 0.5s Evaluation Windows: {len(y_true)}")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1 Score  : {f1:.4f} ({f1 * 100:.2f}%)")
    print(f"Real+Real Control False Alarm Rate: {control_fp_rate * 100:.2f}% ({control_flagged_windows}/{control_total_windows})")
    
    rep_path = ROOT / "data" / "processed" / "partial_fakes_report.json"
    with open(rep_path, "w") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    generate_partial_dataset()
    from hm.infer import VoiceShieldInference
    infer = VoiceShieldInference()
    evaluate_partial_localization(infer)
