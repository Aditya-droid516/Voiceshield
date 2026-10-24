"""Extract hand-made features for VoiceShield evaluation splits:
- Clean dev (1,200 clips)
- Clean test (1,200 clips)
- Clean train (2,400 clips)
- Degraded test sets (8k, mulaw, amr, noise: 400 clips each from test set)
- Augmented training set (clean + 8k + mulaw + amr + noise + mic simulation)
- Real mic audio clips (from local recordings) and speaker-disjoint mic test set
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hm.features import extract_features, FEATURE_NAMES, to_vector
from hm.stability import d_mic_simulation
from ml.augment import d_8k, d_codec, d_noise
from ml.preprocess import preprocess_audio, AudioError

PROCESSED_DIR = ROOT / "data" / "processed"


def _worker_extract(path_and_label):
    path, label = path_and_label
    try:
        y, sr = preprocess_audio(path, strict=False)
        fd = extract_features(y, sr)
        vec = to_vector(fd)
        return vec, int(label), str(path)
    except Exception:
        return None


def extract_split(pairs, out_path, max_workers=6, desc="Extracting"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        d = np.load(out_path, allow_pickle=True)
        print(f"[{out_path.name}] Already exists with {len(d['y'])} samples.")
        return d["X"], d["y"], list(d["paths"])

    print(f"\n{desc}: {len(pairs)} items using {max_workers} workers...")
    t0 = time.time()
    X, y, paths = [], [], []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in tqdm(executor.map(_worker_extract, pairs), total=len(pairs), desc=desc):
            if res is not None:
                vec, lab, p = res
                X.append(vec)
                y.append(lab)
                paths.append(p)

    X = np.stack(X) if X else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    paths = np.array(paths)

    np.savez_compressed(out_path, X=X, y=y, paths=paths, feature_names=np.array(FEATURE_NAMES))
    dt = time.time() - t0
    print(f"Saved {len(y)} clips to {out_path} ({dt:.1f}s, {dt/max(len(pairs),1)*1000:.1f}ms/clip)")
    return X, y, paths


def _worker_degrade_and_extract(item):
    path, label, kind, seed = item
    try:
        y, sr = preprocess_audio(path, strict=False)
        rng = np.random.RandomState(seed)
        if kind == "8k":
            y_deg = d_8k(y, rng, fixed=True)
        elif kind == "mulaw":
            y_deg = d_codec(y, rng, fixed=True, codec="mulaw")
        elif kind == "amr":
            y_deg = d_codec(y, rng, fixed=True, codec="amr")
        elif kind == "noise":
            y_deg = d_noise(y, rng, fixed=True)
        elif kind == "mic":
            y_deg = d_mic_simulation(y, sr=sr, seed=seed)
        else:
            y_deg = y
        fd = extract_features(y_deg, sr)
        return to_vector(fd), int(label), f"{path}#{kind}"
    except Exception:
        return None


def extract_degraded_split(pairs, kind, out_path, max_workers=6, limit=None):
    out_path = Path(out_path)
    if out_path.exists():
        d = np.load(out_path, allow_pickle=True)
        print(f"[{out_path.name}] Already exists with {len(d['y'])} samples.")
        return d["X"], d["y"], list(d["paths"])

    selected = list(pairs)[:limit] if limit else list(pairs)
    items = [(p, l, kind, idx) for idx, (p, l) in enumerate(selected)]

    print(f"\nExtracting Degraded [{kind}]: {len(items)} items...")
    t0 = time.time()
    X, y, paths = [], [], []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in tqdm(executor.map(_worker_degrade_and_extract, items), total=len(items), desc=f"Degraded {kind}"):
            if res is not None:
                vec, lab, p = res
                X.append(vec)
                y.append(lab)
                paths.append(p)

    X = np.stack(X) if X else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    paths = np.array(paths)

    np.savez_compressed(out_path, X=X, y=y, paths=paths, feature_names=np.array(FEATURE_NAMES))
    dt = time.time() - t0
    print(f"Saved {len(y)} clips to {out_path} ({dt:.1f}s)")
    return X, y, paths


def main():
    # 1. Load exact paths from baseline npz files
    test_npz = np.load(PROCESSED_DIR / "test.npz", allow_pickle=True)
    dev_npz = np.load(PROCESSED_DIR / "dev.npz", allow_pickle=True)
    train_npz = np.load(PROCESSED_DIR / "train.npz", allow_pickle=True)

    test_pairs = list(zip(test_npz["paths"], test_npz["y"]))
    dev_pairs = list(zip(dev_npz["paths"], dev_npz["y"]))
    # Take balanced 2400 from train_npz
    train_paths = train_npz["paths"]
    train_y = train_npz["y"]
    real_idx = np.where(train_y == 0)[0][:1200]
    fake_idx = np.where(train_y == 1)[0][:1200]
    train_sub_idx = np.concatenate([real_idx, fake_idx])
    np.random.RandomState(42).shuffle(train_sub_idx)
    train_pairs = [(train_paths[i], train_y[i]) for i in train_sub_idx]

    print("Target splits:")
    print(f"  Test : {len(test_pairs)} clips")
    print(f"  Dev  : {len(dev_pairs)} clips")
    print(f"  Train: {len(train_pairs)} clips")

    # Extract clean sets
    extract_split(dev_pairs, PROCESSED_DIR / "hm_dev.npz", max_workers=6, desc="HM Dev (Clean)")
    extract_split(test_pairs, PROCESSED_DIR / "hm_test.npz", max_workers=6, desc="HM Test (Clean)")
    extract_split(train_pairs, PROCESSED_DIR / "hm_train_clean.npz", max_workers=6, desc="HM Train (Clean)")

    # Extract degraded test sets (400 clips each from test set: 200 real, 200 fake)
    test_real_pairs = [p for p in test_pairs if p[1] == 0][:200]
    test_fake_pairs = [p for p in test_pairs if p[1] == 1][:200]
    test_deg_pairs = test_real_pairs + test_fake_pairs

    for deg in ["8k", "mulaw", "amr", "noise"]:
        extract_degraded_split(test_deg_pairs, deg, PROCESSED_DIR / f"hm_test_{deg}.npz", max_workers=6)

    # Build Augmented Training Set:
    # 2400 clean + 400 8k + 400 mulaw + 400 amr + 400 noise + 400 mic simulation
    # Split evenly between real and fake
    aug_train_path = PROCESSED_DIR / "hm_train_aug.npz"
    if not aug_train_path.exists():
        print("\nBuilding Augmented Training Set (Clean + Degraded Copies of BOTH real and fake)...")
        # Load clean train
        d_clean = np.load(PROCESSED_DIR / "hm_train_clean.npz", allow_pickle=True)
        X_aug = [d_clean["X"]]
        y_aug = [d_clean["y"]]
        paths_aug = [d_clean["paths"]]

        # Take 200 real and 200 fake train clips for each degradation
        tr_real = [p for p in train_pairs if p[1] == 0][:200]
        tr_fake = [p for p in train_pairs if p[1] == 1][:200]
        deg_tr_pairs = tr_real + tr_fake

        for deg in ["8k", "mulaw", "amr", "noise", "mic"]:
            X_d, y_d, p_d = extract_degraded_split(deg_tr_pairs, deg, PROCESSED_DIR / f"hm_train_{deg}_sub.npz", max_workers=6)
            X_aug.append(X_d)
            y_aug.append(y_d)
            paths_aug.append(p_d)

        # Add user mic recordings to train and test
        # Real mic audio from Downloads
        import glob
        wa_files = sorted(glob.glob("C:/Users/omen/Downloads/WhatsApp Audio*"))
        mic_train_items = []
        mic_test_items = []
        for f in wa_files:
            if "22.14.21" in f or "22.14.26" in f:
                mic_train_items.append((f, 0))  # real
            elif "22.17.36" in f:
                mic_test_items.append((f, 0))   # real held-out test
        
        # Add live laptop mic recording to test
        live_mic_f = ROOT / "data" / "diag_test" / "user_live_mic.wav"
        if live_mic_f.exists():
            mic_test_items.append((str(live_mic_f), 0))

        # Add mic train items to augmented train
        for mp, ml in mic_train_items:
            try:
                y_m, sr_m = preprocess_audio(mp, strict=False)
                vec = to_vector(extract_features(y_m, sr_m))
                X_aug.append(vec[None])
                y_aug.append(np.array([ml]))
                paths_aug.append(np.array([mp]))
            except Exception:
                pass

        X_final = np.concatenate(X_aug, axis=0)
        y_final = np.concatenate(y_aug, axis=0)
        paths_final = np.concatenate(paths_aug, axis=0)

        # Shuffle
        perm = np.random.RandomState(42).permutation(len(y_final))
        X_final = X_final[perm]
        y_final = y_final[perm]
        paths_final = paths_final[perm]

        np.savez_compressed(
            aug_train_path,
            X=X_final,
            y=y_final,
            paths=paths_final,
            feature_names=np.array(FEATURE_NAMES),
        )
        print(f"Saved complete augmented training set: {X_final.shape} (fake {100*y_final.mean():.1f}%) to {aug_train_path}")

        # Also save held-out real mic test set
        X_mtest, y_mtest, p_mtest = [], [], []
        for mp, ml in mic_test_items:
            try:
                y_m, sr_m = preprocess_audio(mp, strict=False)
                vec = to_vector(extract_features(y_m, sr_m))
                X_mtest.append(vec)
                y_mtest.append(ml)
                p_mtest.append(mp)
            except Exception:
                pass
        if X_mtest:
            np.savez_compressed(
                PROCESSED_DIR / "hm_test_real_mic.npz",
                X=np.array(X_mtest),
                y=np.array(y_mtest),
                paths=np.array(p_mtest),
                feature_names=np.array(FEATURE_NAMES),
            )
            print(f"Saved held-out real mic test set: {len(y_mtest)} samples to hm_test_real_mic.npz")


if __name__ == "__main__":
    main()
