"""Phone-call degradations. Writes degraded 16 kHz wavs, then extracts embeddings."""
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from ml.config import SAMPLE_RATE

ROOT = Path(__file__).resolve().parent.parent
KINDS = ["8k", "mulaw", "amr", "noise"]
AMR_RATES = ["5900", "7400", "10200", "12200"]


def _ff(args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + args, check=True)


def d_8k(a, rng, fixed):
    lo = librosa.resample(a, orig_sr=SAMPLE_RATE, target_sr=8000)
    return librosa.resample(lo, orig_sr=8000, target_sr=SAMPLE_RATE)


def d_codec(a, rng, fixed, codec):
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "s.wav")
        dst = os.path.join(td, "d.wav")
        sf.write(src, a, SAMPLE_RATE)
        if codec == "mulaw":
            mid = os.path.join(td, "m.wav")
            _ff(["-i", src, "-ar", "8000", "-ac", "1", "-c:a", "pcm_mulaw", mid])
        else:
            mid = os.path.join(td, "m.amr")
            rate = "7400" if fixed else str(rng.choice(AMR_RATES))
            _ff(["-i", src, "-ar", "8000", "-ac", "1", "-c:a", "libopencore_amrnb",
                 "-b:a", rate, mid])
        _ff(["-i", mid, "-ar", str(SAMPLE_RATE), "-ac", "1", dst])
        return sf.read(dst, dtype="float32")[0]


def d_noise(a, rng, fixed):
    snr = 10.0 if fixed else float(rng.uniform(5, 20))
    n = rng.randn(len(a)).astype(np.float32)
    if (not fixed) and rng.rand() < 0.5:
        n = np.convolve(n, np.ones(8, dtype=np.float32) / 8, mode="same")
    p_sig = float(np.mean(a ** 2)) + 1e-10
    n *= np.sqrt(p_sig / (10 ** (snr / 10)) / (float(np.mean(n ** 2)) + 1e-12))
    return (a + n).astype(np.float32)


def degrade(a, kind, rng, fixed):
    if kind == "mix":
        kind = KINDS[rng.randint(len(KINDS))]
    if kind == "8k":
        out = d_8k(a, rng, fixed)
    elif kind in ("mulaw", "amr"):
        out = d_codec(a, rng, fixed, kind)
    elif kind == "noise":
        out = d_noise(a, rng, fixed)
    else:
        raise ValueError(kind)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def make(pairs, kind, out_dir, seed=0, fixed=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    new = []
    for i, (path, y) in enumerate(pairs):
        dst = out_dir / ("%05d.wav" % i)
        try:
            if not dst.exists():
                a, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
                sf.write(str(dst), degrade(a, kind, rng, fixed), SAMPLE_RATE)
            new.append((str(dst), int(y)))
        except Exception as e:
            print("skip", path, str(e)[:60])
        if (i + 1) % 500 == 0:
            print("degraded", i + 1, "/", len(pairs))
    return new


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--kind", required=True, choices=KINDS + ["mix"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    from ml.dataset import get_pairs
    from ml.extract_embeddings import run
    pairs = get_pairs(a.split, n=a.n)
    tag = a.tag or "%s_%s" % (a.split, a.kind)
    out_dir = ROOT / "data" / "processed" / "aug" / tag
    new = make(pairs, a.kind, out_dir, a.seed, fixed=(a.split != "train"))
    run(new, str(ROOT / "data" / "processed" / (tag + ".npz")))
