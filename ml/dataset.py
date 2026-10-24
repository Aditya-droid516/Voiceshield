"""Speaker-disjoint train/dev/test lists for In-the-Wild.

Labels: 1 = spoof/fake, 0 = bona-fide/real (project convention).
"""
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "release_in_the_wild"
SPLITS_CSV = ROOT / "data" / "processed" / "splits.csv"
LABEL_MAP = {"bona-fide": 0, "spoof": 1}


def load_meta(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "meta.csv")
    unknown = set(df["label"].unique()) - set(LABEL_MAP)
    if unknown:
        raise ValueError("Unknown label strings in meta.csv: %s" % unknown)
    df["y"] = df["label"].map(LABEL_MAP).astype(int)
    df["path"] = df["file"].map(lambda f: str(data_dir / f))
    missing = [p for p in df["path"] if not Path(p).exists()]
    if missing:
        print("WARNING: %d files in meta.csv not found, dropping them (e.g. %s)"
              % (len(missing), missing[0]))
        df = df[~df["path"].isin(set(missing))].reset_index(drop=True)
    return df


def make_splits(df, fractions=(0.70, 0.15, 0.15), seed=5):
    """Assign whole speakers to train/dev/test, balancing by clip count."""
    names = ["train", "dev", "test"]
    counts = df["speaker"].value_counts().to_dict()
    speakers = sorted(counts)
    random.Random(seed).shuffle(speakers)
    total = float(sum(counts.values()))
    target = {n: f * total for n, f in zip(names, fractions)}
    filled = {n: 0.0 for n in names}
    assign = {}
    for spk in speakers:
        # give the speaker to the split furthest below its target share
        best = max(names, key=lambda n: (target[n] - filled[n]) / target[n])
        assign[spk] = best
        filled[best] += counts[spk]
    df = df.copy()
    df["split"] = df["speaker"].map(assign)
    return {n: df[df["split"] == n].reset_index(drop=True) for n in names}


def subsample(df, n, seed=0):
    """Random subsample of up to n clips, as close to 50/50 real/fake as possible."""
    if n is None or n >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    real = df[df["y"] == 0].sample(frac=1.0, random_state=seed)
    fake = df[df["y"] == 1].sample(frac=1.0, random_state=seed)
    k = n // 2
    n_fake = min(k, len(fake))
    n_real = min(n - n_fake, len(real))
    n_fake = min(n - n_real, len(fake))
    out = pd.concat([real.iloc[:n_real], fake.iloc[:n_fake]])
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def to_pairs(df):
    """List of (path, label) as used by ml.extract_embeddings.run."""
    return list(zip(df["path"].tolist(), df["y"].astype(int).tolist()))


def get_pairs(split, n=None, seed=5, data_dir=DATA_DIR):
    splits = make_splits(load_meta(data_dir), seed=seed)
    return to_pairs(subsample(splits[split], n, seed=seed))


def summarize_splits(splits):
    print("%-6s %7s %9s %7s %7s %7s" % ("split", "clips", "speakers", "real", "fake", "%fake"))
    for name, d in splits.items():
        fake = int(d["y"].sum())
        real = len(d) - fake
        print("%-6s %7d %9d %7d %7d %6.1f%%" % (
            name, len(d), d["speaker"].nunique(), real, fake,
            100.0 * fake / max(len(d), 1)))
        if real == 0 or fake == 0:
            print("  WARNING: split %s has only one class" % name)
    sets = [set(d["speaker"]) for d in splits.values()]
    overlap = (sets[0] & sets[1]) | (sets[0] & sets[2]) | (sets[1] & sets[2])
    print("Speaker overlap between splits:", len(overlap), "(must be 0)")


if __name__ == "__main__":
    meta = load_meta()
    splits = make_splits(meta)
    summarize_splits(splits)
    SPLITS_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(splits.values())[["file", "speaker", "label", "y", "split"]].to_csv(
        SPLITS_CSV, index=False)
    print("Saved", SPLITS_CSV)


