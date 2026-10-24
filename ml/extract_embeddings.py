"""Turn audio files into cached WavLM embeddings: one [13, 768] array per clip.

WavLM is frozen, so we run it ONCE per clip and train the small head on the cache.
Resumable: if the output file already exists, finished clips are skipped.

    from ml.extract_embeddings import run
    run(pairs, "data/processed/train.npz", limit=3000)   # pairs = [(path, label), ...]
"""
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ml.config import WAVLM_MODEL_NAME, TRAIN_SEGMENT_SECONDS
from ml.preprocess import AudioError, fix_length, preprocess_audio, standardize


def load_encoder(model_name=WAVLM_MODEL_NAME, device="cpu"):
    from transformers import AutoModel
    enc = AutoModel.from_pretrained(model_name)
    return enc.to(device).eval()


@torch.inference_mode()
def embed_waveform(encoder, audio, device="cpu"):
    """audio: 1-D float32 at 16 kHz -> np.ndarray [n_layers+1, hidden]."""
    audio = standardize(fix_length(audio, TRAIN_SEGMENT_SECONDS))
    x = torch.from_numpy(audio)[None].to(device)
    hs = encoder(x, output_hidden_states=True).hidden_states
    return torch.stack([h.mean(dim=1) for h in hs], dim=1)[0].cpu().numpy().astype(np.float32)


def embed_file(encoder, path, device="cpu", strict=False):
    audio, _ = preprocess_audio(path, strict=strict)
    return embed_waveform(encoder, audio, device)


def run(pairs, out_path, limit=None, encoder=None, device="cpu", save_every=250):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs = list(pairs)[:limit] if limit else list(pairs)

    X, y, paths = [], [], []
    if out_path.exists():
        d = np.load(out_path, allow_pickle=True)
        X, y, paths = list(d["X"]), list(d["y"]), list(d["paths"])
    done = set(paths)
    todo = [(p, l) for p, l in pairs if str(p) not in done]
    print(f"{len(done)} already done, {len(todo)} to go")
    if not todo:
        return out_path
    if encoder is None:
        encoder = load_encoder(device=device)

    def save():
        np.savez(out_path, X=np.stack(X), y=np.array(y), paths=np.array(paths))

    skipped = 0
    for n, (p, label) in enumerate(tqdm(todo), 1):
        try:
            X.append(embed_file(encoder, p, device))
        except (AudioError, RuntimeError, ValueError) as e:
            skipped += 1
            tqdm.write(f"skip {p}: {e}")
            continue
        y.append(int(label))
        paths.append(str(p))
        if n % save_every == 0:
            save()
    if X:
        save()
    print(f"saved {len(X)} clips to {out_path} ({skipped} skipped)")
    return out_path