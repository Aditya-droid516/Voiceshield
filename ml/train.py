import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ml.evaluate import compute_eer, pretty, summarize
from ml.head import Head

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "models" / "head.pt"


def predict_scores(model, mean, std, X, batch_size=512):
    """Fake probabilities in [0, 1] for embeddings X [N, n_layers, dim]."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.as_tensor(np.asarray(X[i:i + batch_size]), dtype=torch.float32)
            xb = (xb - mean) / std
            out.append(torch.sigmoid(model(xb)).numpy())
    return np.concatenate(out)


def train_head(X_tr, y_tr, X_dev, y_dev, epochs=30, lr=1e-3, batch_size=128,
               weight_decay=1e-4, seed=0, verbose=True):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    X_tr = np.asarray(X_tr, dtype=np.float32)
    y_tr = np.asarray(y_tr).astype(np.float32)
    mean = torch.from_numpy(X_tr.mean(axis=0))
    std = torch.from_numpy(X_tr.std(axis=0) + 1e-6)
    Xt = (torch.from_numpy(X_tr) - mean) / std
    yt = torch.from_numpy(y_tr)

    n_fake = max(float(y_tr.sum()), 1.0)
    n_real = max(float(len(y_tr)) - n_fake, 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_real / n_fake))
    model = Head(X_tr.shape[1], X_tr.shape[2])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best = {"eer": 2.0}
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(Xt))
        total = 0.0
        for i in range(0, len(perm), batch_size):
            idx = torch.from_numpy(perm[i:i + batch_size])
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        dev_scores = predict_scores(model, mean, std, X_dev)
        eer, thr = compute_eer(y_dev, dev_scores)
        if eer < best["eer"]:
            best = {"eer": eer, "thr": thr, "epoch": ep + 1,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
        if verbose:
            print("epoch %2d  loss %.4f  dev EER %.2f%%  (best %.2f%% @ %d)" % (
                ep + 1, total / len(perm), 100 * eer, 100 * best["eer"], best["epoch"]))
    model.load_state_dict(best["state"])
    model.eval()
    info = {"dev_eer": best["eer"], "threshold": best["thr"], "epoch": best["epoch"]}
    return model, mean, std, info


def save_checkpoint(path, model, mean, std, info):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "mean": mean,
        "std": std,
        "n_layers": int(mean.shape[0]),
        "dim": int(mean.shape[1]),
        "dev_eer": float(info["dev_eer"]),
        "threshold": float(info["threshold"]),
    }, path)


def load_checkpoint(path=DEFAULT_OUT):
    ck = torch.load(str(path), map_location="cpu", weights_only=True)
    model = Head(ck["n_layers"], ck["dim"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck["mean"], ck["std"], {"dev_eer": ck["dev_eer"], "threshold": ck["threshold"]}


def _load(path):
    d = np.load(path, allow_pickle=True)
    return d["X"], d["y"].astype(int)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", default=None)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()

    X_tr, y_tr = _load(a.train)
    X_dev, y_dev = _load(a.dev)
    print("train", X_tr.shape, "fake %.1f%%" % (100 * y_tr.mean()),
          "| dev", X_dev.shape, "fake %.1f%%" % (100 * y_dev.mean()))
    model, mean, std, info = train_head(X_tr, y_tr, X_dev, y_dev, epochs=a.epochs)
    save_checkpoint(a.out, model, mean, std, info)
    print("saved", a.out, "| best epoch", info["epoch"])
    print(pretty("DEV ", summarize(y_dev, predict_scores(model, mean, std, X_dev))))
    if a.test:
        X_te, y_te = _load(a.test)
        print(pretty("TEST", summarize(y_te, predict_scores(model, mean, std, X_te))))
    print("layer weights:", np.round(model.layer_weights().detach().numpy(), 3))
