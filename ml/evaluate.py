import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_eer(y_true, scores):
    """Equal error rate and the score threshold where it occurs. 1 = fake, higher score = more fake."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    fpr, tpr, thr = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2.0), float(thr[i])


def summarize(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    eer, thr = compute_eer(y_true, scores)
    try:
        auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        auc = float("nan")
    return {
        "eer": eer,
        "auc": auc,
        "threshold": thr,
        "n": int(len(y_true)),
        "n_fake": int(y_true.sum()),
        "n_real": int(len(y_true) - y_true.sum()),
    }


def pretty(name, s):
    return "%s: EER %.2f%% | AUC %.4f | n=%d (real %d, fake %d) | thr %.3f" % (
        name, 100 * s["eer"], s["auc"], s["n"], s["n_real"], s["n_fake"], s["threshold"])
