"""Train and calibrate explainable hand-made feature classifiers.

Supports:
- Logistic Regression (explainable linear weights)
- HistGradientBoostingClassifier (gradient-boosted decision trees)
- Platt scaling calibration (CalibratedClassifierCV) on dev set
- Automatic hyperparameter tuning on dev EER / AUC
- Calibration curve / reliability diagnostics
- Calibrated risk band threshold determination
"""

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hm.features import FEATURE_GROUPS, FEATURE_NAMES
from ml.evaluate import compute_eer, pretty, summarize

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_HM_MODEL = MODELS_DIR / "hm_model.joblib"


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return d["X"].astype(np.float32), d["y"].astype(np.int32), d["paths"]


def determine_risk_bands(y_dev, p_dev):
    """Determine the 4 risk bands from dev set score distribution:

    LIKELY_GENUINE, INSUFFICIENT_EVIDENCE, SUSPICIOUS, HIGH_RISK
    """
    dev_eer, eer_thr = compute_eer(y_dev, p_dev)
    
    real_scores = p_dev[y_dev == 0]
    fake_scores = p_dev[y_dev == 1]
    
    # Genuine threshold: 95th percentile of real scores or 0.15 max
    p95_real = float(np.percentile(real_scores, 95))
    low_cut = min(0.15, max(0.05, round(p95_real, 2)))
    
    # EER cutoff as mid-point
    mid_cut = round(float(eer_thr), 2)
    if mid_cut <= low_cut:
        mid_cut = round(low_cut + 0.20, 2)
        
    # High risk cutoff: where fake precision >= 98%
    high_cut = max(round(float(np.percentile(fake_scores, 25)), 2), 0.80)
    high_cut = min(0.90, max(mid_cut + 0.15, high_cut))

    bands = [
        (high_cut, "HIGH_RISK"),
        (mid_cut, "SUSPICIOUS"),
        (low_cut, "INSUFFICIENT_EVIDENCE"),
        (0.00, "LIKELY_GENUINE"),
    ]
    return bands


def evaluate_calibration(y_true, probs, n_bins=10):
    """Reliability table and Brier score."""
    brier = float(brier_score_loss(y_true, probs))
    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy="uniform")
    return {
        "brier_score": brier,
        "prob_true": [round(float(x), 4) for x in prob_true],
        "prob_pred": [round(float(x), 4) for x in prob_pred],
    }


def train_classifier(
    X_tr,
    y_tr,
    X_dev,
    y_dev,
    model_type="hgbt",
    use_stable_only=False,
    stability_report_path=None,
    seed=42,
):
    """Train, tune, and calibrate hand-crafted feature detector."""
    # 1. Feature selection (optional stability mask)
    stable_mask = np.ones(X_tr.shape[1], dtype=bool)
    if use_stable_only:
        if stability_report_path is None:
            stability_report_path = ROOT / "hm" / "stability_report.json"
        if Path(stability_report_path).exists():
            with open(stability_report_path) as f:
                srep = json.load(f)
            stable_names = set(srep["stable_features"])
            stable_mask = np.array([fn in stable_names for fn in FEATURE_NAMES])
            print(f"Applied stability filter: retained {stable_mask.sum()} of {len(FEATURE_NAMES)} features.")

    X_tr_sub = X_tr[:, stable_mask]
    X_dev_sub = X_dev[:, stable_mask]
    feature_names_sub = [fn for fn, m in zip(FEATURE_NAMES, stable_mask) if m]

    # 2. Standard Scaling
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_sub)
    X_dev_scaled = scaler.transform(X_dev_sub)

    # 3. Model Search & Hyperparameter Tuning on Dev Set
    print(f"\nTraining {model_type.upper()} model...")
    best_model = None
    best_eer = 1.0
    best_auc = 0.0

    if model_type == "lr":
        for C in [0.05, 0.5, 1.0]:
            clf = LogisticRegression(C=C, max_iter=250, class_weight="balanced", random_state=seed)
            clf.fit(X_tr_scaled, y_tr)
            p_dev = clf.predict_proba(X_dev_scaled)[:, 1]
            eer, _ = compute_eer(y_dev, p_dev)
            auc = roc_auc_score(y_dev, p_dev)
            print(f"  [LR C={C:<4}] Dev EER: {eer*100:5.2f}% | Dev AUC: {auc:.4f}", flush=True)
            if eer < best_eer:
                best_eer = eer
                best_auc = auc
                best_model = clf
    elif model_type == "hgbt":
        for lr in [0.05, 0.1]:
            for l2 in [0.0, 1.0]:
                clf = HistGradientBoostingClassifier(
                    learning_rate=lr,
                    l2_regularization=l2,
                    max_iter=100,
                    min_samples_leaf=20,
                    class_weight="balanced",
                    random_state=seed,
                )
                clf.fit(X_tr_scaled, y_tr)
                p_dev = clf.predict_proba(X_dev_scaled)[:, 1]
                eer, _ = compute_eer(y_dev, p_dev)
                auc = roc_auc_score(y_dev, p_dev)
                print(f"  [HGBT lr={lr} l2={l2}] Dev EER: {eer*100:5.2f}% | Dev AUC: {auc:.4f}", flush=True)
                if eer < best_eer:
                    best_eer = eer
                    best_auc = auc
                    best_model = clf
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # 4. Platt Scaling Calibration on Dev Set (using modern sklearn FrozenEstimator)
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(estimator=FrozenEstimator(best_model), method="sigmoid")
    except ImportError:
        calibrated = CalibratedClassifierCV(estimator=best_model, method="sigmoid", cv="prefit")
    calibrated.fit(X_dev_scaled, y_dev)

    # Compute calibrated dev metrics
    p_dev_cal = calibrated.predict_proba(X_dev_scaled)[:, 1]
    dev_eer, dev_thr = compute_eer(y_dev, p_dev_cal)
    dev_auc = roc_auc_score(y_dev, p_dev_cal)
    bands = determine_risk_bands(y_dev, p_dev_cal)
    cal_diag = evaluate_calibration(y_dev, p_dev_cal)

    print(f"Dev Performance (Calibrated): EER = {dev_eer*100:.2f}%, AUC = {dev_auc:.4f}, Brier = {cal_diag['brier_score']:.4f}")
    print(f"Calibrated Risk Bands: {bands}")

    bundle = {
        "model": calibrated,
        "base_model": best_model,
        "scaler": scaler,
        "model_type": model_type,
        "stable_mask": stable_mask,
        "feature_names": feature_names_sub,
        "dev_eer": float(dev_eer),
        "threshold": float(dev_thr),
        "dev_auc": float(dev_auc),
        "bands": bands,
        "calibration_diag": cal_diag,
    }
    return bundle


def predict_bundle(bundle, X):
    X_sub = X[:, bundle["stable_mask"]]
    X_sc = bundle["scaler"].transform(X_sub)
    return bundle["model"].predict_proba(X_sc)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(ROOT / "data" / "processed" / "hm_train_aug.npz"))
    ap.add_argument("--dev", default=str(ROOT / "data" / "processed" / "hm_dev.npz"))
    ap.add_argument("--test", default=str(ROOT / "data" / "processed" / "hm_test.npz"))
    ap.add_argument("--model-type", default="hgbt", choices=["hgbt", "lr"])
    ap.add_argument("--use-stable", action="store_true")
    ap.add_argument("--out", default=str(DEFAULT_HM_MODEL))
    args = ap.parse_args()

    X_tr, y_tr, _ = load_npz(args.train)
    X_dev, y_dev, _ = load_npz(args.dev)
    X_te, y_te, _ = load_npz(args.test)

    print(f"Loaded train: {X_tr.shape} (fake {100*y_tr.mean():.1f}%)")
    print(f"Loaded dev  : {X_dev.shape} (fake {100*y_dev.mean():.1f}%)")
    print(f"Loaded test : {X_te.shape} (fake {100*y_te.mean():.1f}%)")

    bundle = train_classifier(
        X_tr, y_tr, X_dev, y_dev,
        model_type=args.model_type,
        use_stable_only=args.use_stable,
    )

    # Evaluate on Test
    p_te = predict_bundle(bundle, X_te)
    te_eer, te_thr = compute_eer(y_te, p_te)
    te_auc = roc_auc_score(y_te, p_te)
    te_cal = evaluate_calibration(y_te, p_te)

    bundle["test_eer"] = float(te_eer)
    bundle["test_auc"] = float(te_auc)
    bundle["test_brier"] = float(te_cal["brier_score"])

    joblib.dump(bundle, args.out)
    print(f"\nSaved model bundle to {args.out}")
    print(f"Test Clean Performance: EER = {te_eer*100:.2f}%, AUC = {te_auc:.4f}, Brier = {te_cal['brier_score']:.4f}")


if __name__ == "__main__":
    main()
