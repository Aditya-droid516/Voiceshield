"""VoiceShield inference: predict(path) -> fake probability, risk level, action."""
from pathlib import Path

import numpy as np
import torch

from ml.extract_embeddings import embed_file, load_encoder
from ml.train import load_checkpoint, predict_scores

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "head.pt"

# Chosen from dev/test score tables (ml/thr_check.py). Higher = more likely fake.
BANDS = [
    (0.90, "HIGH_RISK"),
    (0.50, "SUSPICIOUS"),
    (0.10, "INSUFFICIENT_EVIDENCE"),
    (0.00, "LIKELY_GENUINE"),
]

ACTIONS = {
    "LIKELY_GENUINE": "No strong sign of a synthetic voice. Still, never share OTPs, PINs or passwords on a call you did not start.",
    "INSUFFICIENT_EVIDENCE": "Result is unclear. Do not act on requests for money or codes. Hang up and call back on a number you already know.",
    "SUSPICIOUS": "This voice shows signs of being synthetic. Do not share OTPs or send money. Hang up and call back on a known number.",
    "HIGH_RISK": "Very likely an AI-generated voice. Hang up now, do not share any codes or money, and verify through a number you already trust.",
}


def risk_level(p):
    for cutoff, name in BANDS:
        if p >= cutoff:
            return name
    return "LIKELY_GENUINE"


class VoiceShieldInference:
    def __init__(self, model_path=MODEL_PATH, device="cpu"):
        self.device = device
        self.encoder = load_encoder(device=device)
        self.model, self.mean, self.std, self.info = load_checkpoint(model_path)

    def score_embedding(self, emb):
        emb = np.asarray(emb, dtype=np.float32)[None]
        return float(predict_scores(self.model, self.mean, self.std, emb)[0])

    def predict(self, path):
        emb = embed_file(self.encoder, str(path), device=self.device, strict=True)
        p = self.score_embedding(emb)
        level = risk_level(p)
        return {"fake_probability": round(p, 4), "risk_level": level,
                "recommended_action": ACTIONS[level]}

    def predict_timeline(self, path):
        from ml.streaming import timeline
        return timeline(self, path)
