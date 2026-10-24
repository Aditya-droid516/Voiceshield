"""VoiceShield inference: predict(path) -> verdict, fake probability, risk level, action."""
from pathlib import Path

import numpy as np

from ml.extract_embeddings import embed_file, load_encoder
from ml.train import load_checkpoint, predict_scores

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "head.pt"

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

VERDICT = {
    "HIGH_RISK": "Fake",
    "SUSPICIOUS": "Fake",
    "INSUFFICIENT_EVIDENCE": "Uncertain",
    "LIKELY_GENUINE": "Real",
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
        p = round(self.score_embedding(emb), 4)
        level = risk_level(p)
        verdict = VERDICT[level]
        conf = round(1.0 - p if verdict == "Real" else p, 4)
        base = {"fake_probability": p, "risk_level": level,
                "recommended_action": ACTIONS[level]}
        out = dict(base)
        out.update({"prediction": verdict, "confidence": conf, "details": dict(base)})
        return out

    def predict_timeline(self, path):
        from ml.streaming import timeline
        return timeline(self, path)
