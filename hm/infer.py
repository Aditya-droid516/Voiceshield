"""VoiceShield Hand-Made Feature Inference Engine.

Drop-in replacement for ml.inference.VoiceShieldInference.
Contract:
- Class: VoiceShieldInference
- predict(path) -> fake_probability, risk_level, recommended_action,
                   prediction ("Real", "Fake", "Uncertain"), confidence,
                   details (with explanation of acoustic cues).
- predict_timeline(path) -> times, scores, raw_scores, alert_time, ms_per_update.
- Optional WavLM support behind a flag (default off).
"""

import time
from pathlib import Path

import joblib
import numpy as np

from hm.features import FEATURE_GROUPS, extract_from_file, to_vector
from ml.preprocess import AudioError

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HM_MODEL = ROOT / "models" / "hm_model.joblib"
DEFAULT_WAVLM_MODEL = ROOT / "models" / "head.pt"

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


def get_risk_level(p, bands):
    for cutoff, name in bands:
        if p >= cutoff:
            return name
    return "LIKELY_GENUINE"


class VoiceShieldInference:
    """Drop-in replacement for ml.inference.VoiceShieldInference.

    Computes explainable acoustic features and outputs forensic verdict,
    calibrated confidence, and acoustic cue explanations.
    """

    def __init__(self, model_path=DEFAULT_HM_MODEL, use_wavlm=False, device="cpu"):
        self.device = device
        self.use_wavlm = use_wavlm
        self.model_path = Path(model_path)
        
        # Load hand-made bundle
        if not self.model_path.exists():
            # Fall back to baseline if hm_model not yet trained
            self.bundle = None
        else:
            self.bundle = joblib.load(str(self.model_path))

        # Optional WavLM support behind a flag (default OFF)
        self.wavlm_infer = None
        if self.use_wavlm:
            try:
                from ml.inference import VoiceShieldInference as BaselineInference
                self.wavlm_infer = BaselineInference(model_path=DEFAULT_WAVLM_MODEL, device=device)
            except Exception as e:
                print("Could not load WavLM baseline for hybrid inference:", e)

    def explain_cues(self, feat_vec):
        """Generate human-understandable forensic explanations from acoustic features."""
        if self.bundle is None:
            return ["Baseline model active."]
            
        scaler = self.bundle["scaler"]
        stable_mask = self.bundle["stable_mask"]
        fnames = self.bundle["feature_names"]
        
        # Normalized z-scores
        z = (feat_vec[stable_mask] - scaler.mean_) / scaler.scale_
        
        explanations = []
        # Check pitch & prosody
        pitch_indices = [i for i, fn in enumerate(fnames) if "pitch" in fn]
        if pitch_indices:
            pitch_z = np.mean(z[pitch_indices])
            if pitch_z < -1.0:
                explanations.append("Abnormally flat pitch contour and reduced prosodic variation (common in text-to-speech).")
            elif pitch_z > 1.5:
                explanations.append("High pitch instability and frame jump rate.")

        # Check jitter/shimmer
        js_indices = [i for i, fn in enumerate(fnames) if "jitter" in fn or "shimmer" in fn]
        if js_indices:
            js_z = np.mean(z[js_indices])
            if js_z < -1.2:
                explanations.append("Micro-prosodic perturbation (jitter/shimmer) is unnaturally smooth, typical of synthetic neural synthesis.")

        # Check vocoder / high-freq
        lfcc_indices = [i for i, fn in enumerate(fnames) if "lfcc" in fn or "high_to_low" in fn]
        if lfcc_indices:
            lfcc_z = np.mean(np.abs(z[lfcc_indices]))
            if lfcc_z > 1.8:
                explanations.append("Spectral filterbank distribution exhibits vocoder synthesis artifacts.")

        # Check formants
        formant_indices = [i for i, fn in enumerate(fnames) if "f1_" in fn or "f2_" in fn or "f3_" in fn]
        if formant_indices:
            formant_z = np.mean(z[formant_indices])
            if abs(formant_z) > 1.5:
                explanations.append("Vocal tract resonance steps deviate from typical human articulation.")

        if not explanations:
            explanations.append("Acoustic parameters and harmonic characteristics align with typical human vocal tracts.")

        return explanations

    def predict(self, path):
        path = str(path)
        t0 = time.time()
        
        # 1. Hand-made Feature Pipeline
        fd = extract_from_file(path, strict=True)
        feat_vec = to_vector(fd)
        
        if self.bundle is not None:
            sub_vec = feat_vec[self.bundle["stable_mask"]][None]
            scaled = self.bundle["scaler"].transform(sub_vec)
            p_hm = float(self.bundle["model"].predict_proba(scaled)[0, 1])
            bands = self.bundle.get("bands", [
                (0.90, "HIGH_RISK"),
                (0.50, "SUSPICIOUS"),
                (0.10, "INSUFFICIENT_EVIDENCE"),
                (0.00, "LIKELY_GENUINE"),
            ])
        else:
            p_hm = 0.5
            bands = [(0.9, "HIGH_RISK"), (0.5, "SUSPICIOUS"), (0.1, "INSUFFICIENT_EVIDENCE"), (0.0, "LIKELY_GENUINE")]

        # 2. Hybrid combination if WavLM flag is active
        if self.use_wavlm and self.wavlm_infer is not None:
            res_w = self.wavlm_infer.predict(path)
            p_wavlm = res_w["fake_probability"]
            # Weighted ensemble: 0.7 hand-made, 0.3 WavLM
            p = float(np.clip(0.70 * p_hm + 0.30 * p_wavlm, 0.0, 1.0))
        else:
            p = p_hm

        p = round(float(p), 4)
        level = get_risk_level(p, bands)
        verdict = VERDICT[level]
        conf = round(1.0 - p if verdict == "Real" else p, 4)

        cues = self.explain_cues(feat_vec)

        base = {
            "fake_probability": p,
            "risk_level": level,
            "recommended_action": ACTIONS[level],
            "explanation": cues,
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }
        out = dict(base)
        out.update({
            "prediction": verdict,
            "confidence": conf,
            "details": dict(base),
        })
        return out

    def predict_timeline(self, path):
        from hm.streaming import timeline
        return timeline(self, path)
