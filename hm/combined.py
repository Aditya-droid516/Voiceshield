from ml.inference import VoiceShieldInference as WavLMEngine, ACTIONS, VERDICT, risk_level
from hm.infer import VoiceShieldInference as HandMadeEngine

W_HM = 0.6
DISAGREE = 0.5


def _p(res):
    for src in (res, res.get("details") or {}):
        if "fake_probability" in src:
            return float(src["fake_probability"])
    raise KeyError("no fake_probability in result keys: %s" % list(res))


class VoiceShieldInference:
    def __init__(self, w_hm=W_HM):
        self.hm = HandMadeEngine()
        self.wl = WavLMEngine()
        self.w_hm = w_hm

    def predict(self, path):
        a = _p(self.hm.predict(str(path)))
        b = _p(self.wl.predict(str(path)))
        p = round(self.w_hm * a + (1 - self.w_hm) * b, 4)
        level = risk_level(p)
        disagree = abs(a - b) >= DISAGREE
        if disagree and level in ("HIGH_RISK", "SUSPICIOUS"):
            level = "INSUFFICIENT_EVIDENCE"
        verdict = VERDICT[level]
        conf = round(1.0 - p if verdict == "Real" else p, 4)
        base = {"fake_probability": p, "risk_level": level, "recommended_action": ACTIONS[level]}
        details = dict(base)
        details.update({"handmade_score": round(a, 4), "wavlm_score": round(b, 4), "models_disagree": bool(disagree)})
        out = dict(base)
        out.update({"prediction": verdict, "confidence": conf, "details": details})
        return out

    def predict_timeline(self, path):
        fn = getattr(self.hm, "predict_timeline", None) or self.wl.predict_timeline
        return fn(str(path))
