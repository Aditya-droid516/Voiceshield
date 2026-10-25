"""Unit tests for VoiceShieldInference contract and drop-in compatibility."""
import numpy as np
import pytest
import soundfile as sf
import tempfile
from pathlib import Path

from hm.infer import VoiceShieldInference


@pytest.fixture
def temp_audio_clip():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "test_clip.wav"
        sr = 16000
        t = np.arange(int(sr * 3.0)) / sr
        sig = (0.2 * np.sin(2 * np.pi * 350 * t)).astype(np.float32)
        sf.write(str(p), sig, sr)
        yield p


def test_inference_contract_keys(temp_audio_clip):
    infer = VoiceShieldInference()
    result = infer.predict(temp_audio_clip)

    # Core ML keys
    assert "fake_probability" in result
    assert "risk_level" in result
    assert "recommended_action" in result
    assert 0.0 <= result["fake_probability"] <= 1.0
    assert result["risk_level"] in ["LIKELY_GENUINE", "INSUFFICIENT_EVIDENCE", "SUSPICIOUS", "HIGH_RISK"]

    # Backend / frontend expected keys
    assert "prediction" in result
    assert result["prediction"] in ["Real", "Fake", "Uncertain"]
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert "details" in result
    assert isinstance(result["details"], dict)

    # Explainability key
    assert "explanation" in result["details"]
    assert len(result["details"]["explanation"]) > 0


def test_timeline_contract_keys(temp_audio_clip):
    infer = VoiceShieldInference()
    res = infer.predict_timeline(temp_audio_clip)

    assert "times" in res
    assert "scores" in res
    assert "raw_scores" in res
    assert "alert_time" in res
    assert "ms_per_update" in res

    assert len(res["times"]) == len(res["scores"])
    assert res["ms_per_update"] < 500.0  # limit is 500ms
