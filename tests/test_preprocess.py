"""Unit tests for audio preprocessing and validation."""
import numpy as np
import pytest
import soundfile as sf
import tempfile
from pathlib import Path

from ml.preprocess import (
    AudioError,
    enforce_duration,
    fix_length,
    load_audio,
    normalize_loudness,
    preprocess_audio,
    standardize,
    trim_edge_silence,
)


@pytest.fixture
def temp_audio_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


def test_valid_speech_clip(temp_audio_dir):
    sr = 16000
    t = np.arange(int(sr * 3.5)) / sr
    audio = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    path = temp_audio_dir / "valid_clip.wav"
    sf.write(str(path), audio, sr)

    proc_audio, out_sr = preprocess_audio(path, strict=True)
    assert out_sr == 16000
    assert len(proc_audio) > 0
    rms = np.sqrt(np.mean(proc_audio ** 2))
    assert 0.03 <= rms <= 0.07  # normalized near 0.05


def test_short_clip_strict_vs_non_strict(temp_audio_dir):
    sr = 16000
    t = np.arange(int(sr * 1.0)) / sr  # 1.0s clip (under 2s limit)
    audio = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    path = temp_audio_dir / "short_clip.wav"
    sf.write(str(path), audio, sr)

    # In strict mode (API inference), raises AudioError
    with pytest.raises(AudioError) as exc_info:
        preprocess_audio(path, strict=True)
    assert "Need at least 2s" in str(exc_info.value)

    # In non-strict mode (training), must not raise
    proc_audio, out_sr = preprocess_audio(path, strict=False)
    assert out_sr == 16000
    assert len(proc_audio) > 0


def test_silent_clip_raises_audio_error(temp_audio_dir):
    sr = 16000
    silent = np.zeros(int(sr * 3.0), dtype=np.float32)
    path = temp_audio_dir / "silent.wav"
    sf.write(str(path), silent, sr)

    with pytest.raises(AudioError):
        preprocess_audio(path, strict=True)


def test_standardize_and_fix_length():
    audio = np.random.randn(16000).astype(np.float32) * 5 + 10
    std_audio = standardize(audio)
    assert abs(float(std_audio.mean())) < 1e-4
    assert abs(float(std_audio.std()) - 1.0) < 1e-3

    fixed = fix_length(audio, seconds=4.0, sr=16000)
    assert len(fixed) == 64000
