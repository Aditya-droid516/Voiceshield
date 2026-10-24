import librosa
import numpy as np
import soundfile as sf

from src.config import SAMPLE_RATE, MIN_DURATION, MAX_DURATION, TRAIN_SEGMENT_SECONDS

TARGET_SR = SAMPLE_RATE


class AudioError(ValueError):
    """User-facing audio problem (silent, too short, unreadable)."""


def load_audio(file_path):
    try:
        audio, sr = librosa.load(str(file_path), sr=TARGET_SR, mono=True)
    except Exception as e:
        raise AudioError(f"Could not read audio file: {e}") from e
    if audio.size == 0:
        raise AudioError("Audio file is empty.")
    return audio.astype(np.float32), sr


def trim_edge_silence(audio, top_db=30):
    intervals = librosa.effects.split(audio, top_db=top_db)
    if len(intervals) == 0:
        raise AudioError("No speech detected in audio.")
    return audio[intervals[0][0]:intervals[-1][1]]


def normalize_loudness(audio, target_rms=0.05):
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 1e-8:
        raise AudioError("Audio is silent or too quiet.")
    return np.clip(audio * (target_rms / rms), -1.0, 1.0).astype(np.float32)


def enforce_duration(audio, sr=TARGET_SR, strict=True):
    min_samples = int(MIN_DURATION * sr)
    max_samples = int(MAX_DURATION * sr)
    if len(audio) < min_samples and strict:
        raise AudioError(
            f"Audio is too short ({len(audio) / sr:.1f}s). Need at least {MIN_DURATION}s of speech."
        )
    return audio[:max_samples]


def fix_length(audio, seconds=TRAIN_SEGMENT_SECONDS, sr=TARGET_SR, random_crop=False):
    """Crop or tile-pad to exactly `seconds` so a DataLoader can batch clips."""
    n = int(seconds * sr)
    if len(audio) >= n:
        start = np.random.randint(0, len(audio) - n + 1) if random_crop else 0
        return audio[start:start + n]
    reps = int(np.ceil(n / len(audio)))
    return np.tile(audio, reps)[:n]


def standardize(audio):
    """Zero-mean / unit-variance. Use in BOTH training and inference."""
    return ((audio - audio.mean()) / (audio.std() + 1e-7)).astype(np.float32)


def preprocess_audio(file_path, strict=True):
    audio, sr = load_audio(file_path)
    try:
        audio = trim_edge_silence(audio)
    except AudioError:
        if strict:
            raise
    audio = normalize_loudness(audio)
    audio = enforce_duration(audio, sr, strict=strict)
    return audio, sr


def save_processed_audio(audio, sr, output_path):
    sf.write(output_path, audio, sr)
