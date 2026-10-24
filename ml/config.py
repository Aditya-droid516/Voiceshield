from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DEMO_DIR = DATA_DIR / "demo"
MODEL_DIR = BASE_DIR / "models"

SAMPLE_RATE = 16000
MIN_DURATION = 2
MAX_DURATION = 10
TRAIN_SEGMENT_SECONDS = 4

CHUNK_SECONDS = 2
HOP_SECONDS = 1

WAVLM_MODEL_NAME = "microsoft/wavlm-base"

RISK_THRESHOLDS = {
    "genuine": 0.35,
    "uncertain": 0.60,
    "suspicious": 0.85,
}
