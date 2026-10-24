"""
VoiceShield API v2.1

pip install fastapi uvicorn python-multipart librosa soundfile mutagen scipy torch transformers

Run:  uvicorn voiceshield_api:app --host 0.0.0.0 --port 8000
Env:  HEAD_PATH (trained head, default head.pt), ALLOWED_ORIGINS (comma list, default "*")
Note: .mp3/.m4a/.ogg decoding needs ffmpeg installed on the server.

Live-call WebSocket protocol  (/ws/realtime-call)
  1. (optional) client sends a TEXT frame first: {"sample_rate": 48000}
     (default is 16000 -- best to resample to 16 kHz in the browser)
  2. client then streams BINARY frames: mono, 16-bit little-endian PCM
  3. server replies after every ~0.5 s of new audio with a JSON score
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import struct
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import librosa
import mutagen
import numpy as np
import torch
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, WavLMModel

log = logging.getLogger("voiceshield")
logging.basicConfig(level=logging.INFO)

# ----------------------------------------------------------------- config ---
SR = 16_000
BACKBONE = "microsoft/wavlm-base"
HEAD_PATH = os.getenv("HEAD_PATH", "head.pt")

CHUNK_SECONDS = 2.0          # file analysis window
MIN_CHUNK_SECONDS = 0.5      # shorter tail fragments are dropped
SILENCE_RMS = 1e-3           # below this a window is treated as silence
MAX_AUDIO_SECONDS = 600      # decode at most 10 minutes
FAKE_THRESHOLD = 0.5         # file-level verdict threshold

LIVE_WINDOW_SECONDS = 2.0    # audio context the model sees live
LIVE_HOP_SECONDS = 0.5       # how often a live score is produced
LIVE_MIN_SECONDS = 1.0       # warm-up before first live score
LIVE_SMOOTH_N = 5            # moving average over last N scores
LIVE_ALERT_THRESHOLD = 0.75
LIVE_ALERT_CONSECUTIVE = 2   # alert only after N smoothed hits in a row

MAX_UPLOAD_MB = 25
ALLOWED_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

UPLOAD_DIR = Path("secure_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------------ model ---
class Detector:
    # Uses the team ML pipeline (ml/inference.py) so serving matches training.
    def __init__(self) -> None:
        from ml.inference import VoiceShieldInference
        self.device = "cpu"
        self.engine = VoiceShieldInference()

    def score(self, wav: np.ndarray) -> float:
        # Probability (0-1) that a 16 kHz mono float32 waveform is fake.
        import os, tempfile
        import soundfile as sf
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        min_len = int(SR * 2.1)
        if wav.size < min_len:
            wav = np.tile(wav, int(np.ceil(min_len / max(wav.size, 1))))[:min_len]
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            sf.write(tmp.name, wav, SR)
            return float(self.engine.predict(tmp.name)["fake_probability"])
        except Exception:
            return 0.5  # too short or silent after trimming: no evidence either way
        finally:
            try:
                os.remove(tmp.name)
            except OSError:
                pass


detector: Detector | None = None
live_sem = asyncio.Semaphore(2)  # limit concurrent live inferences


@asynccontextmanager
async def lifespan(_: FastAPI):
    global detector
    detector = Detector()
    log.info("Model loaded on %s", detector.device)
    yield


app = FastAPI(title="VoiceShield Enterprise API", version="2.1", lifespan=lifespan)

_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    # browsers reject "*" together with credentials
    allow_credentials=_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------ FEATURE 1: unique naming ---
def safe_display_name(name: str | None) -> str:
    """Original name, sanitised -- only for display, never used on disk."""
    base = Path(name or "unknown").name
    return re.sub(r"[^\w.\- ]", "_", base)[:100]


async def save_unique_audio(file: UploadFile) -> tuple[Path, int]:
    """
    Stores the upload under a random UUID name, so two users uploading
    'call.wav' at the same time can never overwrite each other.
    Mode "xb" makes the OS refuse to open a file that already exists.
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(
            400, f"Invalid audio format. Supported: {', '.join(sorted(ALLOWED_EXT))}"
        )

    path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    size, limit = 0, MAX_UPLOAD_MB * 1024 * 1024
    try:
        with open(path, "xb") as out:
            while block := await file.read(1024 * 1024):
                size += len(block)
                if size > limit:
                    raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")
                out.write(block)
        if size == 0:
            raise HTTPException(400, "Empty file")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path, size


# ------------------------------------------ FEATURE 2: metadata inspection ---
# (regex on raw bytes, display name, kind)
SIGNATURES: list[tuple[bytes, str, str]] = [
    (rb"audacity", "Audacity", "editor"),
    (rb"adobe", "Adobe (Audition/Premiere)", "editor"),
    (rb"pro ?tools", "Pro Tools", "editor"),
    (rb"reaper", "REAPER", "editor"),
    (rb"ableton", "Ableton Live", "editor"),
    (rb"garageband|logic pro", "Apple GarageBand/Logic", "editor"),
    (rb"fl studio", "FL Studio", "editor"),
    (rb"cubase|steinberg|wavelab", "Steinberg", "editor"),
    (rb"wavepad|nch software", "WavePad/NCH", "editor"),
    (rb"ocenaudio", "ocenaudio", "editor"),
    (rb"sound forge|magix", "Sound Forge/MAGIX", "editor"),
    (rb"goldwave", "GoldWave", "editor"),
    (rb"izotope", "iZotope", "editor"),
    (rb"lavf\d|lavc\d|ffmpeg", "FFmpeg (libavformat/libavcodec)", "converter"),
    (rb"lame\s?\d", "LAME MP3 encoder", "converter"),
    (rb"\bsox\b", "SoX", "converter"),
]
PRO_WAV_CHUNKS = {b"bext": "Broadcast WAV (bext)", b"iXML": "iXML (field recorder / DAW)"}


def parse_wav_chunks(path: Path) -> dict:
    """Walks the RIFF structure: chunk ids + LIST/INFO text (ISFT = software)."""
    result: dict = {"chunk_ids": [], "info": {}}
    try:
        with open(path, "rb") as f:
            head = f.read(12)
            if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
                return result
            while True:
                hdr = f.read(8)
                if len(hdr) < 8:
                    break
                cid, size = hdr[:4], struct.unpack("<I", hdr[4:])[0]
                result["chunk_ids"].append(cid.decode("latin-1"))
                if cid == b"data" or size > 4 * 1024 * 1024:
                    f.seek(size + (size & 1), 1)
                    continue
                body = f.read(size)
                if size & 1:
                    f.read(1)
                if cid == b"LIST" and body[:4] == b"INFO":
                    pos = 4
                    while pos + 8 <= len(body):
                        sid = body[pos : pos + 4].decode("latin-1")
                        slen = struct.unpack("<I", body[pos + 4 : pos + 8])[0]
                        val = body[pos + 8 : pos + 8 + slen]
                        result["info"][sid] = val.split(b"\x00")[0].decode(
                            "utf-8", "replace"
                        )
                        pos += 8 + slen + (slen & 1)
    except Exception as exc:  # corrupt header is itself worth reporting
        result["parse_error"] = str(exc)
    return result


def inspect_audio_metadata(path: Path) -> dict:
    findings: list[dict] = []
    technical: dict = {}
    tag_text = ""

    # --- container / codec info + tags (mp3, flac, m4a, ogg, wav) ---
    try:
        mf = mutagen.File(str(path))
        if mf is not None:
            info = mf.info
            technical = {
                "format": type(mf).__name__,
                "duration_seconds": round(getattr(info, "length", 0) or 0, 2),
                "sample_rate_hz": getattr(info, "sample_rate", None),
                "channels": getattr(info, "channels", None),
                "bitrate_kbps": (
                    round(info.bitrate / 1000) if getattr(info, "bitrate", 0) else None
                ),
            }
            if mf.tags:
                tag_text = "\n".join(f"{k}={v}" for k, v in mf.tags.items())
    except Exception as exc:
        technical["mutagen_error"] = str(exc)

    # --- WAV: RIFF chunk walk ---
    wav = parse_wav_chunks(path) if path.suffix.lower() == ".wav" else {}
    if wav:
        technical["wav_chunks"] = wav["chunk_ids"]
        if wav["info"]:
            technical["wav_info_tags"] = wav["info"]
            tag_text += "\n" + "\n".join(f"{k}={v}" for k, v in wav["info"].items())
        for cid, label in PRO_WAV_CHUNKS.items():
            if cid.decode() in wav["chunk_ids"]:
                findings.append(
                    {"source": "wav_chunk", "kind": "pro_audio", "detail": label}
                )

    # --- explicit tags first (strong evidence), then raw byte scan (weaker) ---
    seen: set[str] = set()

    def scan(blob: bytes, source: str) -> None:
        for pattern, name, kind in SIGNATURES:
            if name not in seen and re.search(pattern, blob, re.IGNORECASE):
                seen.add(name)
                findings.append({"source": source, "kind": kind, "detail": name})

    scan(tag_text.encode("utf-8", "ignore"), "tags")
    with open(path, "rb") as f:
        head = f.read(256 * 1024)
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - 64 * 1024))
        tail = f.read()
    scan(head + tail, "raw_bytes")

    if not tag_text.strip() and not wav.get("info"):
        findings.append(
            {"source": "tags", "kind": "info", "detail": "No tags (never tagged, or stripped)"}
        )

    editors = [x for x in findings if x["kind"] == "editor"]
    converters = [x for x in findings if x["kind"] == "converter"]
    if editors:
        confidence = "high" if any(x["source"] == "tags" for x in editors) else "medium"
    elif converters or any(x["kind"] == "pro_audio" for x in findings):
        confidence = "low"
    else:
        confidence = "none"

    return {
        "potentially_edited": bool(editors),
        "edit_confidence": confidence,
        "re_encoded_suspected": bool(converters),
        "findings": findings,
        "technical": technical,
        "note": (
            "Metadata can be stripped or forged. 'none' means no editing traces "
            "were found, not proof the file is untouched."
        ),
    }


# -------------------------------------- FEATURE 3: 2-second window analysis ---
def split_windows(y: np.ndarray) -> list[tuple[int, np.ndarray]]:
    step = int(SR * CHUNK_SECONDS)
    min_len = int(SR * MIN_CHUNK_SECONDS)
    windows: list[tuple[int, np.ndarray]] = []
    for start in range(0, len(y), step):
        seg = y[start : start + step]
        if len(seg) < min_len and windows:  # drop tiny tail, but never drop everything
            continue
        windows.append((start, seg))
    return windows


def analyze_audio_in_chunks(path: Path) -> list[dict]:
    y, _ = librosa.load(str(path), sr=SR, mono=True, duration=MAX_AUDIO_SECONDS)
    if len(y) == 0:
        raise ValueError("Audio contains no samples")

    timeline = []
    for idx, (start, seg) in enumerate(split_windows(y), start=1):
        t0, t1 = start / SR, (start + len(seg)) / SR
        entry = {
            "chunk_id": idx,
            "start_seconds": round(t0, 2),
            "end_seconds": round(t1, 2),
            "time_window": f"{t0:.2f}s - {t1:.2f}s",
        }
        if float(np.sqrt(np.mean(seg**2))) < SILENCE_RMS:
            entry.update(fake_probability=None, verdict="Silence")
        else:
            p = detector.score(seg.astype(np.float32))
            entry.update(
                fake_probability=round(p, 4),
                verdict="Synthetic/Clone" if p > FAKE_THRESHOLD else "Authentic",
            )
        timeline.append(entry)
    return timeline


def aggregate(timeline: list[dict]) -> dict:
    scored = [c for c in timeline if c["fake_probability"] is not None]
    if not scored:
        return {"overall_verdict": "No speech detected", "average_confidence_score": 0.0,
                "max_chunk_score": 0.0, "fake_chunk_ratio": 0.0}
    w = [c["end_seconds"] - c["start_seconds"] for c in scored]
    avg = sum(c["fake_probability"] * wi for c, wi in zip(scored, w)) / sum(w)
    fake_ratio = sum(c["verdict"] == "Synthetic/Clone" for c in scored) / len(scored)
    return {
        "overall_verdict": (
            "High Risk: Deepfake Detected" if avg > FAKE_THRESHOLD else "Low Risk: Safe"
        ),
        "average_confidence_score": round(avg, 4),
        "max_chunk_score": round(max(c["fake_probability"] for c in scored), 4),
        "fake_chunk_ratio": round(fake_ratio, 4),
    }


# ------------------------------------------------------------- endpoints ---
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": detector is not None}


@app.post("/api/v2/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    display_name = safe_display_name(file.filename)
    path, size = await save_unique_audio(file)
    try:
        # CPU/GPU-heavy work runs off the event loop so other requests
        # (and live calls) are not blocked.
        metadata = await asyncio.to_thread(inspect_audio_metadata, path)
        try:
            timeline = await asyncio.to_thread(analyze_audio_in_chunks, path)
        except Exception as exc:
            raise HTTPException(422, f"Could not decode audio: {exc}")

        return {
            "status": "success",
            "original_filename": display_name,
            "filename_assigned": path.name,
            "size_bytes": size,
            "metadata_inspection": metadata,
            **aggregate(timeline),
            "chunk_seconds": CHUNK_SECONDS,
            "timeline_analysis": timeline,
        }
    finally:
        path.unlink(missing_ok=True)


# ------------------------------------------- FEATURE 4: real-time call ---
@app.websocket("/ws/realtime-call")
async def websocket_realtime_call(ws: WebSocket):
    await ws.accept()
    log.info("Live call connected")

    window = int(SR * LIVE_WINDOW_SECONDS)
    hop = int(SR * LIVE_HOP_SECONDS)
    warmup = int(SR * LIVE_MIN_SECONDS)

    in_sr = SR
    buf = np.zeros(0, dtype=np.float32)
    new_samples = 0
    recent: deque[float] = deque(maxlen=LIVE_SMOOTH_N)
    consecutive = 0

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            # optional config frame
            if msg.get("text") is not None:
                try:
                    in_sr = int(json.loads(msg["text"]).get("sample_rate", SR))
                    if not 8000 <= in_sr <= 96000:
                        raise ValueError
                except Exception:
                    await ws.send_json({"status": "error", "detail": "bad config"})
                    continue
                await ws.send_json({"status": "configured", "sample_rate": in_sr})
                continue

            data = msg.get("bytes")
            if not data or len(data) < 2:
                continue

            # int16 PCM -> float32 [-1, 1], resample to 16 kHz if needed
            pcm = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2")
            pcm = pcm.astype(np.float32) / 32768.0
            if in_sr != SR:
                g = math.gcd(in_sr, SR)
                pcm = resample_poly(pcm, SR // g, in_sr // g).astype(np.float32)

            buf = np.concatenate([buf, pcm])[-window:]  # rolling 2 s context
            new_samples += len(pcm)
            if len(buf) < warmup or new_samples < hop:
                continue
            new_samples = 0
            snap = buf.copy()

            if float(np.sqrt(np.mean(snap**2))) < SILENCE_RMS:
                await ws.send_json({"status": "silence", "alert": "Secure"})
                continue

            async with live_sem:
                score = await asyncio.to_thread(detector.score, snap)

            recent.append(score)
            smoothed = sum(recent) / len(recent)
            consecutive = consecutive + 1 if smoothed > LIVE_ALERT_THRESHOLD else 0
            is_threat = consecutive >= LIVE_ALERT_CONSECUTIVE

            await ws.send_json({
                "status": "analyzed",
                "live_threat_score": round(score, 4),
                "smoothed_score": round(smoothed, 4),
                "window_seconds": round(len(snap) / SR, 2),
                "alert": (
                    "⚠️ WARNING: Synthetic Voice Signature Detected!"
                    if is_threat else "Secure"
                ),
            })
    except WebSocketDisconnect:
        pass
    finally:
        log.info("Live call disconnected")