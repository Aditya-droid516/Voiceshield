from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
from pathlib import Path
import hashlib
from datetime import datetime, timezone

from ml.inference import VoiceShieldInference

app = FastAPI(
    title="VoiceShield API",
    description="Enterprise AI Voice-Clone Detection & Forensic Backend",
    version="2.0"
)

inference_engine = VoiceShieldInference()

@app.get("/")
def read_root():
    return {
        "status": "online",
        "system": "VoiceShield Forensic Engine",
        "message": "Visit /docs for interactive API documentation."
    }

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    allowed_extensions = (".wav", ".mp3", ".flac", ".m4a", ".webm", ".ogg")
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file format. Allowed formats: {', '.join(allowed_extensions)}"
        )
    
    temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / file.filename
    
    try:
        # Save file and calculate cryptographic SHA-256 hash for forensics
        sha256_hash = hashlib.sha256()
        with open(temp_path, "wb") as buffer:
            while chunk := await file.read(8192):
                sha256_hash.update(chunk)
                buffer.write(chunk)
                
        file_hash = sha256_hash.hexdigest()
        
        # Run ML inference
        result = inference_engine.predict(str(temp_path))
        
        # Build forensic compliance certificate block
        timestamp = datetime.now(timezone.utc).isoformat()
        
        forensic_certificate = {
            "certificate_id": f"VS-AUDIT-{file_hash[:8].upper()}",
            "timestamp_utc": timestamp,
            "filename": file.filename,
            "sha256_hash": file_hash,
            "model_architecture": "Microsoft WavLM-Base + Neural Classifier",
            "verdict": result.get("prediction", "Unknown"),
            "confidence_score": result.get("confidence", 0.0),
            "compliance_status": "Audited & Verified"
        }
        
        return {
            "filename": file.filename,
            "prediction": result.get("prediction", "Unknown"),
            "confidence": result.get("confidence", 0.0),
            "details": result.get("details", {}),
            "forensic_certificate": forensic_certificate
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio stream: {str(e)}")
        
    finally:
        if temp_path.exists():
            temp_path.unlink()