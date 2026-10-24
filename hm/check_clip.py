import sys
from pathlib import Path
from hm.combined import VoiceShieldInference as V

EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm"}
files = []
for a in sys.argv[1:]:
    p = Path(a)
    files += sorted(f for f in p.rglob("*") if f.suffix.lower() in EXT) if p.is_dir() else [p]
if not files:
    sys.exit("No audio files found. Usage: python -m hm.check_clip <file or folder> ...")
v = V()
print("%-34s %6s %7s %7s  %s" % ("file", "hand", "wavlm", "blend", "verdict"))
for f in files:
    try:
        o = v.predict(str(f)); d = o["details"]
        print("%-34s %6.3f %7.3f %7.3f  %s" % (f.name[:34], d["handmade_score"], d["wavlm_score"], o["fake_probability"], o["risk_level"]))
    except Exception as e:
        print("%-34s skipped: %s" % (f.name[:34], str(e)[:60]))
