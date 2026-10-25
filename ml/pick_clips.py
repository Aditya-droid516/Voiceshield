import glob, shutil
from pathlib import Path
import pandas as pd
import soundfile as sf

split_csv = glob.glob("**/splits.csv", recursive=True)[0]
d = pd.read_csv(split_csv)
d = d[d["split"] == "test"]

meta = [p for p in Path(".").rglob("meta.csv")]
root = meta[0].parent if meta else Path(".")
print("dataset root:", root)
index = {p.name: p for p in root.rglob("*.wav")}
print("wav files found:", len(index))

out = Path("test_clips")
out.mkdir(exist_ok=True)

def pick(y, tag, n=3):
    got, used = 0, set()
    for _, r in d[d["y"] == y].sample(frac=1, random_state=1).iterrows():
        p = index.get(r["file"])
        if p is None or r["speaker"] in used:
            continue
        try:
            dur = sf.info(str(p)).duration
        except Exception:
            continue
        if not (3.0 <= dur <= 10.0):
            continue
        used.add(r["speaker"])
        name = "%s_%d_%s.wav" % (tag, got + 1, r["speaker"].replace(" ", "_"))
        shutil.copy(p, out / name)
        print("copied", name, "%.1fs" % dur)
        got += 1
        if got == n:
            break

pick(0, "REAL")
pick(1, "FAKE")
print("done -> test_clips")
