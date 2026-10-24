import glob
import pandas as pd, soundfile as sf
from hm.combined import VoiceShieldInference as V
from ml.dataset import DATA_DIR
v = V()
def show(tag, path):
    try:
        o = v.predict(path)
        d = o["details"]
        print("%-8s hm=%.3f wavlm=%.3f -> %.3f %s" % (tag, d["handmade_score"], d["wavlm_score"], o["fake_probability"], o["risk_level"]))
    except Exception as e:
        print(tag, "skipped:", str(e)[:70])
t = pd.read_csv("data/processed/splits.csv")
t = t[t.split == "test"]
for y, tag in ((0, "real"), (1, "fake")):
    n = 0
    for r in t[t.y == y].sample(frac=1, random_state=4).itertuples():
        p = str(DATA_DIR / r.file)
        if sf.info(p).duration >= 3:
            show(tag, p); n += 1
        if n == 3: break
for p in sorted(glob.glob("data/mic/real/*.wav"))[:5]:
    show("my mic", p)
