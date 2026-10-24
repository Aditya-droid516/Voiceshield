import pandas as pd, soundfile as sf
from ml.inference import VoiceShieldInference as V
from ml.dataset import DATA_DIR
d = pd.read_csv("data/processed/splits.csv")
t = d[d.split == "test"]
v = V()
for y in (1, 0):
    for r in t[t.y == y].sample(frac=1, random_state=3).itertuples():
        if sf.info(str(DATA_DIR / r.file)).duration >= 5:
            o = v.predict_timeline(str(DATA_DIR / r.file))
            print(r.label, r.speaker, "| windows", len(o["times"]))
            print(" scores", o["scores"])
            print(" alert at", o["alert_time"], "| ms per update", o["ms_per_update"])
            break
