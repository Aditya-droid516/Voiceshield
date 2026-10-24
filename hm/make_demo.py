import shutil
from pathlib import Path
import pandas as pd, soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hm.combined import VoiceShieldInference as V
from ml.dataset import DATA_DIR

v = V()
t = pd.read_csv("data/processed/splits.csv")
t = t[t.split == "test"]
picked = {}
print("%-6s %-22s %6s %7s %7s  %s" % ("truth", "speaker", "hand", "wavlm", "blend", "verdict"))
for y, tag in ((0, "real"), (1, "fake")):
    out = Path("data/demo") / tag
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for r in t[t.y == y].sample(frac=1, random_state=11).itertuples():
        src = DATA_DIR / r.file
        if 4 <= sf.info(str(src)).duration <= 10:
            dst = out / ("%s_%d.wav" % (tag, n + 1))
            shutil.copy(src, dst)
            o = v.predict(str(dst)); d = o["details"]
            print("%-6s %-22s %6.3f %7.3f %7.3f  %s" % (tag, r.speaker[:22], d["handmade_score"], d["wavlm_score"], o["fake_probability"], o["risk_level"]))
            picked.setdefault(tag, str(dst))
            n += 1
        if n == 3:
            break
fig, ax = plt.subplots(figsize=(7, 3.5))
for tag, color in (("real", "tab:green"), ("fake", "tab:red")):
    tl = v.predict_timeline(picked[tag])
    ax.plot(tl["times"], tl["scores"], marker="o", color=color, label=tag + " clip")
ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
ax.set_ylim(-0.05, 1.05); ax.set_xlabel("seconds into clip"); ax.set_ylabel("fake score")
ax.set_title("VoiceShield score over time (update every 0.5 s)"); ax.legend()
fig.tight_layout(); fig.savefig("demo_timeline.png", dpi=150)
print("saved demo_timeline.png")
