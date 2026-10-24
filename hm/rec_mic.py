import os
import numpy as np, sounddevice as sd, soundfile as sf
from hm.combined import VoiceShieldInference as V
SR = 16000
os.makedirs("data/mic/real", exist_ok=True)
v = V()
for i in range(3):
    input("Press Enter, then speak normally for 5 s (clip %d of 3)..." % (i + 1))
    a = sd.rec(5 * SR, samplerate=SR, channels=1, dtype="float32"); sd.wait()
    p = "data/mic/real/me_%d.wav" % i
    sf.write(p, a[:, 0], SR)
    o = v.predict(p); d = o["details"]
    print("peak %.2f  hm=%.3f wavlm=%.3f -> %.3f %s" % (float(np.abs(a).max()), d["handmade_score"], d["wavlm_score"], o["fake_probability"], o["risk_level"]))
