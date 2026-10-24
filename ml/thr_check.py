import numpy as np
from ml.train import load_checkpoint, predict_scores
m, mean, std, info = load_checkpoint()
for name in ("dev", "test"):
    d = np.load("data/processed/%s.npz" % name, allow_pickle=True)
    s = predict_scores(m, mean, std, d["X"]); y = d["y"].astype(int)
    print("\n" + name.upper(), "| real score pcts 50/90/99:", np.round(np.percentile(s[y == 0], [50, 90, 99]), 3), "| fake pcts 1/10/50:", np.round(np.percentile(s[y == 1], [1, 10, 50]), 3))
    print("cutoff  real_flagged%  fake_caught%")
    for t in (0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99):
        print("%.2f    %6.1f        %6.1f" % (t, 100 * (s[y == 0] >= t).mean(), 100 * (s[y == 1] >= t).mean()))
