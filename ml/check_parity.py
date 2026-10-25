import sys
import pandas as pd
from pathlib import Path
from ml.inference import VoiceShieldInference

# find the splits file
cands = list(Path(".").rglob("splits.csv"))
if not cands:
    sys.exit("splits.csv not found")
splits = pd.read_csv(cands[0])
print("splits file:", cands[0])
print("columns:", list(splits.columns))
print(splits.head(3).to_string())

def pick(names):
    for n in names:
        if n in splits.columns:
            return n
    sys.exit("none of %s in columns" % names)

split_col = pick(["split", "set", "subset"])
label_col = pick(["label", "y", "class"])
path_col = pick(["path", "file", "filepath", "filename"])

print("label values:", splits[label_col].unique())
real_vals = [v for v in splits[label_col].unique() if str(v).lower() in ("bona-fide", "bonafide", "real", "genuine", "0")]
print("treating as real:", real_vals)

test_real = splits[(splits[split_col] == "test") & (splits[label_col].isin(real_vals))]
paths = test_real.sample(min(30, len(test_real)), random_state=0)[path_col].tolist()

m = VoiceShieldInference()
scores = []
for p in paths:
    try:
        scores.append(m.predict(p)["fake_probability"])
    except Exception as e:
        print("skip", p, "->", e)

print()
print(sum(s > 0.5 for s in scores), "of", len(scores), "real test clips flagged fake")
print(sorted(round(s, 3) for s in scores))
