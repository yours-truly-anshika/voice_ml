from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

SR = 16000
WINDOW = 16000

MANIFEST = Path("dataset/processed/positive_windowed_manifest.csv")
SOURCE_DIR = Path("dataset/processed/positive")

df = pd.read_csv(MANIFEST)
df = df[df["split"] == "train"]

rows = []

for _, row in df.iterrows():
    path = SOURCE_DIR / row["source_filename"]
    audio, sr = sf.read(path, dtype="float32")

    start = int(round(row["window_start_sec"] * SR))
    window = audio[start:start + WINDOW]

    rms = float(np.sqrt(np.mean(window ** 2)))

    rows.append({
        "file": row["source_filename"],
        "view": row["window_view"],
        "rms": rms,
    })

x = pd.DataFrame(rows)

p = x.pivot(index="file", columns="view", values="rms")
p["best"] = p[["early", "center", "completion"]].max(axis=1)

ratios = p[["early", "center", "completion"]].div(p["best"], axis=0)

print("RMS RATIO TO BEST TRAINING VIEW")
print()
print(ratios.describe().round(3))