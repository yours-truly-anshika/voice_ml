from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

SR = 16000
WINDOW = 16000
STEP = 4000

SOURCE_DIR = Path("dataset/processed/positive")
MANIFEST = Path("dataset/processed/positive_windowed_manifest.csv")

df = pd.read_csv(MANIFEST)
df = df[df["split"] == "val"]

print("FILE | COMPLETION_RMS | BEST_RMS | RATIO")
print("-" * 110)

for _, row in df.iterrows():
    path = SOURCE_DIR / row["source_filename"]
    audio, sr = sf.read(path, dtype="float32")

    start = int(round(row["window_start_sec"] * SR))
    completion = audio[start:start + WINDOW]
    completion_rms = float(np.sqrt(np.mean(completion ** 2)))

    best_rms = 0.0

    for s in range(0, len(audio) - WINDOW + 1, STEP):
        window = audio[s:s + WINDOW]
        rms = float(np.sqrt(np.mean(window ** 2)))
        best_rms = max(best_rms, rms)

    ratio = completion_rms / best_rms if best_rms > 0 else 0.0

    print(
        f"{row['source_filename']} | "
        f"{completion_rms:.5f} | "
        f"{best_rms:.5f} | "
        f"{ratio:.3f}"
    )