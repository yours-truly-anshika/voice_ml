from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

SR = 16000
WINDOW = 16000
STEP = 4000

SOURCE_DIR = Path("dataset/processed/positive")
OUTPUT_DIR = Path("dataset/features/val_best_windows")

df = pd.read_csv("dataset/processed/positive_windowed_manifest.csv")
df = df[df["split"] == "val"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for _, row in df.iterrows():
    filename = row["source_filename"]
    audio, sr = sf.read(SOURCE_DIR / filename, dtype="float32")

    best_start = 0
    best_rms = -1.0

    for start in range(0, len(audio) - WINDOW + 1, STEP):
        rms = float(np.sqrt(np.mean(audio[start:start + WINDOW] ** 2)))

        if rms > best_rms:
            best_rms = rms
            best_start = start

    output = OUTPUT_DIR / filename.replace(".wav", "_best1s.wav")
    sf.write(output, audio[best_start:best_start + WINDOW], SR)

    print(
        filename,
        "| best start:", f"{best_start / SR:.2f}s",
        "| RMS:", f"{best_rms:.5f}",
        "| saved:", output
    )
