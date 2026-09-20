from pathlib import Path
import numpy as np
import soundfile as sf
import pandas as pd

SR = 16000
WINDOW = 16000
STEP = 4000

SOURCE_DIR = Path("dataset/processed/positive")
META_FILE = Path("dataset/processed/positive_endpoint_with_split.csv")

df = pd.read_csv(META_FILE)
df = df[df["Split"] == "test"]

print("FILE | BEST_WINDOW_START | BEST_WINDOW_RMS")
print("-" * 90)

for _, row in df.iterrows():
    path = SOURCE_DIR / row["Filename"]

    audio, sr = sf.read(path, dtype="float32")

    if sr != SR:
        raise RuntimeError(f"{path}: expected {SR} Hz, got {sr}")

    if audio.ndim != 1:
        raise RuntimeError(f"{path}: expected mono audio")

    best_start = 0
    best_rms = -1.0

    for start in range(0, len(audio) - WINDOW + 1, STEP):
        window = audio[start:start + WINDOW]
        rms = float(np.sqrt(np.mean(window ** 2)))

        if rms > best_rms:
            best_rms = rms
            best_start = start

    print(
        f"{row['Filename']} | "
        f"{best_start / SR:.3f}s | "
        f"{best_rms:.6f}"
    )