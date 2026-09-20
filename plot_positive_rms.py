from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt

SR = 16000
WINDOW = 16000
STEP = 4000

SOURCE_DIR = Path("dataset/processed/positive")
META_FILE = Path("dataset/processed/positive_endpoint_with_split.csv")
OUTPUT = Path("dataset/features/test_positive_rms_profiles.png")

df = pd.read_csv(META_FILE)
df = df[df["Split"] == "test"].reset_index(drop=True)

fig, axes = plt.subplots(13, 2, figsize=(14, 26))
axes = axes.ravel()

for i, row in df.iterrows():
    path = SOURCE_DIR / row["Filename"]
    audio, sr = sf.read(path, dtype="float32")

    if sr != SR:
        raise RuntimeError(f"{path}: expected {SR} Hz, got {sr}")

    starts = []
    rms_values = []

    for start in range(0, len(audio) - WINDOW + 1, STEP):
        window = audio[start:start + WINDOW]
        rms = float(np.sqrt(np.mean(window ** 2)))

        starts.append(start / SR)
        rms_values.append(rms)

    axes[i].plot(starts, rms_values)
    axes[i].set_title(row["Filename"], fontsize=8)
    axes[i].set_xlabel("1-second window start (s)")
    axes[i].set_ylabel("RMS")
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT, dpi=150)
plt.close()

print("SAVED:", OUTPUT)