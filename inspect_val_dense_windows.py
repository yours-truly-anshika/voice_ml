from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

SR = 16000
WINDOW = 16000
STEP = 800  # 50 ms

SOURCE_DIR = Path("dataset/processed/positive")
OUTPUT_DIR = Path("dataset/features/val_dense_windows")

META_FILE = Path("dataset/processed/positive_endpoint_with_split.csv")

df = pd.read_csv(META_FILE)
df = df[df["Split"] == "val"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for _, row in df.iterrows():
    filename = row["Filename"]
    audio, sr = sf.read(SOURCE_DIR / filename, dtype="float32")

    if sr != SR:
        raise RuntimeError(f"{filename}: expected {SR} Hz, got {sr}")

    if audio.ndim != 1:
        raise RuntimeError(f"{filename}: expected mono audio")

    candidates = []

    for start in range(0, len(audio) - WINDOW + 1, STEP):
        window = audio[start:start + WINDOW]
        rms = float(np.sqrt(np.mean(window ** 2)))
        candidates.append((rms, start))

    # Highest-RMS candidates, while requiring at least 200 ms
    # between selected windows so we get genuinely different views.
    candidates.sort(reverse=True)

    selected = []

    for rms, start in candidates:
        if all(abs(start - other_start) >= 3200 for _, other_start in selected):
            selected.append((rms, start))

        if len(selected) == 3:
            break

    for rank, (rms, start) in enumerate(selected, start=1):
        output = (
            OUTPUT_DIR
            / f"{Path(filename).stem}__dense{rank}.wav"
        )

        sf.write(
            output,
            audio[start:start + WINDOW],
            SR,
        )

        print(
            filename,
            f"| rank {rank}",
            f"| start {start / SR:.2f}s",
            f"| RMS {rms:.5f}",
            f"| {output}"
        )

print()
print("GENERATED CANDIDATES:", len(df) * 3)
print("OUTPUT:", OUTPUT_DIR)