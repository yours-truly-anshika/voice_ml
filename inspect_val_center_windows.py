from pathlib import Path
import pandas as pd
import soundfile as sf

SR = 16000
WINDOW = 16000

SOURCE_DIR = Path("dataset/processed/positive")
OUTPUT_DIR = Path("dataset/features/val_center_windows")

META_FILE = Path("dataset/processed/positive_endpoint_with_split.csv")

df = pd.read_csv(META_FILE)
df = df[df["Split"] == "val"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for _, row in df.iterrows():
    filename = row["Filename"]
    source_path = SOURCE_DIR / filename

    audio, sr = sf.read(source_path, dtype="float32")

    if sr != SR:
        raise RuntimeError(f"{filename}: expected {SR} Hz, got {sr}")

    if audio.ndim != 1:
        raise RuntimeError(f"{filename}: expected mono audio")

    source_samples = len(audio)

    onset = round(float(row["SpeechOnsetSec"]) * SR)
    offset = round(float(row["SpeechOffsetSec"]) * SR)

    midpoint = (onset + offset) / 2

    start = max(
        0,
        min(
            round(midpoint - WINDOW / 2),
            source_samples - WINDOW,
        ),
    )

    window = audio[start:start + WINDOW]

    if len(window) != WINDOW:
        raise RuntimeError(
            f"{filename}: got {len(window)} samples, expected {WINDOW}"
        )

    output = OUTPUT_DIR / filename.replace(".wav", "_center1s.wav")
    sf.write(output, window, SR)

    print(
        filename,
        "| center start:", f"{start / SR:.2f}s",
        "| saved:", output
    )

print()
print("GENERATED:", len(df))
print("OUTPUT:", OUTPUT_DIR)