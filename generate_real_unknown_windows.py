from pathlib import Path
import pandas as pd
import numpy as np
import wave

SR = 16000
WINDOW = 16000

PROJECT = Path(".")
SOURCE_MANIFEST = PROJECT / "dataset" / "processed" / "authoritative_dataset_manifest.csv"
OUTPUT_DIR = PROJECT / "dataset" / "windowed"
MANIFEST_FILE = PROJECT / "dataset" / "processed" / "real_unknown_windowed_manifest.csv"

if MANIFEST_FILE.exists():
    raise RuntimeError(
        f"STOP: {MANIFEST_FILE} already exists. "
        "Refusing to overwrite it."
    )

df = pd.read_csv(SOURCE_MANIFEST)

unknown = df[
    (df["class"] == "unknown") &
    (df["source_type"] == "real")
].copy()

if len(unknown) != 173:
    raise RuntimeError(
        f"Expected 173 real unknowns, found {len(unknown)}"
    )

records = []

def read_wav(path):
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.getnframes()
        raw = wf.readframes(frames)

    if channels != 1:
        raise RuntimeError(f"{path}: expected mono, found {channels}")

    if sample_width != 2:
        raise RuntimeError(
            f"{path}: expected 16-bit PCM, found {sample_width * 8}-bit"
        )

    if sample_rate != SR:
        raise RuntimeError(
            f"{path}: expected {SR} Hz, found {sample_rate} Hz"
        )

    audio = np.frombuffer(raw, dtype="<i2").copy()

    if len(audio) != frames:
        raise RuntimeError(f"{path}: frame count mismatch")

    return audio

def write_wav(path, audio):
    path.parent.mkdir(parents=True, exist_ok=True)

    audio = np.asarray(audio, dtype="<i2")

    if len(audio) != WINDOW:
        raise RuntimeError(
            f"{path}: refusing to write {len(audio)} samples"
        )

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(audio.tobytes())

for _, row in unknown.iterrows():

    split = str(row["split"])
    source_filename = str(row["filename"])
    source_path = Path(row["full_path"])

    if not source_path.exists():
        raise FileNotFoundError(
            f"Source WAV not found: {source_path}"
        )

    audio = read_wav(source_path)
    original_samples = len(audio)

    if original_samples > WINDOW:
        raise RuntimeError(
            f"{source_filename}: source has {original_samples} samples, "
            "but real unknown policy does not allow cropping."
        )

    if original_samples < WINDOW:
        padded = np.zeros(WINDOW, dtype="<i2")
        padded[:original_samples] = audio
        window_audio = padded
        padding_samples = WINDOW - original_samples
        strategy = "full_recording_trailing_zero_padding"
    else:
        window_audio = audio
        padding_samples = 0
        strategy = "full_recording_unchanged"

    output_filename = Path(source_filename).name
    output_path = OUTPUT_DIR / split / "unknown" / output_filename

    write_wav(output_path, window_audio)

    records.append({
        "split": split,
        "class": "unknown",
        "source_type": "real",
        "source_filename": source_filename,
        "source_path": str(source_path).replace("\\", "/"),
        "window_filename": output_filename,
        "window_path": str(output_path).replace("\\", "/"),
        "window_start_sample": 0,
        "window_end_sample": WINDOW,
        "window_start_sec": 0.0,
        "window_end_sec": 1.0,
        "window_duration_sec": 1.0,
        "original_sample_count": original_samples,
        "original_duration_sec": original_samples / SR,
        "padding_samples": padding_samples,
        "padding_duration_sec": padding_samples / SR,
        "window_strategy": strategy,
    })

manifest = pd.DataFrame(records)

if len(manifest) != 173:
    raise RuntimeError(
        f"Expected 173 generated windows, got {len(manifest)}"
    )

if manifest["window_duration_sec"].ne(1.0).any():
    raise RuntimeError("Not all generated windows are exactly 1 second.")

manifest.to_csv(MANIFEST_FILE, index=False)

print("REAL UNKNOWN WINDOW GENERATION COMPLETE")
print()
print("SOURCE RECORDINGS:", len(unknown))
print("WINDOWS GENERATED:", len(manifest))
print()
print("BY SPLIT:")
print(manifest.groupby("split").size())
print()
print("BY STRATEGY:")
print(manifest.groupby("window_strategy").size())
print()
print("TOTAL PADDING SAMPLES:", int(manifest["padding_samples"].sum()))
print("FILES WITH PADDING:", int((manifest["padding_samples"] > 0).sum()))
print()
print("MANIFEST:", MANIFEST_FILE)
print("OUTPUT:", OUTPUT_DIR)
