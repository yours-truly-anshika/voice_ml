from pathlib import Path
import pandas as pd
import numpy as np
import wave

SR = 16000
SOURCE_SAMPLES = 48000
WINDOW = 16000

PROJECT = Path(".")
SOURCE_MANIFEST = (
    PROJECT
    / "dataset"
    / "synthetic"
    / "train"
    / "synthetic_unknown_manifest.csv"
)

SOURCE_DIR = PROJECT / "dataset" / "synthetic" / "train"
OUTPUT_DIR = PROJECT / "dataset" / "windowed" / "train" / "unknown"
MANIFEST_FILE = (
    PROJECT
    / "dataset"
    / "processed"
    / "synthetic_unknown_windowed_manifest.csv"
)

if MANIFEST_FILE.exists():
    raise RuntimeError(
        f"STOP: {MANIFEST_FILE} already exists. "
        "Refusing to overwrite it."
    )

df = pd.read_csv(SOURCE_MANIFEST)

if len(df) != 242:
    raise RuntimeError(
        f"Expected 242 synthetic unknowns, found {len(df)}"
    )

if df["variant"].value_counts().to_dict() != {
    "gain_noise": 121,
    "shift_reverb": 121,
}:
    raise RuntimeError("Unexpected synthetic variant counts.")

records = []


def read_wav(path):
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.getnframes()
        raw = wf.readframes(frames)

    if channels != 1:
        raise RuntimeError(
            f"{path}: expected mono, found {channels}"
        )

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


for _, row in df.iterrows():

    synthetic_filename = str(row["synthetic_filename"])
    source_filename = str(row["source_filename"])
    shift_ms = float(row["shift_ms"])

    source_path = PROJECT / "dataset" / "synthetic" / "train" / "unknown" / synthetic_filename

    if not source_path.exists():
        raise FileNotFoundError(
            f"Synthetic source not found: {source_path}"
        )

    audio = read_wav(source_path)

    if len(audio) != SOURCE_SAMPLES:
        raise RuntimeError(
            f"{synthetic_filename}: expected 48000 samples, "
            f"found {len(audio)}"
        )

    # Reconstruct the exact placement used by
    # generate_synthetic_unknown.py.
    source_length = int(
        wave.open(
            str(
                PROJECT
                / "dataset"
                / "final"
                / "train"
                / "unknown"
                / source_filename
            ),
            "rb",
        ).getnframes()
    )

    if source_length > WINDOW:
        raise RuntimeError(
            f"{source_filename}: source length exceeds 1-second window."
        )

    center_start = (SOURCE_SAMPLES - source_length) // 2
    shift_samples = int(SR * shift_ms / 1000)

    source_start = center_start + shift_samples
    source_start = max(
        0,
        min(
            source_start,
            SOURCE_SAMPLES - source_length,
        ),
    )

    source_end = source_start + source_length
    source_center = source_start + source_length / 2

    window_start = int(round(source_center - WINDOW / 2))
    window_start = max(
        0,
        min(
            window_start,
            SOURCE_SAMPLES - WINDOW,
        ),
    )

    window_end = window_start + WINDOW

    if window_end > SOURCE_SAMPLES:
        raise RuntimeError(
            f"{synthetic_filename}: calculated window exceeds source."
        )

    window_audio = audio[window_start:window_end]

    if len(window_audio) != WINDOW:
        raise RuntimeError(
            f"{synthetic_filename}: generated window has "
            f"{len(window_audio)} samples."
        )

    output_path = OUTPUT_DIR / synthetic_filename

    write_wav(output_path, window_audio)

    records.append({
        "split": "train",
        "class": "unknown",
        "source_type": "synthetic",
        "source_filename": source_filename,
        "source_path": str(source_path).replace("\\", "/"),
        "synthetic_filename": synthetic_filename,
        "window_filename": synthetic_filename,
        "window_path": str(output_path).replace("\\", "/"),

        "source_start_sample": source_start,
        "source_end_sample": source_end,
        "source_start_sec": source_start / SR,
        "source_end_sec": source_end / SR,
        "source_duration_sec": source_length / SR,

        "window_start_sample": window_start,
        "window_end_sample": window_end,
        "window_start_sec": window_start / SR,
        "window_end_sec": window_end / SR,
        "window_duration_sec": WINDOW / SR,

        "shift_ms": shift_ms,
        "gain_db": row["gain_db"],
        "snr_db": row["snr_db"],
        "variant": row["variant"],
        "reverb": row["reverb"],
        "seed": row["seed"],

        "window_strategy": (
            "centered_on_embedded_source"
        ),
    })


manifest = pd.DataFrame(records)

if len(manifest) != 242:
    raise RuntimeError(
        f"Expected 242 generated windows, got {len(manifest)}"
    )

if manifest["window_duration_sec"].ne(1.0).any():
    raise RuntimeError(
        "Not all generated windows are exactly 1 second."
    )

if (
    manifest["window_start_sample"] < 0
).any() or (
    manifest["window_end_sample"] > SOURCE_SAMPLES
).any():
    raise RuntimeError(
        "At least one generated window is out of bounds."
    )

manifest.to_csv(MANIFEST_FILE, index=False)

print("SYNTHETIC UNKNOWN WINDOW GENERATION COMPLETE")
print()
print("SOURCE RECORDINGS:", len(df))
print("WINDOWS GENERATED:", len(manifest))
print()
print("BY VARIANT:")
print(manifest.groupby("variant").size())
print()
print("WINDOW DURATION:")
print(manifest["window_duration_sec"].value_counts())
print()
print("WINDOW START SAMPLE:")
print(
    manifest["window_start_sample"]
    .describe()
    .round(3)
)
print()
print("WINDOW END SAMPLE:")
print(
    manifest["window_end_sample"]
    .describe()
    .round(3)
)
print()
print("MANIFEST:", MANIFEST_FILE)
print("OUTPUT:", OUTPUT_DIR)
