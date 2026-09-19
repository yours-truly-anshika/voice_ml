from pathlib import Path
import pandas as pd
import numpy as np
import wave

SR = 16000
WINDOW = 16000
PRE_ONSET = 3200

PROJECT = Path(".")
SOURCE_DIR = PROJECT / "dataset" / "processed" / "positive"
META_FILE = PROJECT / "dataset" / "processed" / "positive_endpoint_with_split.csv"

OUTPUT_DIR = PROJECT / "dataset" / "windowed"
MANIFEST_FILE = PROJECT / "dataset" / "processed" / "positive_windowed_manifest.csv"

# Safety: do not overwrite an existing generated dataset.
if OUTPUT_DIR.exists():
    existing = list(OUTPUT_DIR.rglob("*"))
    if existing:
        raise RuntimeError(
            f"STOP: {OUTPUT_DIR} already exists and is not empty. "
            "Delete/rename it manually only if you intentionally want a fresh generation."
        )

if MANIFEST_FILE.exists():
    raise RuntimeError(
        f"STOP: {MANIFEST_FILE} already exists. "
        "This script will not overwrite an existing manifest."
    )

df = pd.read_csv(META_FILE)

required_columns = {
    "Split",
    "Filename",
    "DurationSec",
    "SpeechOnsetSec",
    "SpeechOffsetSec",
    "SpeechSpanSec",
    "LeadingSilenceSec",
    "TrailingSilenceSec",
    "Detected",
}

missing = required_columns - set(df.columns)
if missing:
    raise RuntimeError(f"Missing metadata columns: {sorted(missing)}")

if len(df) != 173:
    raise RuntimeError(f"Expected 173 positive sources, found {len(df)}")

records = []

def read_wav(path):
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if channels != 1:
        raise RuntimeError(f"{path}: expected mono, found {channels} channels")

    if sample_width != 2:
        raise RuntimeError(f"{path}: expected 16-bit PCM, found {sample_width * 8}-bit")

    if sample_rate != SR:
        raise RuntimeError(f"{path}: expected {SR} Hz, found {sample_rate} Hz")

    audio = np.frombuffer(raw, dtype="<i2").copy()

    if len(audio) != nframes:
        raise RuntimeError(f"{path}: frame count mismatch")

    return audio

def write_wav(path, audio):
    path.parent.mkdir(parents=True, exist_ok=True)

    audio = np.asarray(audio, dtype="<i2")

    if len(audio) != WINDOW:
        raise RuntimeError(
            f"{path}: refusing to write {len(audio)} samples; expected {WINDOW}"
        )

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(audio.tobytes())

for _, row in df.iterrows():

    split = str(row["Split"])
    source_filename = str(row["Filename"])
    source_path = SOURCE_DIR / source_filename

    if not source_path.exists():
        raise FileNotFoundError(f"Source WAV not found: {source_path}")

    audio = read_wav(source_path)
    source_samples = len(audio)
    source_duration = source_samples / SR

    onset = round(float(row["SpeechOnsetSec"]) * SR)
    offset = round(float(row["SpeechOffsetSec"]) * SR)

    if split == "train":

        midpoint = (onset + offset) / 2

        windows = [
            ("early", max(0, onset - PRE_ONSET)),
            (
                "center",
                max(
                    0,
                    min(
                        round(midpoint - WINDOW / 2),
                        source_samples - WINDOW,
                    ),
                ),
            ),
            (
                "completion",
                max(
                    0,
                    min(offset, source_samples) - WINDOW,
                ),
            ),
        ]

    else:

        windows = [
            (
                "completion",
                max(
                    0,
                    min(offset, source_samples) - WINDOW,
                ),
            )
        ]

    for window_index, (view, start) in enumerate(windows, start=1):

        end = start + WINDOW

        if start < 0 or end > source_samples:
            raise RuntimeError(
                f"{source_filename} / {view}: "
                f"invalid window [{start}, {end}) for {source_samples} samples"
            )

        window_audio = audio[start:end]

        if len(window_audio) != WINDOW:
            raise RuntimeError(
                f"{source_filename} / {view}: "
                f"got {len(window_audio)} samples"
            )

        output_filename = (
            f"{Path(source_filename).stem}__{view}.wav"
        )

        output_path = OUTPUT_DIR / split / "positive" / output_filename

        write_wav(output_path, window_audio)

        records.append({
            "split": split,
            "class": "positive",
            "source_filename": source_filename,
            "source_path": str(source_path).replace("\\", "/"),
            "window_filename": output_filename,
            "window_path": str(output_path).replace("\\", "/"),
            "window_view": view,
            "window_index": window_index,
            "window_start_sample": start,
            "window_end_sample": end,
            "window_start_sec": start / SR,
            "window_end_sec": end / SR,
            "window_duration_sec": WINDOW / SR,
            "source_sample_count": source_samples,
            "source_duration_sec": source_duration,
            "speech_onset_sec": float(row["SpeechOnsetSec"]),
            "speech_offset_sec": float(row["SpeechOffsetSec"]),
            "speech_span_sec": float(row["SpeechSpanSec"]),
            "leading_silence_sec": float(row["LeadingSilenceSec"]),
            "trailing_silence_sec": float(row["TrailingSilenceSec"]),
            "endpoint_detected": row["Detected"],
            "window_strategy": (
                "200ms_pre_onset"
                if view == "early"
                else "speech_midpoint"
                if view == "center"
                else "speech_offset_completion"
            ),
        })

manifest = pd.DataFrame(records)

if len(manifest) != 415:
    raise RuntimeError(f"Expected 415 windows, generated {len(manifest)}")

if manifest["window_duration_sec"].ne(1.0).any():
    raise RuntimeError("Not all windows are exactly 1 second")

manifest.to_csv(MANIFEST_FILE, index=False)

print("POSITIVE WINDOW GENERATION COMPLETE")
print()
print("SOURCE RECORDINGS:", len(df))
print("WINDOWS GENERATED:", len(manifest))
print()
print("BY SPLIT:")
print(manifest.groupby("split").size())
print()
print("BY VIEW:")
print(manifest.groupby("window_view").size())
print()
print("MANIFEST:", MANIFEST_FILE)
print("OUTPUT:", OUTPUT_DIR)
