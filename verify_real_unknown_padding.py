import pandas as pd
import numpy as np
import wave

manifest = pd.read_csv("dataset/processed/real_unknown_windowed_manifest.csv")

errors = []

padded = manifest[manifest["padding_samples"] > 0]

for _, row in padded.iterrows():

    with wave.open(row["source_path"], "rb") as wf:
        source = np.frombuffer(
            wf.readframes(wf.getnframes()),
            dtype="<i2"
        ).copy()

    with wave.open(row["window_path"], "rb") as wf:
        window = np.frombuffer(
            wf.readframes(wf.getnframes()),
            dtype="<i2"
        ).copy()

    original_samples = int(row["original_sample_count"])

    if not np.array_equal(source, window[:original_samples]):
        errors.append(
            (row["source_filename"], "original_audio_mismatch")
        )

    if np.any(window[original_samples:] != 0):
        errors.append(
            (row["source_filename"], "padding_not_zero")
        )

print("PADDED FILES CHECKED:", len(padded))
print("ERRORS:", len(errors))
print(errors[:10])
