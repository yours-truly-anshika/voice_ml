import pandas as pd
import numpy as np
import soundfile as sf
from pathlib import Path

manifest = pd.read_csv("dataset/processed/positive_windowed_manifest.csv")
test = manifest[manifest["split"] == "test"].copy()

rows = []

for _, r in test.iterrows():
    audio, sr = sf.read(r["source_path"], always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)

    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))

    rows.append({
        "file": r["source_filename"],
        "participant": r["source_filename"].split("_")[0],
        "duration": r["source_duration_sec"],
        "onset": r["speech_onset_sec"],
        "offset": r["speech_offset_sec"],
        "speech_span": r["speech_span_sec"],
        "rms": rms,
        "peak": peak,
    })

out = pd.DataFrame(rows)
print(out.to_string(index=False))
