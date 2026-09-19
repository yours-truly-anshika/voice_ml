import soundfile as sf
import numpy as np
from pathlib import Path

cases = [
    ("FP unknown", "dataset/windowed/test/unknown", ["0149.wav", "0152.wav", "0157.wav", "0160.wav", "0166.wav"]),
    ("FN positive", "dataset/windowed/test/positive", [
        "2c574cb7-1d2b-440f-bf2f-6428bfe6bd07_optional_7__completion.wav",
        "81fdb9e2-3612-4678-bf20-f0113528505c_mandatory_1__completion.wav",
        "81fdb9e2-3612-4678-bf20-f0113528505c_mandatory_2__completion.wav",
        "c7d753f1-f749-4a37-8bd5-53dc9bab4fea_mandatory_1__completion.wav"
    ])
]

print("TYPE | FILE | RMS | PEAK")
print("-" * 90)

for label, folder, files in cases:
    for filename in files:
        x, sr = sf.read(Path(folder) / filename, dtype="float32")
        rms = np.sqrt(np.mean(x * x))
        peak = np.max(np.abs(x))
        print(f"{label} | {filename} | {rms:.6f} | {peak:.6f}")
