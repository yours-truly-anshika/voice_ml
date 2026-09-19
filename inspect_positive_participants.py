import soundfile as sf
import numpy as np
from pathlib import Path

folder = Path("dataset/windowed/test/positive")
prefixes = [
    "2c574cb7-1d2b-440f-bf2f-6428bfe6bd07",
    "81fdb9e2-3612-4678-bf20-f0113528505c",
    "c7d753f1-f749-4a37-8bd5-53dc9bab4fea"
]

files = []
for prefix in prefixes:
    files.extend(sorted(folder.glob(prefix + "*.wav")))

print("FILE | RMS | PEAK")
print("-" * 100)

for path in files:
    x, sr = sf.read(path, dtype="float32")
    rms = np.sqrt(np.mean(x * x))
    peak = np.max(np.abs(x))
    print(f"{path.name} | {rms:.6f} | {peak:.6f}")
