import soundfile as sf
import numpy as np
from pathlib import Path

folder = Path("dataset/windowed/test/unknown")
files = ["0149.wav", "0152.wav", "0157.wav", "0160.wav", "0166.wav"]

print("FILE | SAMPLES | RMS | PEAK | CLIPPED")
print("-" * 65)

for filename in files:
    x, sr = sf.read(folder / filename, dtype="float32")
    rms = np.sqrt(np.mean(x * x))
    peak = np.max(np.abs(x))
    clipped = np.sum(np.abs(x) >= 0.999)

    print(
        f"{filename} | {len(x)} | "
        f"{rms:.6f} | {peak:.6f} | {clipped}"
    )
