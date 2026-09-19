import pandas as pd
import soundfile as sf
import numpy as np

m = pd.read_csv(r"dataset\processed\positive_windowed_manifest.csv")
m = m[(m["split"] == "test") & (m["window_view"] == "completion")]

print("FILE | FULL_RMS | WINDOW_RMS | RATIO")
print("-" * 75)

for _, r in m.iterrows():
    full, sr = sf.read(r["source_path"])
    window, _ = sf.read(r["window_path"])

    full_rms = np.sqrt(np.mean(full ** 2))
    window_rms = np.sqrt(np.mean(window ** 2))
    ratio = window_rms / max(full_rms, 1e-12)

    print(
        f"{r['window_filename']} | "
        f"{full_rms:.5f} | "
        f"{window_rms:.5f} | "
        f"{ratio:.3f}"
    )
