import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

files = ["0149", "0152", "0157", "0160", "0166"]

fig, axes = plt.subplots(5, 1, figsize=(10, 12))

for i, f in enumerate(files):
    feature = np.load(
        Path("dataset/features/test/unknown") / (f + ".npy")
    )[:, :, 0]

    axes[i].imshow(
        feature,
        origin="lower",
        aspect="auto"
    )
    axes[i].set_title("False-positive unknown: " + f)
    axes[i].set_ylabel("Mel bins")

axes[-1].set_xlabel("Time frames")

plt.tight_layout()
plt.savefig(
    "dataset/features/false_positive_unknowns.png",
    dpi=150
)

print("Saved: dataset/features/false_positive_unknowns.png")
