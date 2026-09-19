import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

files = [
    "2c574cb7-1d2b-440f-bf2f-6428bfe6bd07_optional_7__completion",
    "81fdbf2e-3612-4678-bf20-f0113528505c_mandatory_1__completion",
    "81fdbf2e-3612-4678-bf20-f0113528505c_mandatory_2__completion",
    "c7d753f1-f749-4a37-8bd5-53dc9bab4fea_mandatory_1__completion"
]

folder = Path("dataset/features/test/positive")

fig, axes = plt.subplots(4, 1, figsize=(10, 10))

for i, f in enumerate(files):
    feature = np.load(folder / (f + ".npy"))[:, :, 0]

    axes[i].imshow(
        feature,
        origin="lower",
        aspect="auto"
    )
    axes[i].set_title("False-negative positive: " + f)
    axes[i].set_ylabel("Mel bins")

axes[-1].set_xlabel("Time frames")

plt.tight_layout()
plt.savefig(
    "dataset/features/false_negative_positives.png",
    dpi=150
)

print("Saved: dataset/features/false_negative_positives.png")
