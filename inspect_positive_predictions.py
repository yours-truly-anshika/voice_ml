import numpy as np
import tensorflow as tf
from pathlib import Path

d = np.load("dataset/features/dataset_features.npz")
model = tf.keras.models.load_model("models/dscnn_best.keras")

X = d["X_test"]
y = d["y_test"]

pred = model.predict(X, verbose=0)
files = sorted(Path("dataset/features/test/positive").glob("*.npy"))

classes = ["positive", "unknown", "background"]

print("FILE | P(positive) | P(unknown) | P(background) | PRED")
print("-" * 120)

for i, f in enumerate(files):
    predicted = classes[int(np.argmax(pred[i]))]
    print(
        f"{f.stem} | "
        f"{pred[i,0]:.4f} | "
        f"{pred[i,1]:.4f} | "
        f"{pred[i,2]:.4f} | "
        f"{predicted}"
    )
