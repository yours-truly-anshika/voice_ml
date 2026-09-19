import numpy as np
import tensorflow as tf
from pathlib import Path


DATASET = "dataset/features/dataset_features.npz"
MODEL = "models/dscnn_best.keras"

CLASS_NAMES = ["positive", "unknown", "background"]


data = np.load(DATASET)
model = tf.keras.models.load_model(MODEL)

X_test = data["X_test"]
y_test = data["y_test"]

predictions = model.predict(X_test, verbose=0)
predicted = np.argmax(predictions, axis=1)

names = []

for class_name in CLASS_NAMES:
    directory = Path("dataset/features/test") / class_name
    names.extend(
        [path.stem for path in sorted(directory.glob("*.npy"))]
    )


print("INDEX | ACTUAL     | PREDICTED  | CONFIDENCE | FILE")
print("-" * 80)

for i in range(len(y_test)):
    if y_test[i] != predicted[i]:
        confidence = float(np.max(predictions[i]))

        print(
            f"{i:5d} | "
            f"{CLASS_NAMES[y_test[i]]:10s} | "
            f"{CLASS_NAMES[predicted[i]]:10s} | "
            f"{confidence:.4f}     | "
            f"{names[i]}"
        )
