from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


FEATURES = Path("dataset/features/dataset_features.npz")
MODEL = Path("models/dscnn_best.keras")

CLASS_NAMES = ["positive", "unknown", "background"]
POSITIVE_INDEX = 0

THRESHOLDS = [0.50, 0.70, 0.80, 0.90, 0.95]


def longest_run(values):
    best = 0
    current = 0

    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0

    return best


def main():
    data = np.load(FEATURES)

    model = tf.keras.models.load_model(MODEL)

    X_val = data["X_val"]
    y_val = data["y_val"]

    X_test = data["X_test"]
    y_test = data["y_test"]

    for split_name, X, y in [
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        probabilities = model.predict(
            X,
            batch_size=32,
            verbose=0,
        )

        predicted = np.argmax(probabilities, axis=1)

        print(f"\n{'=' * 70}")
        print(f"{split_name.upper()}")
        print(f"{'=' * 70}")

        for class_index, class_name in enumerate(CLASS_NAMES):
            mask = y == class_index

            print(
                f"\n{class_name.upper()}: "
                f"{int(mask.sum())} windows"
            )

            positive_conf = probabilities[
                mask,
                POSITIVE_INDEX,
            ]

            print(
                f"Positive confidence: "
                f"mean={positive_conf.mean():.4f}, "
                f"median={np.median(positive_conf):.4f}, "
                f"max={positive_conf.max():.4f}"
            )

            for threshold in THRESHOLDS:
                count = int(
                    np.sum(
                        positive_conf >= threshold
                    )
                )

                print(
                    f"  >= {threshold:.2f}: "
                    f"{count}/{len(positive_conf)}"
                )

        print("\nFALSE POSITIVE DETAILS:")

        for class_index, class_name in [
            (1, "unknown"),
            (2, "background"),
        ]:
            mask = y == class_index

            indices = np.where(
                mask
                & (
                    predicted
                    == POSITIVE_INDEX
                )
            )[0]

            if len(indices) == 0:
                print(
                    f"\n{class_name}: "
                    f"no positive predictions"
                )
                continue

            print(
                f"\n{class_name}: "
                f"{len(indices)} positive predictions"
            )

            for idx in indices:
                print(
                    f"  index={idx:3d} "
                    f"positive_conf="
                    f"{probabilities[idx, POSITIVE_INDEX]:.4f} "
                    f"predicted="
                    f"{CLASS_NAMES[predicted[idx]]}"
                )

        print("\nNOTE:")
        print(
            "The val/test negative windows are independent "
            "1-second examples, so this script measures "
            "single-window false-positive behavior only. "
            "It does not claim they form a real temporal stream."
        )


if __name__ == "__main__":
    main()