import argparse
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from models.dscnn import build_dscnn


SEED = 26172

MODEL_DIR = Path("models")

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

SHAPE_MAP = {
    1.0: (40, 49, 1),
    1.5: (40, 74, 1),
    2.0: (40, 99, 1),
}


def main():
    parser = argparse.ArgumentParser(description="Train the duration-specific DS-CNN")
    parser.add_argument(
        "--duration",
        type=float,
        required=True,
        choices=sorted(SHAPE_MAP),
        help="Input window duration in seconds",
    )
    args = parser.parse_args()

    duration = args.duration
    input_shape = SHAPE_MAP[duration]
    dataset_path = Path(f"dataset/features_{duration}s/dataset_features.npz")
    best_model_path = MODEL_DIR / f"dscnn_{duration}s_best.keras"

    print(f"DURATION: {duration}s")
    print("INPUT SHAPE:", input_shape)
    print("DATASET:", dataset_path)
    print("MODEL OUTPUT:", best_model_path)

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    data = np.load(dataset_path)

    X_train = data["X_train"]
    y_train = data["y_train"]

    X_val = data["X_val"]
    y_val = data["y_val"]

    print("TRAIN:", X_train.shape, y_train.shape)
    print("VAL:  ", X_val.shape, y_val.shape)

    model = build_dscnn(input_shape=input_shape)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            best_model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    best_epoch = int(np.argmax(history.history["val_accuracy"])) + 1
    best_val_accuracy = float(
        np.max(history.history["val_accuracy"])
    )

    print()
    print("TRAINING COMPLETE")
    print("BEST EPOCH:", best_epoch)
    print("BEST VAL ACCURACY:", best_val_accuracy)
    print("BEST MODEL:", best_model_path)


if __name__ == "__main__":
    main()
