import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from models.dscnn import build_dscnn


SEED = 26172

DATASET_PATH = Path("dataset/features/dataset_features.npz")
MODEL_DIR = Path("models")
BEST_MODEL_PATH = MODEL_DIR / "dscnn_best.keras"

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    data = np.load(DATASET_PATH)

    X_train = data["X_train"]
    y_train = data["y_train"]

    X_val = data["X_val"]
    y_val = data["y_val"]

    print("TRAIN:", X_train.shape, y_train.shape)
    print("VAL:  ", X_val.shape, y_val.shape)

    model = build_dscnn()

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
            BEST_MODEL_PATH,
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
    print("BEST MODEL:", BEST_MODEL_PATH)


if __name__ == "__main__":
    main()
