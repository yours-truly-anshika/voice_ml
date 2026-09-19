from pathlib import Path
import numpy as np

FEATURE_ROOT = Path("dataset/features")
OUTPUT_PATH = Path("dataset/features/dataset_features.npz")

SPLITS = ["train", "val", "test"]
CLASSES = ["positive", "unknown", "background"]
LABELS = {
    "positive": 0,
    "unknown": 1,
    "background": 2,
}

SEED = 26172


def load_files(split, class_name):
    directory = FEATURE_ROOT / split / class_name
    return sorted(directory.glob("*.npy"))


def load_feature(path):
    x = np.load(path)

    if x.shape != (40, 49, 1):
        raise ValueError(f"{path}: expected (40,49,1), got {x.shape}")

    if x.dtype != np.float32:
        raise ValueError(f"{path}: expected float32, got {x.dtype}")

    if not np.isfinite(x).all():
        raise ValueError(f"{path}: contains NaN or Inf")

    return x


def build_split(split, balance=False):
    rng = np.random.default_rng(SEED)

    files_by_class = {
        c: load_files(split, c)
        for c in CLASSES
    }

    if balance:
        target_count = max(len(files) for files in files_by_class.values())

        for c in CLASSES:
            files = files_by_class[c]

            if len(files) < target_count:
                indices = rng.choice(
                    len(files),
                    size=target_count,
                    replace=True,
                )
                files_by_class[c] = [files[i] for i in indices]

    features = []
    labels = []

    for class_name in CLASSES:
        for path in files_by_class[class_name]:
            features.append(load_feature(path))
            labels.append(LABELS[class_name])

    X = np.stack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)

    return X, y


def main():
    X_train, y_train = build_split("train", balance=True)
    X_val, y_val = build_split("val", balance=False)
    X_test, y_test = build_split("test", balance=False)

    np.savez_compressed(
        OUTPUT_PATH,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    print("DATASET PACKING COMPLETE")
    print("Output:", OUTPUT_PATH)
    print()
    print("TRAIN:", X_train.shape, y_train.shape)
    print("VAL:  ", X_val.shape, y_val.shape)
    print("TEST: ", X_test.shape, y_test.shape)
    print()
    print("TRAIN CLASS COUNTS:", np.bincount(y_train, minlength=3))
    print("VAL CLASS COUNTS:  ", np.bincount(y_val, minlength=3))
    print("TEST CLASS COUNTS: ", np.bincount(y_test, minlength=3))


if __name__ == "__main__":
    main()
