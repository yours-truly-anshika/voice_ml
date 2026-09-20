from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_ROOT / "dataset"

FILES_TO_CHECK = [
    DATASET_DIR / "processed" / "split_manifest.csv",
    DATASET_DIR / "processed" / "dataset_manifest.csv",
    DATASET_DIR / "features" / "dataset_features.npz",
]


def print_csv_info(path):
    print("\n" + "=" * 70)
    print(path)
    print("=" * 70)

    if not path.exists():
        print("NOT FOUND")
        return

    df = pd.read_csv(path)

    print("\nShape:")
    print(f"  rows: {len(df)}")
    print(f"  columns: {len(df.columns)}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  {column}")

    print("\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))

    for column in df.columns:
        name = str(column).lower()

        if any(
            keyword in name
            for keyword in [
                "class",
                "label",
                "split",
                "source",
                "filename",
                "path",
                "type",
            ]
        ):
            print(f"\nValue counts: {column}")

            print(
                df[column]
                .astype(str)
                .value_counts()
                .head(30)
                .to_string()
            )


def inspect_npz(path):
    print("\n" + "=" * 70)
    print(path)
    print("=" * 70)

    if not path.exists():
        print("NOT FOUND")
        return

    import numpy as np

    data = np.load(path)

    print("\nArrays:")

    for key in data.files:
        array = data[key]

        print(
            f"  {key}: "
            f"shape={array.shape}, "
            f"dtype={array.dtype}"
        )

        if array.ndim == 1 and len(array) <= 20:
            print(f"    values={array}")


def main():
    print("=" * 70)
    print("NEGATIVE DATASET PROVENANCE INSPECTION")
    print("=" * 70)

    for path in FILES_TO_CHECK:

        if path.suffix.lower() == ".csv":
            print_csv_info(path)

        elif path.suffix.lower() == ".npz":
            inspect_npz(path)

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()