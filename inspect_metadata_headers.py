from pathlib import Path

import pandas as pd


FILES = [
    Path("dataset/processed/positive_endpoint_with_split.csv"),
    Path("dataset/processed/positive_endpoint_analysis.csv"),
]


for path in FILES:
    print()
    print("=" * 70)
    print(path)
    print("=" * 70)

    df = pd.read_csv(path)

    print()
    print("COLUMNS:")
    for column in df.columns:
        print(f"  {column}")

    print()
    print("FIRST ROW:")
    print(df.head(1).to_string(index=False))
