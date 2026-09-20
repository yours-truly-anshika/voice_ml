from pathlib import Path
import pandas as pd


MANIFEST = Path("dataset/processed/authoritative_dataset_manifest.csv")


def main():
    df = pd.read_csv(MANIFEST)

    print("MANIFEST SHAPE:", df.shape)
    print("\nCOLUMNS:")
    for col in df.columns:
        print(f"  {col}")

    print("\nCLASS COUNTS:")
    print(df["class"].value_counts().sort_index())

    print("\nSOURCE TYPE COUNTS:")
    print(df["source_type"].value_counts().sort_index())

    print("\nSPLIT × CLASS:")
    print(
        pd.crosstab(
            df["split"],
            df["class"],
            margins=True,
        )
    )

    print("\nNEGATIVE SOURCE TYPES:")
    negative = df[df["class"] != "positive"].copy()
    print(
        negative.groupby(["class", "source_type"])
        .size()
        .reset_index(name="count")
        .to_string(index=False)
    )

    print("\nNEGATIVE FILE PATH EXAMPLES:")
    print(
        negative[
            ["split", "class", "source_type", "filename", "full_path"]
        ]
        .head(30)
        .to_string(index=False)
    )

    print("\nNEGATIVE COUNTS BY SPLIT:")
    print(
        negative.groupby(["split", "class"])
        .size()
        .unstack(fill_value=0)
    )


if __name__ == "__main__":
    main()