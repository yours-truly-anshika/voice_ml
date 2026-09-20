from pathlib import Path

import pandas as pd
import soundfile as sf


MANIFEST = Path("dataset/processed/authoritative_dataset_manifest.csv")


def get_duration(path):
    info = sf.info(path)
    return info.frames / info.samplerate


def main():
    df = pd.read_csv(MANIFEST)

    negative = df[
        (df["class"] != "positive")
        & (df["split"].isin(["val", "test"]))
    ].copy()

    rows = []

    for _, row in negative.iterrows():
        path = Path(row["full_path"])

        if not path.exists():
            rows.append({
                "split": row["split"],
                "class": row["class"],
                "source_type": row["source_type"],
                "filename": row["filename"],
                "duration_s": None,
                "exists": False,
            })
            continue

        duration = get_duration(path)

        rows.append({
            "split": row["split"],
            "class": row["class"],
            "source_type": row["source_type"],
            "filename": row["filename"],
            "duration_s": duration,
            "exists": True,
        })

    result = pd.DataFrame(rows)

    print("NEGATIVE VALIDATION/TEST SOURCES:", len(result))

    print("\nEXISTENCE:")
    print(result["exists"].value_counts())

    print("\nDURATION SUMMARY BY SPLIT / CLASS / SOURCE TYPE:")
    print(
        result.groupby(
            ["split", "class", "source_type"]
        )["duration_s"]
        .describe()
        .round(3)
        .to_string()
    )

    print("\nFILES SHORTER THAN 1 SECOND:")
    short = result[
        result["exists"]
        & (result["duration_s"] < 1.0)
    ]

    print(
        short[
            [
                "split",
                "class",
                "source_type",
                "filename",
                "duration_s",
            ]
        ]
        .to_string(index=False)
    )

    print("\nFILES AT LEAST 2 SECONDS:")
    long_enough = result[
        result["exists"]
        & (result["duration_s"] >= 2.0)
    ]

    print(
        long_enough.groupby(
            ["split", "class", "source_type"]
        ).size()
        .to_string()
    )


if __name__ == "__main__":
    main()