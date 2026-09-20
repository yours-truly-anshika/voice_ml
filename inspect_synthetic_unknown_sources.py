from pathlib import Path

import pandas as pd
import soundfile as sf


MANIFEST = Path("dataset/processed/authoritative_dataset_manifest.csv")


def main():
    df = pd.read_csv(MANIFEST)

    synthetic_unknown = df[
        (df["class"] == "unknown")
        & (df["source_type"] == "synthetic")
    ].copy()

    print("SYNTHETIC UNKNOWN RECORDS:", len(synthetic_unknown))

    print("\nCOUNTS BY SPLIT:")
    print(
        synthetic_unknown["split"]
        .value_counts()
        .sort_index()
    )

    rows = []

    for _, row in synthetic_unknown.iterrows():
        path = Path(row["full_path"])

        if not path.exists():
            rows.append({
                "split": row["split"],
                "filename": row["filename"],
                "exists": False,
                "duration_s": None,
                "sample_rate": None,
                "channels": None,
            })
            continue

        info = sf.info(path)

        rows.append({
            "split": row["split"],
            "filename": row["filename"],
            "exists": True,
            "duration_s": info.frames / info.samplerate,
            "sample_rate": info.samplerate,
            "channels": info.channels,
        })

    result = pd.DataFrame(rows)

    print("\nEXISTENCE:")
    print(result["exists"].value_counts())

    print("\nDURATION BY SPLIT:")
    print(
        result.groupby("split")["duration_s"]
        .describe()
        .round(3)
        .to_string()
    )

    print("\nSAMPLE RATE COUNTS:")
    print(
        result["sample_rate"]
        .value_counts()
        .sort_index()
    )

    print("\nCHANNEL COUNTS:")
    print(
        result["channels"]
        .value_counts()
        .sort_index()
    )

    print("\nFILES SHORTER THAN 2 SECONDS:")
    print(
        result[
            result["exists"]
            & (result["duration_s"] < 2.0)
        ]
        .to_string(index=False)
    )

    print("\nFIRST 10 FILES:")
    print(
        result.head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()