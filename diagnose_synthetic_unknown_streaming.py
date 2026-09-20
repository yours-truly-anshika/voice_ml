from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf

from feature_frontend import extract_log_mel


MODEL = Path("models/dscnn_best.keras")
MANIFEST = Path("dataset/processed/authoritative_dataset_manifest.csv")

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 16000
HOP_SAMPLES = 1600

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


def load_audio(path):
    audio, sr = sf.read(path, dtype="float32")

    if sr != SAMPLE_RATE:
        raise ValueError(
            f"{path}: expected {SAMPLE_RATE} Hz, got {sr} Hz"
        )

    if audio.ndim != 1:
        raise ValueError(
            f"{path}: expected mono audio, got shape {audio.shape}"
        )

    return audio


def main():
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL)

    manifest = pd.read_csv(MANIFEST)

    sources = manifest[
        (manifest["class"] == "unknown")
        & (manifest["source_type"] == "synthetic")
    ].copy()

    print(f"SYNTHETIC UNKNOWN SOURCES: {len(sources)}")

    all_predictions = []

    for source_number, (_, record) in enumerate(
        sources.iterrows(),
        start=1,
    ):
        path = Path(record["full_path"])
        audio = load_audio(path)

        starts = list(
            range(
                0,
                len(audio) - WINDOW_SAMPLES + 1,
                HOP_SAMPLES,
            )
        )

        windows = np.stack(
            [
                extract_log_mel(
                    audio[
                        start:start + WINDOW_SAMPLES
                    ]
                )
                for start in starts
            ],
            axis=0,
        )

        probabilities = model.predict(
            windows,
            batch_size=32,
            verbose=0,
        )

        for window_index, start in enumerate(starts):
            all_predictions.append({
                "filename": record["filename"],
                "start_s": start / SAMPLE_RATE,
                "end_s": (
                    start + WINDOW_SAMPLES
                ) / SAMPLE_RATE,
                "positive_confidence": float(
                    probabilities[
                        window_index,
                        POSITIVE_INDEX,
                    ]
                ),
            })

        if (
            source_number == 1
            or source_number % 25 == 0
            or source_number == len(sources)
        ):
            print(
                f"Processed "
                f"{source_number}/{len(sources)} "
                f"recordings"
            )

    predictions = pd.DataFrame(all_predictions)

    output_predictions = Path(
        "dataset/processed/"
        "synthetic_unknown_streaming_predictions.csv"
    )

    predictions.to_csv(
        output_predictions,
        index=False,
    )

    summary_rows = []

    for filename, group in predictions.groupby(
        "filename"
    ):
        confidence = group[
            "positive_confidence"
        ].to_numpy()

        row = {
            "filename": filename,
            "windows": len(group),
            "max_positive_confidence": float(
                confidence.max()
            ),
        }

        for threshold in THRESHOLDS:
            above = confidence >= threshold

            row[
                f"windows_ge_{threshold:.2f}"
            ] = int(above.sum())

            row[
                f"longest_run_ge_{threshold:.2f}"
            ] = longest_run(above)

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    output_summary = Path(
        "dataset/processed/"
        "synthetic_unknown_streaming_summary.csv"
    )

    summary.to_csv(
        output_summary,
        index=False,
    )

    print("\nSUMMARY BY THRESHOLD:")

    for threshold in THRESHOLDS:
        longest = summary[
            f"longest_run_ge_{threshold:.2f}"
        ]

        print(
            f"\nThreshold {threshold:.2f}"
        )

        print(
            f"  recordings with >=1: "
            f"{int((longest >= 1).sum())}/"
            f"{len(summary)}"
        )

        print(
            f"  recordings with >=2: "
            f"{int((longest >= 2).sum())}/"
            f"{len(summary)}"
        )

        print(
            f"  recordings with >=3: "
            f"{int((longest >= 3).sum())}/"
            f"{len(summary)}"
        )

        print(
            f"  recordings with >=4: "
            f"{int((longest >= 4).sum())}/"
            f"{len(summary)}"
        )

        print(
            f"  median longest run: "
            f"{longest.median():.1f}"
        )

        print(
            f"  maximum longest run: "
            f"{longest.max()}"
        )

    print("\nMAX POSITIVE CONFIDENCE:")

    print(
        summary[
            "max_positive_confidence"
        ]
        .describe()
        .round(4)
        .to_string()
    )

    print(
        f"\nSaved:\n"
        f"  {output_predictions}\n"
        f"  {output_summary}"
    )


if __name__ == "__main__":
    main()