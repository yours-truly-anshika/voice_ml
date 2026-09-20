from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

POSITIVE_INPUT = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "streaming_positive_predictions.csv"
)

UNKNOWN_INPUT = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "synthetic_unknown_streaming_predictions.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "streaming_trigger_analysis.csv"
)

THRESHOLDS = [0.70, 0.80, 0.90, 0.95]
RUN_LENGTHS = [1, 2, 3, 4]

WEAK_VALIDATION_FILES = [
    "f5f4bb28-973c-4278-9967-38fb81c6a050_mandatory_5.wav",
    "263875b0-b8cb-4e7f-a43a-d6e551ee48ed_mandatory_2.wav",
    "263875b0-b8cb-4e7f-a43a-d6e551ee48ed_mandatory_3.wav",
    "263875b0-b8cb-4e7f-a43a-d6e551ee48ed_mandatory_5.wav",
    "f5f4bb28-973c-4278-9967-38fb81c6a050_mandatory_3.wav",
]


def longest_consecutive_run(values):
    longest = 0
    current = 0

    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def analyze_recordings(df, filename_column):
    rows = []

    for filename, group in df.groupby(filename_column):
        group = group.sort_values("StartSec").reset_index(drop=True)

        row = {
            "Filename": filename,
        }

        for threshold in THRESHOLDS:
            above_threshold = (
                group["PositiveConfidence"].to_numpy() >= threshold
            )

            longest_run = longest_consecutive_run(
                above_threshold
            )

            row[f"LongestRunAt_{threshold:.2f}"] = longest_run

        rows.append(row)

    return pd.DataFrame(rows)


def print_weak_validation_windows(validation_positive):
    print("\n" + "=" * 70)
    print("WEAK VALIDATION RECORDING WINDOW DETAILS")
    print("=" * 70)

    weak = validation_positive[
        validation_positive["Filename"].isin(
            WEAK_VALIDATION_FILES
        )
    ].copy()

    if weak.empty:
        print("No weak validation recordings were found.")
        return

    for filename in WEAK_VALIDATION_FILES:
        rows = weak[
            weak["Filename"] == filename
        ].copy()

        print("\n" + filename)

        if rows.empty:
            print("  No prediction rows found.")
            continue

        rows = rows.sort_values(
            "PositiveConfidence",
            ascending=False,
        ).head(5)

        display_columns = [
            "WindowIndex",
            "StartSec",
            "EndSec",
            "PositiveConfidence",
            "UnknownConfidence",
            "BackgroundConfidence",
            "PredictedClass",
        ]

        print(
            rows[display_columns].to_string(
                index=False
            )
        )


def main():
    print("=" * 70)
    print("STREAMING TRIGGER POLICY ANALYSIS")
    print("=" * 70)

    if not POSITIVE_INPUT.exists():
        raise FileNotFoundError(
            f"Positive prediction file not found:\n{POSITIVE_INPUT}"
        )

    if not UNKNOWN_INPUT.exists():
        raise FileNotFoundError(
            f"Synthetic unknown prediction file not found:\n{UNKNOWN_INPUT}"
        )

    positive_df = pd.read_csv(POSITIVE_INPUT)
    unknown_df = pd.read_csv(UNKNOWN_INPUT)

    positive_df["Split"] = (
        positive_df["Split"].astype(str).str.lower()
    )

    validation_positive = positive_df[
        positive_df["Split"] == "val"
    ].copy()

    unknown_df = unknown_df.rename(
        columns={
            "filename": "Filename",
            "start_s": "StartSec",
            "positive_confidence": "PositiveConfidence",
        }
    )

    unknown_df["Filename"] = (
        unknown_df["Filename"].astype(str)
    )

    positive_results = analyze_recordings(
        validation_positive,
        "Filename",
    )

    unknown_results = analyze_recordings(
        unknown_df,
        "Filename",
    )

    print("\n" + "=" * 70)
    print("VALIDATION POSITIVE VS SYNTHETIC UNKNOWN")
    print("=" * 70)

    output_rows = []

    for threshold in THRESHOLDS:
        print(f"\nThreshold: {threshold:.2f}")

        positive_column = (
            f"LongestRunAt_{threshold:.2f}"
        )

        unknown_column = (
            f"LongestRunAt_{threshold:.2f}"
        )

        positive_count = len(positive_results)
        unknown_count = len(unknown_results)

        for run_length in RUN_LENGTHS:
            positive_triggers = int(
                (
                    positive_results[positive_column]
                    >= run_length
                ).sum()
            )

            unknown_triggers = int(
                (
                    unknown_results[unknown_column]
                    >= run_length
                ).sum()
            )

            positive_rate = (
                positive_triggers / positive_count
            )

            unknown_rate = (
                unknown_triggers / unknown_count
            )

            print(
                f"  {run_length} consecutive windows:"
            )

            print(
                f"    validation positives: "
                f"{positive_triggers}/{positive_count} "
                f"({positive_rate:.1%})"
            )

            print(
                f"    synthetic unknowns:   "
                f"{unknown_triggers}/{unknown_count} "
                f"({unknown_rate:.1%})"
            )

            output_rows.append(
                {
                    "Threshold": threshold,
                    "ConsecutiveWindows": run_length,
                    "ValidationPositiveTriggers": (
                        positive_triggers
                    ),
                    "ValidationPositiveTotal": (
                        positive_count
                    ),
                    "ValidationPositiveRate": (
                        positive_rate
                    ),
                    "SyntheticUnknownTriggers": (
                        unknown_triggers
                    ),
                    "SyntheticUnknownTotal": (
                        unknown_count
                    ),
                    "SyntheticUnknownRate": (
                        unknown_rate
                    ),
                }
            )

    results = pd.DataFrame(output_rows)

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 70)
    print("VALIDATION RECORDING DETAILS")
    print("=" * 70)

    detail_columns = [
        "Filename",
        "LongestRunAt_0.80",
        "LongestRunAt_0.90",
        "LongestRunAt_0.95",
    ]

    detail = positive_results[
        detail_columns
    ].copy()

    detail["BestPositiveConfidence"] = (
        validation_positive
        .groupby("Filename")["PositiveConfidence"]
        .max()
        .reindex(detail["Filename"])
        .to_numpy()
    )

    detail = detail.sort_values(
        [
            "LongestRunAt_0.90",
            "BestPositiveConfidence",
        ],
        ascending=[True, True],
    )

    print(
        detail.to_string(index=False)
    )

    print_weak_validation_windows(
        validation_positive
    )

    print("\n" + "=" * 70)
    print("SAVED")
    print("=" * 70)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()