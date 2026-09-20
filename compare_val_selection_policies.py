from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


SR = 16000
WINDOW = 16000
FRAME = 480
STEP = 800

SOURCE_DIR = Path("dataset/processed/positive")
META_FILE = Path("dataset/processed/positive_endpoint_with_split.csv")


def rms(audio):
    return float(np.sqrt(np.mean(audio ** 2)))


def frame_rms(audio):
    values = []

    for start in range(0, len(audio) - FRAME + 1, FRAME):
        frame = audio[start:start + FRAME]
        values.append(rms(frame))

    return np.asarray(values)


def find_rms_best(audio):
    best_rms = -1.0
    best_start = 0

    for start in range(
        0,
        len(audio) - WINDOW + 1,
        STEP,
    ):
        value = rms(audio[start:start + WINDOW])

        if value > best_rms:
            best_rms = value
            best_start = start

    return best_start, best_rms


def find_activity_best(audio):
    frames = frame_rms(audio)

    noise_level = float(
        np.percentile(frames, 20)
    )

    threshold = max(
        noise_level * 3.0,
        1e-4,
    )

    best_ratio = -1.0
    best_rms = -1.0
    best_start = 0

    for start in range(
        0,
        len(audio) - WINDOW + 1,
        STEP,
    ):
        window = audio[start:start + WINDOW]

        window_frames = frame_rms(window)

        activity_ratio = float(
            np.mean(window_frames > threshold)
        )

        window_rms = rms(window)

        if (
            activity_ratio > best_ratio
            or (
                activity_ratio == best_ratio
                and window_rms > best_rms
            )
        ):
            best_ratio = activity_ratio
            best_rms = window_rms
            best_start = start

    return (
        best_start,
        best_rms,
        best_ratio,
        threshold,
    )


df = pd.read_csv(META_FILE)

df = df[
    df["Split"] == "val"
].copy()


results = []


for _, row in df.iterrows():

    filename = row["Filename"]

    audio, sr = sf.read(
        SOURCE_DIR / filename,
        dtype="float32",
    )

    if sr != SR:
        raise RuntimeError(
            f"{filename}: expected {SR} Hz, got {sr}"
        )

    if audio.ndim != 1:
        raise RuntimeError(
            f"{filename}: expected mono audio"
        )

    duration = len(audio) / SR

    # --------------------------------------------------
    # Policy 1: current completion window
    # --------------------------------------------------

    speech_offset = float(
        row["SpeechOffsetSec"]
    )

    completion_start = max(
        0.0,
        speech_offset - 1.0,
    )

    completion_start_sample = int(
        round(completion_start * SR)
    )

    completion_window = audio[
        completion_start_sample:
        completion_start_sample + WINDOW
    ]

    completion_rms = rms(completion_window)

    # --------------------------------------------------
    # Policy 2: RMS-best window
    # --------------------------------------------------

    rms_start, rms_best = find_rms_best(audio)

    # --------------------------------------------------
    # Policy 3: activity-best window
    # --------------------------------------------------

    (
        activity_start,
        activity_best_rms,
        activity_ratio,
        activity_threshold,
    ) = find_activity_best(audio)

    results.append(
        {
            "filename": filename,
            "duration_s": duration,

            "completion_start_s": (
                completion_start_sample / SR
            ),
            "completion_rms": completion_rms,

            "rms_best_start_s": (
                rms_start / SR
            ),
            "rms_best_rms": rms_best,

            "activity_best_start_s": (
                activity_start / SR
            ),
            "activity_best_rms": activity_best_rms,
            "activity_ratio": activity_ratio,
            "activity_threshold": activity_threshold,
        }
    )


result_df = pd.DataFrame(results)


result_df["completion_vs_rms_ratio"] = (
    result_df["completion_rms"]
    / result_df["rms_best_rms"]
)


result_df["activity_vs_rms_ratio"] = (
    result_df["activity_best_rms"]
    / result_df["rms_best_rms"]
)


result_df["completion_activity_start_diff_s"] = (
    result_df["completion_start_s"]
    - result_df["activity_best_start_s"]
).abs()


result_df["rms_activity_start_diff_s"] = (
    result_df["rms_best_start_s"]
    - result_df["activity_best_start_s"]
).abs()


print()
print("=" * 80)
print("VALIDATION WINDOW SELECTION COMPARISON")
print("=" * 80)


print()
print("SUMMARY:")

print(
    result_df[
        [
            "completion_vs_rms_ratio",
            "activity_vs_rms_ratio",
            "activity_ratio",
            "completion_activity_start_diff_s",
            "rms_activity_start_diff_s",
        ]
    ].describe().round(3)
)


print()
print("INDIVIDUAL RECORDINGS:")

columns = [
    "filename",
    "completion_start_s",
    "rms_best_start_s",
    "activity_best_start_s",
    "completion_vs_rms_ratio",
    "activity_vs_rms_ratio",
    "activity_ratio",
]

print(
    result_df[
        columns
    ].to_string(
        index=False,
        formatters={
            "completion_start_s": "{:.2f}".format,
            "rms_best_start_s": "{:.2f}".format,
            "activity_best_start_s": "{:.2f}".format,
            "completion_vs_rms_ratio": "{:.3f}".format,
            "activity_vs_rms_ratio": "{:.3f}".format,
            "activity_ratio": "{:.3f}".format,
        },
    )
)


print()
print("LOW COMPLETION/RMS CASES:")

low_completion = result_df.sort_values(
    "completion_vs_rms_ratio"
).head(10)

print(
    low_completion[
        [
            "filename",
            "completion_start_s",
            "rms_best_start_s",
            "activity_best_start_s",
            "completion_vs_rms_ratio",
        ]
    ].to_string(index=False)
)


print()
print("LARGEST RMS/ACTIVITY DISAGREEMENTS:")

disagreement = result_df.sort_values(
    "rms_activity_start_diff_s",
    ascending=False,
).head(10)

print(
    disagreement[
        [
            "filename",
            "rms_best_start_s",
            "activity_best_start_s",
            "rms_activity_start_diff_s",
            "rms_best_rms",
            "activity_best_rms",
            "activity_ratio",
        ]
    ].to_string(index=False)
)