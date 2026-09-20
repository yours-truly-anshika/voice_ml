from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


SR = 16000
FRAME = 480       # 30 ms
WINDOW = 16000    # 1 second
STEP = 800        # 50 ms


SOURCE_DIR = Path("dataset/processed/positive")
META_FILE = Path("dataset/processed/positive_endpoint_with_split.csv")


df = pd.read_csv(META_FILE)
df = df[df["Split"] == "val"]


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

    # Calculate short-frame RMS values.
    frame_rms = []

    for start in range(0, len(audio) - FRAME + 1, FRAME):
        frame = audio[start:start + FRAME]

        rms = float(
            np.sqrt(np.mean(frame ** 2))
        )

        frame_rms.append(rms)

    frame_rms = np.asarray(frame_rms)

    # Estimate the quiet/noise level from the lower
    # portion of the recording's short-frame RMS values.
    noise_level = float(
        np.percentile(frame_rms, 20)
    )

    # Require a meaningful margin above the estimated
    # quiet level before calling a frame "active".
    threshold = max(
        noise_level * 3.0,
        1e-4,
    )

    best_activity_ratio = 0.0
    best_start = 0.0
    best_rms = 0.0

    # Search all possible 1-second windows.
    for start in range(
        0,
        len(audio) - WINDOW + 1,
        STEP,
    ):
        window = audio[start:start + WINDOW]

        window_rms = float(
            np.sqrt(np.mean(window ** 2))
        )

        # Calculate 30-ms RMS for this candidate window.
        active_frames = 0
        total_frames = 0

        for frame_start in range(
            0,
            WINDOW - FRAME + 1,
            FRAME,
        ):
            frame = window[
                frame_start:frame_start + FRAME
            ]

            rms = float(
                np.sqrt(np.mean(frame ** 2))
            )

            if rms > threshold:
                active_frames += 1

            total_frames += 1

        activity_ratio = (
            active_frames / total_frames
        )

        # Prefer sustained activity.
        # RMS is used only as a tie-breaker.
        if (
            activity_ratio > best_activity_ratio
            or (
                activity_ratio == best_activity_ratio
                and window_rms > best_rms
            )
        ):
            best_activity_ratio = activity_ratio
            best_start = start / SR
            best_rms = window_rms

    results.append(
        {
            "filename": filename,
            "noise_level": noise_level,
            "threshold": threshold,
            "best_start_s": best_start,
            "best_activity_ratio": best_activity_ratio,
            "best_window_rms": best_rms,
        }
    )


result_df = pd.DataFrame(results)


print()
print("VALIDATION RECORDINGS:", len(result_df))

print()
print("ACTIVITY-BASED BEST WINDOW STATISTICS:")

print(
    result_df[
        [
            "best_activity_ratio",
            "best_window_rms",
            "best_start_s",
        ]
    ].describe().round(4)
)

print()
print("INDIVIDUAL RECORDINGS:")

print(
    result_df[
        [
            "filename",
            "best_start_s",
            "best_activity_ratio",
            "best_window_rms",
        ]
    ].to_string(index=False)
)