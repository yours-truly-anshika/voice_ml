from pathlib import Path

import pandas as pd
import soundfile as sf


SR = 16000
WINDOW = 16000

SOURCE_DIR = Path("dataset/processed/positive")
META_FILE = Path("dataset/processed/positive_endpoint_with_split.csv")
OUTPUT_DIR = Path("dataset/processed/selection_disagreements")


FILES = [
    "f5f4bb28-973c-4278-9967-38fb81c6a050_mandatory_1.wav",
    "f5f4bb28-973c-4278-9967-38fb81c6a050_mandatory_2.wav",
    "f5f4bb28-973c-4278-9967-38fb81c6a050_mandatory_3.wav",
    "263875b0-b8cb-4e7f-a43a-d6e551ee48ed_mandatory_1.wav",
]


df = pd.read_csv(META_FILE)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def rms(audio):
    return float(
        (audio ** 2).mean() ** 0.5
    )


def frame_rms(audio, frame_size=480):
    values = []

    for start in range(
        0,
        len(audio) - frame_size + 1,
        frame_size,
    ):
        frame = audio[
            start:start + frame_size
        ]

        values.append(rms(frame))

    return values


def find_rms_best(audio, step=800):
    best_start = 0
    best_value = -1.0

    for start in range(
        0,
        len(audio) - WINDOW + 1,
        step,
    ):
        value = rms(
            audio[
                start:start + WINDOW
            ]
        )

        if value > best_value:
            best_value = value
            best_start = start

    return best_start


def find_activity_best(audio, step=800):
    frames = frame_rms(audio)

    noise_level = sorted(frames)[
        max(0, int(len(frames) * 0.20) - 1)
    ]

    threshold = max(
        noise_level * 3.0,
        1e-4,
    )

    best_start = 0
    best_ratio = -1.0
    best_rms = -1.0

    for start in range(
        0,
        len(audio) - WINDOW + 1,
        step,
    ):
        window = audio[
            start:start + WINDOW
        ]

        values = frame_rms(window)

        activity_ratio = sum(
            value > threshold
            for value in values
        ) / len(values)

        window_rms = rms(window)

        if (
            activity_ratio > best_ratio
            or (
                activity_ratio == best_ratio
                and window_rms > best_rms
            )
        ):
            best_start = start
            best_ratio = activity_ratio
            best_rms = window_rms

    return best_start


for filename in FILES:

    source = SOURCE_DIR / filename

    audio, sr = sf.read(
        source,
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

    rms_start = find_rms_best(audio)
    activity_start = find_activity_best(audio)

    stem = Path(filename).stem

    sf.write(
        OUTPUT_DIR / f"{stem}__FULL.wav",
        audio,
        SR,
    )

    sf.write(
        OUTPUT_DIR / f"{stem}__RMS_BEST.wav",
        audio[
            rms_start:rms_start + WINDOW
        ],
        SR,
    )

    sf.write(
        OUTPUT_DIR / f"{stem}__ACTIVITY_BEST.wav",
        audio[
            activity_start:activity_start + WINDOW
        ],
        SR,
    )

    print(
        filename,
        f"| RMS {rms_start / SR:.2f}s",
        f"| Activity {activity_start / SR:.2f}s",
    )


print()
print("GENERATED:", len(FILES) * 3)
print("OUTPUT:", OUTPUT_DIR)