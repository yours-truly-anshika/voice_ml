import os
import csv
import random
import numpy as np
import soundfile as sf


SEED = 26172

TARGET_SR = 16000
TARGET_SAMPLES = 16000

SPLITS = {
    "train": {
        "source_dir": r"dataset\final\train\background",
        "output_dir": r"dataset\windowed\train\background",
        "expected_sources": 121,
        "augmentations": 2,
    },
    "val": {
        "source_dir": r"dataset\final\val\background",
        "output_dir": r"dataset\windowed\val\background",
        "expected_sources": 26,
        "augmentations": 1,
    },
    "test": {
        "source_dir": r"dataset\final\test\background",
        "output_dir": r"dataset\windowed\test\background",
        "expected_sources": 26,
        "augmentations": 1,
    },
}

MANIFEST_PATH = r"dataset\processed\background_windowed_manifest.csv"

rng = random.Random(SEED)


def prepare_audio(audio):
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    return audio.astype(np.float32)


def normalize_to_one_second(audio):
    """
    Convert an approximately 1-second source recording
    into exactly 16,000 samples.

    Shorter than 1 second:
        trailing zero padding.

    Longer than 1 second:
        deterministic center crop.
    """

    length = len(audio)

    if length < TARGET_SAMPLES:
        padding = TARGET_SAMPLES - length

        output = np.pad(
            audio,
            (0, padding),
            mode="constant",
            constant_values=0.0,
        )

        strategy = "trailing_zero_padding"

    elif length > TARGET_SAMPLES:
        excess = length - TARGET_SAMPLES

        start = excess // 2
        end = start + TARGET_SAMPLES

        output = audio[start:end]

        strategy = "center_crop"

    else:
        output = audio
        strategy = "unchanged"

    return output.astype(np.float32), strategy


def apply_gain(audio, gain_db):
    gain = 10.0 ** (gain_db / 20.0)
    return audio * gain


def circular_shift(audio, shift_samples):
    return np.roll(audio, shift_samples)


def safe_peak_limit(audio, limit=0.95):
    peak = float(np.max(np.abs(audio)))

    if peak > limit:
        audio = audio * (limit / peak)

    return audio


rows = []

for split, config in SPLITS.items():

    source_dir = config["source_dir"]
    output_dir = config["output_dir"]

    os.makedirs(output_dir, exist_ok=True)

    source_files = sorted(
        [
            os.path.join(source_dir, filename)
            for filename in os.listdir(source_dir)
            if filename.lower().endswith(".wav")
        ]
    )

    if len(source_files) != config["expected_sources"]:
        raise RuntimeError(
            f"{split}: expected "
            f"{config['expected_sources']} background files, "
            f"found {len(source_files)}"
        )

    for source_path in source_files:

        source_filename = os.path.basename(source_path)

        source_audio, source_sr = sf.read(
            source_path,
            dtype="float32",
        )

        if source_sr != TARGET_SR:
            raise RuntimeError(
                f"Unexpected sample rate in {source_path}: "
                f"{source_sr}"
            )

        source_audio = prepare_audio(source_audio)

        normalized_audio, window_strategy = normalize_to_one_second(
            source_audio
        )

        if split == "train":

            # Variant 1: gain only
            gain_db = rng.uniform(-3.0, 3.0)

            augmented = apply_gain(
                normalized_audio,
                gain_db,
            )

            augmented = safe_peak_limit(
                augmented
            )

            filename = (
                f"train_{source_filename[:-4]}_v01.wav"
            )

            output_path = os.path.join(
                output_dir,
                filename,
            )

            sf.write(
                output_path,
                augmented.astype(np.float32),
                TARGET_SR,
                subtype="PCM_16",
            )

            rows.append(
                {
                    "split": split,
                    "source_filename": source_filename,
                    "window_filename": filename,
                    "window_start_sample": (
                        0
                        if len(source_audio) <= TARGET_SAMPLES
                        else (
                            len(source_audio) - TARGET_SAMPLES
                        ) // 2
                    ),
                    "window_end_sample": (
                        min(len(source_audio), TARGET_SAMPLES)
                        if len(source_audio) <= TARGET_SAMPLES
                        else (
                            (len(source_audio) - TARGET_SAMPLES)
                            // 2
                            + TARGET_SAMPLES
                        )
                    ),
                    "original_sample_count": len(source_audio),
                    "window_sample_count": TARGET_SAMPLES,
                    "window_duration_sec": 1.0,
                    "window_strategy": window_strategy,
                    "variant": "gain",
                    "gain_db": round(gain_db, 6),
                    "shift_ms": 0.0,
                    "seed": SEED,
                }
            )

            # Variant 2: gain + circular shift
            gain_db = rng.uniform(-3.0, 3.0)

            shift_ms = rng.uniform(
                -250.0,
                250.0,
            )

            shift_samples = int(
                round(
                    shift_ms
                    * TARGET_SR
                    / 1000.0
                )
            )

            augmented = apply_gain(
                normalized_audio,
                gain_db,
            )

            augmented = circular_shift(
                augmented,
                shift_samples,
            )

            augmented = safe_peak_limit(
                augmented
            )

            filename = (
                f"train_{source_filename[:-4]}_v02.wav"
            )

            output_path = os.path.join(
                output_dir,
                filename,
            )

            sf.write(
                output_path,
                augmented.astype(np.float32),
                TARGET_SR,
                subtype="PCM_16",
            )

            rows.append(
                {
                    "split": split,
                    "source_filename": source_filename,
                    "window_filename": filename,
                    "window_start_sample": (
                        0
                        if len(source_audio) <= TARGET_SAMPLES
                        else (
                            len(source_audio) - TARGET_SAMPLES
                        ) // 2
                    ),
                    "window_end_sample": (
                        min(len(source_audio), TARGET_SAMPLES)
                        if len(source_audio) <= TARGET_SAMPLES
                        else (
                            (len(source_audio) - TARGET_SAMPLES)
                            // 2
                            + TARGET_SAMPLES
                        )
                    ),
                    "original_sample_count": len(source_audio),
                    "window_sample_count": TARGET_SAMPLES,
                    "window_duration_sec": 1.0,
                    "window_strategy": window_strategy,
                    "variant": "gain_shift",
                    "gain_db": round(gain_db, 6),
                    "shift_ms": round(shift_ms, 6),
                    "seed": SEED,
                }
            )

        else:

            # Validation/test:
            # no random augmentation.
            filename = (
                f"{split}_{source_filename}"
            )

            output_path = os.path.join(
                output_dir,
                filename,
            )

            sf.write(
                output_path,
                normalized_audio.astype(np.float32),
                TARGET_SR,
                subtype="PCM_16",
            )

            rows.append(
                {
                    "split": split,
                    "source_filename": source_filename,
                    "window_filename": filename,
                    "window_start_sample": (
                        0
                        if len(source_audio) <= TARGET_SAMPLES
                        else (
                            len(source_audio) - TARGET_SAMPLES
                        ) // 2
                    ),
                    "window_end_sample": (
                        min(len(source_audio), TARGET_SAMPLES)
                        if len(source_audio) <= TARGET_SAMPLES
                        else (
                            (len(source_audio) - TARGET_SAMPLES)
                            // 2
                            + TARGET_SAMPLES
                        )
                    ),
                    "original_sample_count": len(source_audio),
                    "window_sample_count": TARGET_SAMPLES,
                    "window_duration_sec": 1.0,
                    "window_strategy": window_strategy,
                    "variant": "original",
                    "gain_db": 0.0,
                    "shift_ms": 0.0,
                    "seed": SEED,
                }
            )


os.makedirs(
    os.path.dirname(MANIFEST_PATH),
    exist_ok=True,
)

fieldnames = [
    "split",
    "source_filename",
    "window_filename",
    "window_start_sample",
    "window_end_sample",
    "original_sample_count",
    "window_sample_count",
    "window_duration_sec",
    "window_strategy",
    "variant",
    "gain_db",
    "shift_ms",
    "seed",
]

with open(
    MANIFEST_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(rows)


print("BACKGROUND WINDOW GENERATION COMPLETE")
print()

for split in SPLITS:
    split_rows = [
        row for row in rows
        if row["split"] == split
    ]

    print(
        f"{split}: "
        f"{len(split_rows)} windows"
    )

print()
print("TOTAL WINDOWS:", len(rows))
print("TARGET SAMPLE COUNT:", TARGET_SAMPLES)
print("TARGET DURATION: 1.0 sec")
print("MANIFEST:", MANIFEST_PATH)