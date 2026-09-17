import os
import csv
import random
import numpy as np
import soundfile as sf

SEED = 26172

TARGET_SR = 16000
TARGET_SECONDS = 3
TARGET_SAMPLES = TARGET_SR * TARGET_SECONDS

N_VARIANTS_PER_SOURCE = 2

SOURCE_DIR = r"dataset\final\train\background"
OUTPUT_DIR = r"dataset\synthetic\train\background"
MANIFEST_PATH = r"dataset\synthetic\train\synthetic_background_manifest.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

rng = random.Random(SEED)


def rms(x):
    x = x.astype(np.float64)
    return float(np.sqrt(np.mean(x * x)))


def safe_peak_limit(x, limit=0.95):
    peak = float(np.max(np.abs(x)))

    if peak > limit:
        x = x * (limit / peak)

    return x


def apply_gain(x, gain_db):
    gain = 10.0 ** (gain_db / 20.0)
    return x * gain


def tile_to_length(audio, target_samples):
    if len(audio) == 0:
        raise RuntimeError("Empty background audio")

    repeats = int(
        np.ceil(target_samples / len(audio))
    )

    return np.tile(
        audio,
        repeats
    )[:target_samples]


def circular_shift(x, shift_samples):
    return np.roll(x, shift_samples)


def prepare_audio(audio):
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    return audio.astype(np.float32)


source_files = sorted(
    [
        os.path.join(SOURCE_DIR, f)
        for f in os.listdir(SOURCE_DIR)
        if f.lower().endswith(".wav")
    ]
)

if len(source_files) != 121:
    raise RuntimeError(
        f"Expected 121 training background files, found {len(source_files)}"
    )


rows = []


for source_path in source_files:

    source_audio, source_sr = sf.read(
        source_path,
        dtype="float32"
    )

    if source_sr != TARGET_SR:
        raise RuntimeError(
            f"Unexpected sample rate in {source_path}: {source_sr}"
        )

    source_audio = prepare_audio(source_audio)

    source_filename = os.path.basename(source_path)

    base_audio = tile_to_length(
        source_audio,
        TARGET_SAMPLES
    )

    for variant_index in range(
        1,
        N_VARIANTS_PER_SOURCE + 1
    ):

        if variant_index == 1:

            gain_db = rng.uniform(
                -3.0,
                3.0
            )

            shift_ms = 0.0

            augmented = apply_gain(
                base_audio,
                gain_db
            )

            variant_name = "gain"

        else:

            gain_db = rng.uniform(
                -3.0,
                3.0
            )

            shift_ms = rng.uniform(
                -250.0,
                250.0
            )

            shift_samples = int(
                TARGET_SR * shift_ms / 1000
            )

            augmented = apply_gain(
                base_audio,
                gain_db
            )

            augmented = circular_shift(
                augmented,
                shift_samples
            )

            variant_name = "gain_shift"

        augmented = safe_peak_limit(
            augmented
        )

        synthetic_filename = (
            f"{source_filename[:-4]}_v{variant_index:02d}.wav"
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            synthetic_filename
        )

        sf.write(
            output_path,
            augmented.astype(np.float32),
            TARGET_SR,
            subtype="PCM_16"
        )

        rows.append(
            {
                "synthetic_filename": synthetic_filename,
                "source_filename": source_filename,
                "variant": variant_name,
                "gain_db": round(gain_db, 6),
                "shift_ms": round(shift_ms, 6),
                "seed": SEED,
            }
        )


fieldnames = [
    "synthetic_filename",
    "source_filename",
    "variant",
    "gain_db",
    "shift_ms",
    "seed",
]


with open(
    MANIFEST_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(rows)


print("SYNTHETIC BACKGROUND GENERATOR CREATED DATA")
print("Source backgrounds:", len(source_files))
print("Variants per source:", N_VARIANTS_PER_SOURCE)
print("Synthetic backgrounds:", len(rows))
print("Output:", OUTPUT_DIR)
print("Manifest:", MANIFEST_PATH)