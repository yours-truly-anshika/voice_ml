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

SOURCE_DIR = r"dataset\final\train\unknown"
BACKGROUND_DIR = r"dataset\final\train\background"

OUTPUT_DIR = r"dataset\synthetic\train\unknown"
MANIFEST_PATH = r"dataset\synthetic\train\synthetic_unknown_manifest.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

rng = random.Random(SEED)
np_rng = np.random.default_rng(SEED)


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


def add_noise_at_snr(signal, noise, snr_db):
    signal_rms = rms(signal)
    noise_rms = rms(noise)

    if signal_rms < 1e-8 or noise_rms < 1e-8:
        return signal.copy()

    desired_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))

    noise_scaled = noise * (
        desired_noise_rms / noise_rms
    )

    return signal + noise_scaled


def mild_reverb(x):
    delays_ms = [11, 23, 37, 53]
    gains = [0.18, 0.11, 0.07, 0.04]

    y = x.astype(np.float32).copy()

    for delay_ms, gain in zip(delays_ms, gains):
        delay = int(TARGET_SR * delay_ms / 1000)

        delayed = np.zeros_like(x)

        if delay < len(x):
            delayed[delay:] = x[:-delay]

        y += gain * delayed

    return y


def place_in_window(audio, shift_ms):
    """
    Place the original unknown utterance inside a 3-second window.

    The shift controls the placement around the center.
    No circular wrapping is used.
    """

    output = np.zeros(TARGET_SAMPLES, dtype=np.float32)

    length = len(audio)

    if length >= TARGET_SAMPLES:
        return audio[:TARGET_SAMPLES].astype(np.float32)

    center_start = (TARGET_SAMPLES - length) // 2

    shift_samples = int(
        TARGET_SR * shift_ms / 1000
    )

    start = center_start + shift_samples

    start = max(
        0,
        min(start, TARGET_SAMPLES - length)
    )

    output[start:start + length] = audio

    return output


def prepare_source(audio):
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

background_files = sorted(
    [
        os.path.join(BACKGROUND_DIR, f)
        for f in os.listdir(BACKGROUND_DIR)
        if f.lower().endswith(".wav")
    ]
)

if len(source_files) != 121:
    raise RuntimeError(
        f"Expected 121 unknown training files, found {len(source_files)}"
    )

if len(background_files) != 121:
    raise RuntimeError(
        f"Expected 121 background training files, found {len(background_files)}"
    )


rows = []


for source_index, source_path in enumerate(source_files):

    source_audio, source_sr = sf.read(
        source_path,
        dtype="float32"
    )

    if source_sr != TARGET_SR:
        raise RuntimeError(
            f"Unexpected sample rate in {source_path}: {source_sr}"
        )

    source_audio = prepare_source(source_audio)

    source_filename = os.path.basename(source_path)

    background_path = background_files[
        source_index % len(background_files)
    ]

    background_audio, background_sr = sf.read(
        background_path,
        dtype="float32"
    )

    if background_sr != TARGET_SR:
        raise RuntimeError(
            f"Unexpected background sample rate: {background_sr}"
        )

    background_audio = prepare_source(background_audio)

    for variant_index in range(1, N_VARIANTS_PER_SOURCE + 1):

        if variant_index == 1:

            gain_db = rng.uniform(-3.0, 3.0)
            snr_db = rng.uniform(15.0, 25.0)

            placed = place_in_window(
                source_audio,
                shift_ms=rng.uniform(-120.0, 120.0)
            )

            background = background_audio

            if len(background) < TARGET_SAMPLES:
                repeats = int(
                    np.ceil(
                        TARGET_SAMPLES / len(background)
                    )
                )

                background = np.tile(
                    background,
                    repeats
                )[:TARGET_SAMPLES]

            else:
                background = background[:TARGET_SAMPLES]

            augmented = apply_gain(
                placed,
                gain_db
            )

            augmented = add_noise_at_snr(
                augmented,
                background,
                snr_db
            )

            augmented = safe_peak_limit(
                augmented
            )

            variant_name = "gain_noise"
            shift_ms = 0.0
            reverb = False

        else:

            gain_db = 0.0
            snr_db = None

            shift_ms = rng.uniform(
                -120.0,
                120.0
            )

            placed = place_in_window(
                source_audio,
                shift_ms
            )

            augmented = mild_reverb(
                placed
            )

            augmented = safe_peak_limit(
                augmented
            )

            variant_name = "shift_reverb"
            reverb = True

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
                "background_filename": (
                    os.path.basename(background_path)
                    if variant_name == "gain_noise"
                    else ""
                ),
                "variant": variant_name,
                "gain_db": round(gain_db, 6),
                "snr_db": (
                    round(snr_db, 6)
                    if snr_db is not None
                    else ""
                ),
                "shift_ms": round(shift_ms, 6),
                "reverb": reverb,
                "seed": SEED,
            }
        )


fieldnames = [
    "synthetic_filename",
    "source_filename",
    "background_filename",
    "variant",
    "gain_db",
    "snr_db",
    "shift_ms",
    "reverb",
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


print("SYNTHETIC UNKNOWN GENERATOR CREATED DATA")
print("Source unknowns:", len(source_files))
print("Variants per source:", N_VARIANTS_PER_SOURCE)
print("Synthetic unknowns:", len(rows))
print("Output:", OUTPUT_DIR)
print("Manifest:", MANIFEST_PATH)