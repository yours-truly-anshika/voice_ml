import os
import csv
import random
import numpy as np
import soundfile as sf

# ============================================================
# Synthetic Positive Generator
# SIH26172 - Activate Orbit
#
# IMPORTANT:
# - Reads ONLY training positives
# - Reads ONLY training background
# - Never modifies originals
# - Creates exactly 2 variants per source
# - Uses at most 2 transformations per variant
# ============================================================

SEED = 26172

SAMPLE_RATE = 16000
TARGET_SECONDS = 3.0
TARGET_SAMPLES = int(SAMPLE_RATE * TARGET_SECONDS)

N_VARIANTS_PER_SOURCE = 2

SOURCE_DIR = r"dataset\final\train\positive"
BACKGROUND_DIR = r"dataset\final\train\background"

OUTPUT_DIR = r"dataset\synthetic\train\positive"
METADATA_PATH = r"dataset\synthetic\train\synthetic_positive_manifest.csv"

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


def shift_signal(x, shift_samples):
    return np.roll(x, shift_samples)

def add_noise_at_snr(signal, noise, snr_db):
    """
    Scale noise so that:
        20*log10(signal_rms / noise_rms) = snr_db

    The original noise file is never modified.
    """
    signal_rms = rms(signal)
    noise_rms = rms(noise)

    if signal_rms < 1e-8 or noise_rms < 1e-8:
        return signal.copy()

    desired_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    noise_scaled = noise * (desired_noise_rms / noise_rms)

    return signal + noise_scaled


def mild_reverb(x):
    """
    Small deterministic synthetic room response.

    This is deliberately mild:
    direct signal + several very low-level delayed reflections.
    """
    y = x.copy()

    delays_ms = [11, 23, 37, 53]
    gains = [0.18, 0.11, 0.07, 0.04]

    for delay_ms, gain in zip(delays_ms, gains):
        delay = int(SAMPLE_RATE * delay_ms / 1000.0)

        if delay < len(x):
            y[delay:] += gain * x[:-delay]

    return y


def prepare_audio(audio):
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if len(audio) < TARGET_SAMPLES:
        repeats = int(np.ceil(TARGET_SAMPLES / len(audio)))
        audio = np.tile(audio, repeats)[:TARGET_SAMPLES]
    else:
        audio = audio[:TARGET_SAMPLES]

    return audio.astype(np.float32)


# ------------------------------------------------------------
# Discover source files
# ------------------------------------------------------------

source_files = sorted(
    f for f in os.listdir(SOURCE_DIR)
    if f.lower().endswith(".wav")
)

background_files = sorted(
    f for f in os.listdir(BACKGROUND_DIR)
    if f.lower().endswith(".wav")
)

if len(source_files) != 121:
    raise RuntimeError(
        f"Expected 121 training positives, found {len(source_files)}"
    )

if len(background_files) != 121:
    raise RuntimeError(
        f"Expected 121 training background files, found {len(background_files)}"
    )


# ------------------------------------------------------------
# Generation plan
#
# Variant 1:
#   gain + background noise
#
# Variant 2:
#   timing shift + mild reverb
#
# This ensures at most two transformations per sample.
# ------------------------------------------------------------

rows = []

for source_index, source_file in enumerate(source_files):

    source_path = os.path.join(SOURCE_DIR, source_file)

    source_audio, source_sr = sf.read(
        source_path,
        dtype="float32"
    )

    if source_sr != SAMPLE_RATE:
        raise RuntimeError(
            f"Unexpected sample rate in {source_file}: {source_sr}"
        )

    source_audio = prepare_audio(source_audio)

    # Deterministically pair each source with one background.
    background_file = background_files[
        source_index % len(background_files)
    ]

    background_path = os.path.join(
        BACKGROUND_DIR,
        background_file
    )

    background_audio, background_sr = sf.read(
        background_path,
        dtype="float32"
    )

    if background_sr != SAMPLE_RATE:
        raise RuntimeError(
            f"Unexpected sample rate in {background_file}: {background_sr}"
        )

    background_audio = prepare_audio(background_audio)

    # --------------------------------------------------------
    # Variant 1: gain + noise
    # --------------------------------------------------------

    gain_db_1 = rng.uniform(-3.0, 3.0)
    snr_db_1 = rng.uniform(15.0, 25.0)

    variant1 = apply_gain(source_audio, gain_db_1)
    variant1 = add_noise_at_snr(
        variant1,
        background_audio,
        snr_db_1
    )

    variant1 = safe_peak_limit(variant1)

    output1 = f"{source_index:04d}_v01.wav"
    output1_path = os.path.join(OUTPUT_DIR, output1)

    sf.write(
        output1_path,
        variant1.astype(np.float32),
        SAMPLE_RATE,
        subtype="PCM_16"
    )

    rows.append({
        "synthetic_filename": output1,
        "source_filename": source_file,
        "background_filename": background_file,
        "variant": "gain_noise",
        "gain_db": round(gain_db_1, 4),
        "snr_db": round(snr_db_1, 4),
        "shift_ms": 0.0,
        "reverb": False,
        "seed": SEED + source_index * 2,
    })

    # --------------------------------------------------------
    # Variant 2: timing shift + mild reverb
    # --------------------------------------------------------

    shift_ms_2 = rng.uniform(-120.0, 120.0)
    shift_samples_2 = int(
        SAMPLE_RATE * shift_ms_2 / 1000.0
    )

    variant2 = shift_signal(
        source_audio,
        shift_samples_2
    )

    variant2 = mild_reverb(variant2)

    variant2 = safe_peak_limit(variant2)

    output2 = f"{source_index:04d}_v02.wav"
    output2_path = os.path.join(OUTPUT_DIR, output2)

    sf.write(
        output2_path,
        variant2.astype(np.float32),
        SAMPLE_RATE,
        subtype="PCM_16"
    )

    rows.append({
        "synthetic_filename": output2,
        "source_filename": source_file,
        "background_filename": "",
        "variant": "shift_reverb",
        "gain_db": 0.0,
        "snr_db": "",
        "shift_ms": round(shift_ms_2, 4),
        "reverb": True,
        "seed": SEED + source_index * 2 + 1,
    })


# ------------------------------------------------------------
# Write metadata
# ------------------------------------------------------------

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
    METADATA_PATH,
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


# ------------------------------------------------------------
# Final sanity checks
# ------------------------------------------------------------

generated_files = sorted(
    f for f in os.listdir(OUTPUT_DIR)
    if f.lower().endswith(".wav")
)

if len(generated_files) != 242:
    raise RuntimeError(
        f"Expected 242 generated WAVs, found {len(generated_files)}"
    )

if len(rows) != 242:
    raise RuntimeError(
        f"Expected 242 metadata rows, found {len(rows)}"
    )

print("SYNTHETIC POSITIVE GENERATOR CREATED DATA")
print("Source positives:", len(source_files))
print("Variants per source:", N_VARIANTS_PER_SOURCE)
print("Synthetic positives:", len(generated_files))
print("Output:", OUTPUT_DIR)
print("Manifest:", METADATA_PATH)
