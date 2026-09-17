from pathlib import Path
import csv
import math
import wave
import numpy as np

# ============================================================
# Configuration
# ============================================================

INPUT_DIR = Path(r"C:\Users\Anshika\voice_ml\dataset\processed\positive")
OUTPUT_CSV = Path(r"C:\Users\Anshika\voice_ml\dataset\processed\positive_endpoint_analysis.csv")

SAMPLE_RATE = 16000

# Short-time energy analysis
FRAME_MS = 30
HOP_MS = 10

# Speech detection parameters
# We estimate the noise floor from the quietest part of each recording.
NOISE_PERCENTILE = 20

# Minimum energy threshold above estimated noise floor.
THRESHOLD_DB_ABOVE_NOISE = 10.0

# Smooth detection so tiny gaps do not split speech.
SMOOTHING_MS = 100

# Ignore extremely short active regions.
MIN_SPEECH_REGION_MS = 80


# ============================================================
# WAV loading
# ============================================================

def load_wav(path):
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if channels != 1:
        raise ValueError(f"Expected mono, got {channels} channels")

    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM, got {sample_width * 8}-bit")

    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sample_rate} Hz")

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

    # Normalize to [-1, 1]
    audio /= 32768.0

    return audio, sample_rate


# ============================================================
# RMS calculation
# ============================================================

def calculate_rms_db(audio, frame_samples, hop_samples):
    if len(audio) < frame_samples:
        return np.array([])

    values = []

    for start in range(0, len(audio) - frame_samples + 1, hop_samples):
        frame = audio[start:start + frame_samples]

        rms = np.sqrt(np.mean(frame * frame) + 1e-12)

        # Convert to dBFS
        db = 20.0 * math.log10(max(rms, 1e-12))
        values.append(db)

    return np.array(values, dtype=np.float32)


# ============================================================
# Boolean smoothing
# ============================================================

def smooth_boolean(mask, max_gap_frames):
    """
    Fill short inactive gaps inside active speech.

    This is important because a two-word phrase can naturally
    contain a pause between words. We don't want every pause
    to become a separate speech region.
    """

    mask = mask.copy()

    if len(mask) == 0:
        return mask

    # Fill short FALSE gaps between TRUE regions.
    i = 0

    while i < len(mask):
        if mask[i]:
            i += 1
            continue

        start = i

        while i < len(mask) and not mask[i]:
            i += 1

        end = i

        gap_length = end - start

        if (
            start > 0
            and end < len(mask)
            and gap_length <= max_gap_frames
        ):
            mask[start:end] = True

    return mask


# ============================================================
# Remove tiny active regions
# ============================================================

def remove_short_active_regions(mask, min_length_frames):
    mask = mask.copy()

    i = 0

    while i < len(mask):
        if not mask[i]:
            i += 1
            continue

        start = i

        while i < len(mask) and mask[i]:
            i += 1

        end = i

        if end - start < min_length_frames:
            mask[start:end] = False

    return mask


# ============================================================
# Analyze one file
# ============================================================

def analyze_file(path):

    audio, sample_rate = load_wav(path)

    duration_sec = len(audio) / sample_rate

    frame_samples = int(sample_rate * FRAME_MS / 1000)
    hop_samples = int(sample_rate * HOP_MS / 1000)

    rms_db = calculate_rms_db(
        audio,
        frame_samples,
        hop_samples
    )

    if len(rms_db) == 0:
        raise ValueError("Audio shorter than analysis frame")

    # Estimate the recording's noise floor.
    noise_floor_db = float(
        np.percentile(rms_db, NOISE_PERCENTILE)
    )

    threshold_db = (
        noise_floor_db
        + THRESHOLD_DB_ABOVE_NOISE
    )

    active = rms_db >= threshold_db

    # Allow short gaps inside the spoken phrase.
    smoothing_frames = max(
        1,
        int(round(SMOOTHING_MS / HOP_MS))
    )

    active = smooth_boolean(
        active,
        smoothing_frames
    )

    # Remove tiny accidental active regions.
    min_region_frames = max(
        1,
        int(round(MIN_SPEECH_REGION_MS / HOP_MS))
    )

    active = remove_short_active_regions(
        active,
        min_region_frames
    )

    active_indices = np.where(active)[0]

    if len(active_indices) == 0:
        return {
            "filename": path.name,
            "duration_sec": round(duration_sec, 4),
            "noise_floor_db": round(noise_floor_db, 2),
            "threshold_db": round(threshold_db, 2),
            "speech_onset_sec": "",
            "speech_offset_sec": "",
            "speech_span_sec": "",
            "leading_silence_sec": "",
            "trailing_silence_sec": "",
            "detected": False,
        }

    first_frame = int(active_indices[0])
    last_frame = int(active_indices[-1])

    onset_sec = (
        first_frame * hop_samples / sample_rate
    )

    # Include the complete final analysis frame.
    offset_sec = (
        last_frame * hop_samples / sample_rate
        + frame_samples / sample_rate
    )

    offset_sec = min(offset_sec, duration_sec)

    span_sec = offset_sec - onset_sec

    leading_silence_sec = onset_sec

    trailing_silence_sec = max(
        0.0,
        duration_sec - offset_sec
    )

    return {
        "filename": path.name,
        "duration_sec": round(duration_sec, 4),
        "noise_floor_db": round(noise_floor_db, 2),
        "threshold_db": round(threshold_db, 2),
        "speech_onset_sec": round(onset_sec, 4),
        "speech_offset_sec": round(offset_sec, 4),
        "speech_span_sec": round(span_sec, 4),
        "leading_silence_sec": round(leading_silence_sec, 4),
        "trailing_silence_sec": round(trailing_silence_sec, 4),
        "detected": True,
    }


# ============================================================
# Main
# ============================================================

def main():

    files = sorted(INPUT_DIR.glob("*.wav"))

    if not files:
        raise SystemExit(
            f"No WAV files found in:\n{INPUT_DIR}"
        )

    print(f"Found {len(files)} positive WAV files.")
    print()

    results = []
    errors = []

    for index, path in enumerate(files, start=1):

        try:
            result = analyze_file(path)
            results.append(result)

            print(
                f"[{index:3d}/{len(files)}] "
                f"{path.name} -> "
                f"span={result['speech_span_sec']} s"
            )

        except Exception as e:
            errors.append({
                "filename": path.name,
                "error": str(e)
            })

            print(
                f"[{index:3d}/{len(files)}] "
                f"ERROR: {path.name}: {e}"
            )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "filename",
        "duration_sec",
        "noise_floor_db",
        "threshold_db",
        "speech_onset_sec",
        "speech_offset_sec",
        "speech_span_sec",
        "leading_silence_sec",
        "trailing_silence_sec",
        "detected",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Files found:     {len(files)}")
    print(f"Successfully analyzed: {len(results)}")
    print(f"Errors:          {len(errors)}")
    print()
    print(f"CSV written to:")
    print(OUTPUT_CSV)

    # Summary statistics
    spans = np.array([
        float(r["speech_span_sec"])
        for r in results
        if r["detected"]
    ])

    if len(spans):

        print()
        print("Detected speech-span distribution")
        print("-" * 40)
        print(f"Count:   {len(spans)}")
        print(f"Minimum: {np.min(spans):.3f} s")
        print(f"25th:    {np.percentile(spans, 25):.3f} s")
        print(f"Median:  {np.percentile(spans, 50):.3f} s")
        print(f"75th:    {np.percentile(spans, 75):.3f} s")
        print(f"90th:    {np.percentile(spans, 90):.3f} s")
        print(f"95th:    {np.percentile(spans, 95):.3f} s")
        print(f"99th:    {np.percentile(spans, 99):.3f} s")
        print(f"Maximum: {np.max(spans):.3f} s")

    if errors:
        print()
        print("Errors:")
        for error in errors:
            print(
                f"  {error['filename']}: "
                f"{error['error']}"
            )


if __name__ == "__main__":
    main()