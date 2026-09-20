import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import sys

from feature_frontend import extract_log_mel, SAMPLE_RATE

SPLITS = ["train", "val", "test"]
CLASSES = ["positive", "unknown", "background"]

SHAPE_MAP = {
    1.0: (40, 49, 1),
    1.5: (40, 74, 1),
    2.0: (40, 99, 1)
}

def main():
    parser = argparse.ArgumentParser(description="Extract features for a specific window duration")
    parser.add_argument("--duration", type=float, required=True, choices=[1.0, 1.5, 2.0], help="Duration of the window in seconds")
    # For dry-run/testing
    parser.add_argument("--dry-run", action="store_true", help="Print expected actions without saving")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files to process per class (0 for no limit)")
    args = parser.parse_args()

    duration = args.duration
    expected_samples = int(duration * SAMPLE_RATE)
    expected_shape = SHAPE_MAP[duration]

    input_root = Path(f"dataset/windowed_{duration}s")
    output_root = Path(f"dataset/features_{duration}s")

    if not input_root.exists():
        print(f"Error: Input root {input_root} does not exist.")
        sys.exit(1)

    total = 0

    for split in SPLITS:
        for class_name in CLASSES:
            input_dir = input_root / split / class_name
            output_dir = output_root / split / class_name

            if not args.dry_run:
                output_dir.mkdir(parents=True, exist_ok=True)

            if not input_dir.exists():
                continue

            wav_files = sorted(input_dir.glob("*.wav"))
            print(f"\n{split}/{class_name}: {len(wav_files)} WAV files")

            count = 0
            for wav_path in wav_files:
                if args.limit > 0 and count >= args.limit:
                    break

                try:
                    audio, sample_rate = sf.read(
                        wav_path,
                        dtype="float32",
                        always_2d=False,
                    )
                except Exception as e:
                    print(f"Skipping {wav_path}: {e}")
                    continue

                if sample_rate != SAMPLE_RATE:
                    raise ValueError(
                        f"{wav_path}: expected {SAMPLE_RATE} Hz, "
                        f"got {sample_rate} Hz"
                    )

                if audio.ndim != 1:
                    raise ValueError(
                        f"{wav_path}: expected mono audio, "
                        f"got shape {audio.shape}"
                    )

                if len(audio) != expected_samples:
                    raise ValueError(
                        f"{wav_path}: expected {expected_samples} samples, "
                        f"got {len(audio)}"
                    )

                features = extract_log_mel(audio)

                if features.shape != expected_shape:
                    raise ValueError(
                        f"{wav_path}: expected feature shape "
                        f"{expected_shape}, got {features.shape}"
                    )

                if features.dtype != np.float32:
                    raise ValueError(
                        f"{wav_path}: expected float32 features, "
                        f"got {features.dtype}"
                    )

                output_path = output_dir / f"{wav_path.stem}.npy"

                if not args.dry_run:
                    np.save(output_path, features)

                total += 1
                count += 1

    print("\nFEATURE EXTRACTION COMPLETE")
    print(f"Total feature files processed: {total}")

if __name__ == "__main__":
    main()