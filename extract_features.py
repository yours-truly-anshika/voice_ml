from pathlib import Path

import numpy as np
import soundfile as sf

from feature_frontend import extract_log_mel, SAMPLE_RATE


INPUT_ROOT = Path("dataset/windowed")
OUTPUT_ROOT = Path("dataset/features")

SPLITS = ["train", "val", "test"]
CLASSES = ["positive", "unknown", "background"]


def main():
    total = 0

    for split in SPLITS:
        for class_name in CLASSES:

            input_dir = INPUT_ROOT / split / class_name
            output_dir = OUTPUT_ROOT / split / class_name

            output_dir.mkdir(parents=True, exist_ok=True)

            wav_files = sorted(input_dir.glob("*.wav"))

            print(f"\n{split}/{class_name}: {len(wav_files)} WAV files")

            for wav_path in wav_files:

                audio, sample_rate = sf.read(
                    wav_path,
                    dtype="float32",
                    always_2d=False,
                )

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

                if len(audio) != SAMPLE_RATE:
                    raise ValueError(
                        f"{wav_path}: expected {SAMPLE_RATE} samples, "
                        f"got {len(audio)}"
                    )

                features = extract_log_mel(audio)

                if features.shape != (40, 49, 1):
                    raise ValueError(
                        f"{wav_path}: expected feature shape "
                        f"(40, 49, 1), got {features.shape}"
                    )

                if features.dtype != np.float32:
                    raise ValueError(
                        f"{wav_path}: expected float32 features, "
                        f"got {features.dtype}"
                    )

                output_path = output_dir / f"{wav_path.stem}.npy"

                np.save(output_path, features)

                total += 1

    print("\nFEATURE EXTRACTION COMPLETE")
    print(f"Total feature files: {total}")


if __name__ == "__main__":
    main()