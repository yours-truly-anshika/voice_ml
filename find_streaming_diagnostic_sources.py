from pathlib import Path
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"


MIN_DURATION_SEC = 2.5
MAX_DURATION_SEC = 4.0


def inspect_wav(path):
    try:
        info = sf.info(path)
        duration = info.frames / info.samplerate

        return {
            "path": path,
            "duration": duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
        }

    except Exception:
        return None


def main():
    print("=" * 70)
    print("SEARCHING FOR FULL-LENGTH UNKNOWN/BACKGROUND SOURCE AUDIO")
    print("=" * 70)

    print(f"\nDataset root:")
    print(f"  {DATASET_DIR}")

    candidates = []

    for path in DATASET_DIR.rglob("*.wav"):

        result = inspect_wav(path)

        if result is None:
            continue

        if MIN_DURATION_SEC <= result["duration"] <= MAX_DURATION_SEC:
            candidates.append(result)

    if not candidates:
        print("\nNo WAV files in the requested duration range were found.")
        return

    candidates.sort(key=lambda x: str(x["path"]).lower())

    print(
        f"\nFound {len(candidates)} WAV files between "
        f"{MIN_DURATION_SEC:.1f}s and {MAX_DURATION_SEC:.1f}s."
    )

    print("\nDIRECTORY SUMMARY")
    print("-" * 70)

    directory_counts = {}

    for item in candidates:
        directory = item["path"].parent

        if directory not in directory_counts:
            directory_counts[directory] = 0

        directory_counts[directory] += 1

    for directory, count in sorted(
        directory_counts.items(),
        key=lambda item: str(item[0]).lower(),
    ):
        print(f"{count:5d}  {directory}")

    print("\nSAMPLE FILES")
    print("-" * 70)

    for item in candidates[:50]:
        print(
            f"{item['duration']:6.3f}s  "
            f"{item['sample_rate']:5d} Hz  "
            f"{item['channels']} ch  "
            f"{item['path']}"
        )

    if len(candidates) > 50:
        print(
            f"\n... and {len(candidates) - 50} more files."
        )


if __name__ == "__main__":
    main()