from pathlib import Path
import contextlib
import io

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf

from feature_frontend import extract_log_mel


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "models" / "dscnn_best.keras"
MANIFEST_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "positive_endpoint_with_split.csv"
)
POSITIVE_DIR = PROJECT_ROOT / "dataset" / "processed" / "positive"

OUTPUT_DIR = PROJECT_ROOT / "dataset" / "processed"
WINDOW_RESULTS_PATH = OUTPUT_DIR / "streaming_positive_predictions.csv"
SUMMARY_PATH = OUTPUT_DIR / "streaming_positive_summary.csv"

SAMPLE_RATE = 16000
WINDOW_SAMPLES = SAMPLE_RATE
HOP_SAMPLES = int(0.10 * SAMPLE_RATE)  # 100 ms

CLASS_NAMES = [
    "positive",
    "unknown",
    "background",
]


# ============================================================
# Helpers
# ============================================================

def load_audio(path):
    audio, sample_rate = sf.read(path, dtype="float32")

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"{path.name}: expected {SAMPLE_RATE} Hz, got {sample_rate} Hz"
        )

    return audio


def make_windows(audio):
    """
    Generate all complete 1-second windows using a 100 ms hop.
    """
    windows = []
    starts = []

    if len(audio) < WINDOW_SAMPLES:
        return np.empty((0, WINDOW_SAMPLES), dtype=np.float32), []

    for start in range(
        0,
        len(audio) - WINDOW_SAMPLES + 1,
        HOP_SAMPLES,
    ):
        end = start + WINDOW_SAMPLES
        windows.append(audio[start:end])
        starts.append(start / SAMPLE_RATE)

    return np.asarray(windows, dtype=np.float32), starts


def extract_features(windows):
    """
    Use the project's existing feature frontend exactly as-is.
    Suppress its diagnostic print for this batch operation.
    """
    features = []

    for window in windows:
        with contextlib.redirect_stdout(io.StringIO()):
            feature = extract_log_mel(window)

        features.append(feature)

    return np.asarray(features, dtype=np.float32)


# ============================================================
# Main diagnostic
# ============================================================

def main():
    print("=" * 70)
    print("STREAMING POSITIVE PREDICTION DIAGNOSTIC")
    print("=" * 70)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    if not POSITIVE_DIR.exists():
        raise FileNotFoundError(f"Positive directory not found: {POSITIVE_DIR}")

    print(f"\nLoading model:")
    print(f"  {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)

    print("\nModel loaded.")

    manifest = pd.read_csv(MANIFEST_PATH)

    # Use all positive recordings for this diagnostic.
    manifest = manifest[
        manifest["Split"].astype(str).str.lower().isin(
            ["train", "val", "test"]
        )
    ].copy()

    print(f"\nPositive recordings: {len(manifest)}")
    print(
        manifest["Split"]
        .astype(str)
        .str.lower()
        .value_counts()
        .sort_index()
        .to_string()
    )

    all_rows = []
    summary_rows = []

    for record_index, row in manifest.iterrows():

        filename = str(row["Filename"])
        split = str(row["Split"]).lower()

        audio_path = POSITIVE_DIR / filename

        if not audio_path.exists():
            print(f"\nWARNING: missing file: {audio_path}")
            continue

        audio = load_audio(audio_path)

        windows, starts = make_windows(audio)

        if len(windows) == 0:
            print(f"\nWARNING: no complete 1-second windows: {filename}")
            continue

        features = extract_features(windows)

        predictions = model.predict(
            features,
            verbose=0,
        )

        predicted_classes = np.argmax(predictions, axis=1)

        positive_confidences = predictions[:, 0]

        best_index = int(np.argmax(positive_confidences))

        best_positive_conf = float(
            positive_confidences[best_index]
        )

        best_positive_start = float(
            starts[best_index]
        )

        positive_dominant_count = int(
            np.sum(predicted_classes == 0)
        )

        print(
            f"\n{split.upper():4s}  {filename}"
        )
        print(
            f"  duration: {len(audio) / SAMPLE_RATE:.3f} s"
        )
        print(
            f"  windows:  {len(windows)}"
        )
        print(
            f"  best positive confidence: "
            f"{best_positive_conf:.4f}"
        )
        print(
            f"  best positive start: "
            f"{best_positive_start:.2f} s"
        )
        print(
            f"  positive-dominant windows: "
            f"{positive_dominant_count}"
        )

        for window_index, start in enumerate(starts):

            probs = predictions[window_index]

            predicted_index = int(
                predicted_classes[window_index]
            )

            all_rows.append(
                {
                    "Split": split,
                    "Filename": filename,
                    "WindowIndex": window_index,
                    "StartSec": round(start, 3),
                    "EndSec": round(start + 1.0, 3),
                    "PositiveConfidence": float(probs[0]),
                    "UnknownConfidence": float(probs[1]),
                    "BackgroundConfidence": float(probs[2]),
                    "PredictedClass": CLASS_NAMES[predicted_index],
                }
            )

        summary_rows.append(
            {
                "Split": split,
                "Filename": filename,
                "DurationSec": len(audio) / SAMPLE_RATE,
                "WindowCount": len(windows),
                "PositiveDominantWindows": positive_dominant_count,
                "BestPositiveConfidence": best_positive_conf,
                "BestPositiveStartSec": best_positive_start,
            }
        )

    results_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(summary_rows)

    results_df.to_csv(
        WINDOW_RESULTS_PATH,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)

    print(f"\nWindow-level results:")
    print(f"  {WINDOW_RESULTS_PATH}")

    print(f"\nRecording-level summary:")
    print(f"  {SUMMARY_PATH}")

    print("\nBest positive confidence by split:")

    print(
        summary_df.groupby("Split")["BestPositiveConfidence"]
        .agg(["count", "mean", "median", "min", "max"])
        .round(4)
        .to_string()
    )

    print("\nNumber of positive-dominant windows by split:")

    print(
        summary_df.groupby("Split")["PositiveDominantWindows"]
        .agg(["count", "mean", "median", "min", "max"])
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()