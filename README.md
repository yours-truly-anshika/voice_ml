# Voice ML

A Python research repository for developing a low-latency voice keyword-spotting pipeline for edge devices. The current project focuses on detecting the wake phrase **“activate orbit”** using three audio classes: `positive`, `unknown`, and `background`.

> **Project status:** Data preparation, audio analysis, feature extraction, and a TensorFlow DS-CNN training path are included. Edge firmware, quantized deployment artifacts, and a production inference application are not included yet.

## Goals

- Build a speaker-independent keyword-spotting dataset.
- Analyze speech endpoints and recording quality before windowing.
- Convert audio into log-Mel spectrogram features.
- Train a compact depthwise-separable convolutional neural network (DS-CNN).
- Prepare a model suitable for future edge-device deployment.

## Repository structure

```text
.
├── dataset/
│   ├── raw/          # Original recordings and external source data
│   ├── processed/    # Converted audio, manifests, and analysis outputs
│   └── final/        # Train/validation/test audio grouped by class
├── models/
│   └── dscnn.py      # DS-CNN model definition
├── feature_frontend.py
├── extract_features.py
├── train.py
├── endpoint_detection.py
├── check_audio.py
├── analysis.py
├── analyze_*.py       # Dataset and experiment analysis scripts
├── compare_*.py      # Comparison and validation utilities
└── context.md         # Detailed project handoff and data notes
```

The dataset is organized into these classes:

- `positive`: recordings of the target wake phrase.
- `unknown`: speech that is not the target phrase.
- `background`: environmental noise and non-speech background audio.

## Audio and feature configuration

The feature frontend currently expects mono, 16 kHz audio and supports 1.0, 1.5, and 2.0 second windows. It computes:

- 40 Mel bands
- 30 ms Hann analysis window
- 20 ms hop length
- 512-point FFT
- 20 Hz–8 kHz frequency range
- Log-scaled Mel power features

For a 1-second window, the expected model input shape is `(40, 49, 1)`.

> **Important:** Many files in the prepared dataset are approximately 3 seconds long, while the current frontend expects a fixed-duration window. Define and apply an explicit 3-second-to-model-window policy before extracting production training features; blindly truncating files can remove the wake phrase.

## Installation

Python 3.8 or newer is recommended. Create an isolated environment and install the packages used by the pipeline:

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn librosa soundfile tensorflow
```

`endpoint_detection.py` additionally requires `ffmpeg` and `ffprobe` available on your `PATH`.

## Usage

### Inspect the collected recordings

```bash
python analysis.py
```

### Analyze speech endpoints

```bash
python endpoint_detection.py \
  --input_dir dataset/processed/positive \
  --output_csv dataset/processed/positive_endpoints.csv \
  --noise_db -35 \
  --min_silence_dur 0.2
```

On Windows, use equivalent paths such as `dataset\\processed\\positive`.

### Extract log-Mel features

Feature extraction works on windowed audio under `dataset/windowed_<duration>s/` and writes NumPy features under `dataset/features_<duration>s/`:

```bash
python extract_features.py --duration 1.0 --dry-run
python extract_features.py --duration 1.0
```

Use `--limit N` to process only a limited number of files per class while testing the pipeline.

### Inspect the DS-CNN

```bash
python models/dscnn.py
```

### Train a duration-specific model

Training expects a prepared file at `dataset/features_<duration>s/dataset_features.npz` containing `X_train`, `y_train`, `X_val`, and `y_val`:

```bash
python train.py --duration 1.0
```

The best checkpoint is written to `models/dscnn_<duration>s_best.keras`.

## Model

The included DS-CNN uses:

1. An initial 2D convolution.
2. Depthwise-separable convolution blocks.
3. Batch normalization and ReLU activations.
4. Global average pooling.
5. A three-class softmax output.

The default class order used by the dataset pipeline is:

```text
positive, unknown, background
```

Keep this order consistent when preparing labels and interpreting predictions.

## Dataset and evaluation notes

- Train, validation, and test data should remain speaker-independent.
- Validation and test data should not contain synthetic derivatives of training recordings.
- Dataset manifests under `dataset/processed/` are important for reproducibility and auditing.
- Preserve raw recordings and avoid destructive modifications.
- Do not commit personally identifying participant data, credentials, tokens, or other sensitive information to a public repository.

## Reproducibility

Several scripts use the project seed `26172`. When adding experiments, record the preprocessing configuration, window policy, class mapping, random seed, and manifest used for the run.

## Current limitations and next steps

- Finalize the fixed-window policy for approximately 3-second recordings.
- Add a documented dependency lock file or `requirements.txt`.
- Add automated tests for audio validation and feature shapes.
- Add evaluation metrics beyond validation accuracy, including per-class precision, recall, F1 score, and confusion matrices.
- Export and validate a quantized model for the target edge platform.
- Add on-device inference and latency/memory benchmarks.

## License

No license has been specified yet. Add a `LICENSE` file before distributing this project.

## Acknowledgements

The project uses the Speech Commands dataset for unknown speech and background-noise sources. Review and comply with the applicable dataset terms before redistribution.
