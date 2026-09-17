# SIH26172 Technical Handoff Context

This file was created for a future AI assistant continuing the project from a fresh chat.

Repository inspected: `C:\Users\Anshika\voice_ml`

Creation date in active environment: 2026-09-17

Rule for this document:

- `VERIFIED FACT`: directly checked from repository files, scripts, manifests, directories, command output, or computed hashes.
- `PROJECT DECISION`: project direction supplied in the handoff request or encoded in scripts/manifests. If not independently provable from the repository, it is explicitly marked.
- `DESIGN INTENT`: intended architecture or rationale from comments/request text, not necessarily implemented.
- `ASSUMPTION`: useful working assumption that must be rechecked before implementation.
- `TODO / NOT YET DONE`: missing, incomplete, or unresolved.
- `NOT VERIFIED FROM REPOSITORY`: the repository did not contain enough evidence.

Important privacy note:

- `dataset\raw\participants.csv` contains personally identifying columns including `full_name`, `phone_number`, and `auth_user_id`. This handoff intentionally documents column names and counts but does not reproduce names, phone numbers, or auth user IDs.

## Project Identity

- PROJECT DECISION / USER-SUPPLIED, NOT VERIFIED FROM REPOSITORY: SIH year = `2026`.
- VERIFIED FACT: repository scripts/comments reference SIH problem ID `SIH26172` in `generate_synthetic_positive.py` and `select_unknown_and_background.py`.
- PROJECT DECISION / USER-SUPPLIED, NOT VERIFIED FROM REPOSITORY: SIH problem title = `Low Latency and Efficient Voice Activator for Edge Devices`.
- PROJECT DECISION / USER-SUPPLIED, NOT VERIFIED FROM REPOSITORY: team name = `Quintus`.
- VERIFIED FACT: `select_unknown_and_background.py` comment says unknown speech is distinct from the target phrase `activate orbit`.
- PROJECT DECISION: wake word / keyword = `activate orbit`.
- DESIGN INTENT: build a low-latency keyword spotting system for edge devices. Local KWS detects `activate orbit`; after wake-word detection, downstream ASR/command handling can run.
- DESIGN INTENT / NOT VERIFIED FROM REPOSITORY: intended target is ESP32-S3 edge deployment.
- DESIGN INTENT / NOT VERIFIED FROM REPOSITORY: keyword spotting should run locally to avoid continuously streaming audio, reduce latency, reduce network dependency, and avoid continuously running ASR on constrained hardware.
- TODO / NOT YET DONE: final on-device firmware, quantized model, and ESP32 deployment are not present in this repository.

## Repository Scope

VERIFIED FACT: this workspace is not a Git repository. `git status --short` returned `fatal: not a git repository`.

VERIFIED FACT: top-level project files/directories include:

- `analysis.py`
- `analyze_positive_endpoints.py`
- `check_audio.py`
- `check_sc_partitions.py`
- `feature_frontend.py`
- `find_split.py`
- `freeze_split.py`
- `generate_synthetic_background.py`
- `generate_synthetic_positive.py`
- `generate_synthetic_unknown.py`
- `select_unknown_and_background.py`
- `select_unknown_speech.py`
- `split_audit.txt`
- `verify_split_manifest.py`
- `waveform_test.png`
- `dataset\`
- `venv\`
- `.venv\`
- `__pycache__\`

VERIFIED FACT: no `models` directory exists in this workspace. `models\dscnn.py` is not present.

VERIFIED FACT: no ESP-IDF project files were found by targeted search for `CMakeLists.txt`, `sdkconfig`, `sdkconfig.defaults`, `partitions.csv`, or `idf_component.yml` outside virtualenv and raw external dataset.

VERIFIED FACT: no web app project files such as `package.json` were found outside virtualenv and raw external dataset.

## Hardware / Edge Architecture

- ESP32-S3: PROJECT DECISION / USER-SUPPLIED, NOT VERIFIED FROM REPOSITORY.
- ESP-IDF version: NOT VERIFIED FROM REPOSITORY.
- ESP-IDF project name: NOT VERIFIED FROM REPOSITORY.
- ESP-IDF build status: NOT VERIFIED FROM REPOSITORY.
- Bluetooth status: NOT VERIFIED FROM REPOSITORY.
- IPv6 status: NOT VERIFIED FROM REPOSITORY.
- Edge memory/latency requirements: NOT VERIFIED FROM REPOSITORY.
- Microphone assumptions: NOT VERIFIED FROM REPOSITORY.
- Audio format expected by current dataset/frontend: VERIFIED FACT: mono, 16 kHz WAV for processed/final/synthetic artifacts.
- Channels: VERIFIED FACT: final and synthetic WAVs are mono.
- KWS expected model input: VERIFIED FACT: current frontend outputs `(40, 49, 1)` log-Mel tensors from exactly 1 second of 16 kHz audio.
- ASR/Vosk architecture: VERIFIED FACT: `venv` contains `vosk==0.3.45`, but repository code does not implement ASR flow. Any ASR/Vosk edge/cloud split is NOT VERIFIED FROM REPOSITORY.
- DESIGN INTENT / NOT VERIFIED FROM REPOSITORY: ASR should not run continuously on the edge; KWS gates it after wake-word detection.

## Machine Learning Architecture

- KWS task: PROJECT DECISION: classify audio for wake word detection around `activate orbit`.
- Class count: VERIFIED FACT from manifests = 3 classes.
- Exact class names: VERIFIED FACT: `positive`, `unknown`, `background`.
- Positive class meaning: PROJECT DECISION: target keyword phrase `activate orbit`.
- Unknown class meaning: VERIFIED FACT: selected Speech Commands spoken words distinct from the target phrase.
- Background class meaning: VERIFIED FACT: background/noise segments.
- DS-CNN model decision: PROJECT DECISION / USER-SUPPLIED, NOT VERIFIED FROM REPOSITORY. No DS-CNN implementation file is present.
- Model architecture: NOT VERIFIED FROM REPOSITORY.
- Training status: TODO / NOT YET DONE. No training script, trained model file, TensorFlow/Keras/PyTorch checkpoint, or TFLite file was found.
- Deployment status: TODO / NOT YET DONE. No firmware or model deployment artifact found.
- ML frameworks actually present in `venv`: VERIFIED FACT: `librosa==0.11.0`, `numpy==2.4.6`, `pandas==3.0.5`, `scikit-learn==1.9.1`, `scipy==1.17.1`, `soundfile==0.14.0`, `vosk==0.3.45`.
- TensorFlow/PyTorch/Keras: VERIFIED FACT: not present in `venv\Scripts\python.exe -m pip freeze`.

## Existing Feature Frontend

Source file: `feature_frontend.py`

VERIFIED FACT: exact configuration:

- `SAMPLE_RATE = 16000`
- `N_MELS = 40`
- `FRAME_LENGTH_MS = 30`
- `FRAME_STEP_MS = 20`
- `FRAME_LENGTH = int(16000 * 30 / 1000) = 480`
- `FRAME_STEP = int(16000 * 20 / 1000) = 320`
- `N_FFT = 512`
- window type: `"hann"`
- `center=False`
- `fmin=20`
- `fmax=SAMPLE_RATE // 2 = 8000`
- power: `2.0`
- librosa function: `librosa.feature.melspectrogram(...)`
- dB conversion: `librosa.power_to_db(mel, ref=np.max)`
- dtype conversion: input converted with `np.asarray(audio, dtype=np.float32)`
- target length: `target_length = SAMPLE_RATE`

VERIFIED FACT: behavior:

1. `extract_log_mel(audio)` converts input to `float32`.
2. It forces exactly 1 second of audio because `target_length = SAMPLE_RATE`.
3. If `len(audio) < 16000`, it pads zeros at the end.
4. If `len(audio) >= 16000`, it truncates to the first 16000 samples.
5. It computes a Mel spectrogram with `n_fft=512`, `hop_length=320`, `win_length=480`, Hann window, no centering, 40 Mel bins, 20 Hz to 8 kHz.
6. It converts power Mel values to dB using `ref=np.max`.
7. It prints feature shape.
8. It returns `log_mel[..., np.newaxis]`.

VERIFIED FACT: running `.\venv\Scripts\python.exe feature_frontend.py` printed:

```text
Feature shape: (40, 49)
Final tensor shape: (40, 49, 1)
```

VERIFIED FACT / IMPORTANT MISMATCH:

- The docstring says "49 x 40 x 1".
- The inline comment says `# Expected: (49, 40)`.
- Actual repository-verified output is `(40, 49)` and final tensor shape `(40, 49, 1)`.

CRITICAL PIPELINE FACT:

- Current frontend processes exactly 1 second, not 3 seconds.
- It must NOT be assumed to correctly handle the current approximately 3-second dataset until a deliberate 3 s -> 1 s windowing policy is designed.
- Naively applying it to 3-second WAVs will keep only the first 1 second and discard the remainder.

## Data Collection Web App

VERIFIED FACT: this repository contains exported data CSVs:

- `dataset\raw\participants.csv`
- `dataset\raw\voice_notes.csv`

VERIFIED FACT: `participants.csv` has 64 rows and columns:

- `id`
- `auth_user_id`
- `full_name`
- `phone_number`
- `nda_agreed_at`
- `consent_agreed_at`
- `nda_version`
- `consent_version`
- `created_at`

VERIFIED FACT: `voice_notes.csv` has 291 rows, 51 unique `participant_id` values, and columns:

- `id`
- `participant_id`
- `storage_path`
- `note_type`
- `sequence_number`
- `duration_ms`
- `created_at`
- `variation_tag`

VERIFIED FACT: `voice_notes.csv` `note_type` counts:

- `mandatory`: 243
- `optional`: 48

VERIFIED FACT: `voice_notes.csv` `variation_tag` counts:

- `normal`: 63
- `faster`: 58
- `slower`: 57
- `intonation`: 57
- `distance`: 56

VERIFIED FACT: `voice_notes.csv` `storage_path` values follow paths like `<participant_id>/<note_type>/<sequence_number>.webm`.

VERIFIED FACT: `dataset\raw\audio` preserves 173 `.webm` files plus 16 `.emptyFolderPlaceholder` files.

NOT VERIFIED FROM REPOSITORY:

- deployed application URL
- web app source repository
- Supabase project settings
- Supabase bucket name
- authentication implementation details
- Email OTP behavior
- RLS policies
- Vercel deployment
- Resend SMTP configuration
- environment variable names or values

PROJECT DECISION / PRIVACY:

- Do not expose participant names, phone numbers, auth IDs, API keys, tokens, passwords, or private credentials in future handoffs/logs.

## Raw and Processed Positive Data

VERIFIED FACT: raw positive recordings:

- `dataset\raw\audio`: 173 `.webm` files.
- `dataset\processed\audio_wav`: 173 `.wav` files, still organized by participant/type directories.
- `dataset\processed\positive`: 173 flattened `.wav` files.
- `dataset\processed\dataset_manifest.csv`: 173 rows.
- `dataset\processed\positive_review_manifest.csv`: 173 rows.
- `dataset\processed\positive_endpoint_analysis.csv`: 173 rows.

VERIFIED FACT: `dataset_manifest.csv` columns:

- `ParticipantID`
- `NoteType`
- `SequenceNumber`
- `Filename`
- `FullPath`

VERIFIED FACT: `dataset_manifest.csv` contains 37 unique participants, 132 mandatory rows, and 41 optional rows.

VERIFIED FACT: `positive_review_manifest.csv` columns:

- `file_path`
- `participant_id`
- `note_type`
- `filename`
- `decision`
- `reason`

VERIFIED FACT: all 173 `decision` values and all 173 `reason` values are blank.

VERIFIED FACT: `positive_endpoint_analysis.csv` columns:

- `filename`
- `duration_sec`
- `noise_floor_db`
- `threshold_db`
- `speech_onset_sec`
- `speech_offset_sec`
- `speech_span_sec`
- `leading_silence_sec`
- `trailing_silence_sec`
- `detected`

VERIFIED FACT: no conversion script from WebM to WAV was found. The repository verifies separated raw and converted artifacts, not the exact command used for conversion.

PROJECT DECISION: original recordings must never be modified destructively.

## Positive Training Statistics

These values were recomputed from repository files on 2026-09-17 using `venv` packages and sample standard deviation (`ddof=1`) where a standard deviation is reported.

Actual training WAV durations from `dataset\final\train\positive`:

- count: 121
- mean: 2.886 s
- std: 0.313 s
- min: 1.440 s
- Q25: 2.940 s
- median: 3.000 s
- Q75: 3.000 s
- max: 3.040 s
- `<1 sec`: 0
- `>=2 sec`: 116
- `>=2.5 sec`: 109
- `>=3 sec`: 89

Endpoint/speech span for training positives, joining `split_manifest.csv` train rows to `positive_endpoint_analysis.csv`:

- rows: 121
- detected: 121
- speech span mean: 1.871 s
- std: 0.592 s
- min: 0.890 s
- Q25: 1.320 s
- median: 1.860 s
- Q75: 2.290 s
- max: 3.020 s
- `>=1.0 s`: 116/121
- `<1.0 s`: 5/121

Leading silence:

- mean: 0.478 s
- std: 0.354 s
- min: 0.000 s
- Q25: 0.090 s
- median: 0.500 s
- Q75: 0.760 s
- max: 1.230 s

Trailing silence:

- mean: 0.537 s
- std: 0.449 s
- min: 0.000 s
- Q25: 0.110 s
- median: 0.480 s
- Q75: 0.880 s
- max: 1.890 s

Whole-file audio levels for training positives:

- peak mean: 0.6752
- peak std: 0.3146
- peak min: 0.0530
- peak Q25: 0.4009
- peak median: 0.7432
- peak Q75: 0.9897
- peak max: 1.0
- RMS mean: 0.0680
- RMS std: 0.0402
- RMS min: 0.0043
- RMS Q25: 0.0331
- RMS median: 0.0609
- RMS Q75: 0.0951
- RMS max: 0.1901
- RMS dB mean: -25.16
- RMS dB std: 6.18
- RMS dB min: -47.31
- RMS dB Q25: -29.60
- RMS dB median: -24.31
- RMS dB Q75: -20.44
- RMS dB max: -14.42
- clipping fraction mean: 0.0000166 repository-computed; handoff prompt supplied 0.000018
- clipping fraction std: 0.0000516 repository-computed; handoff prompt supplied 0.000055
- clipping fraction max: 0.000354
- files with some clipping: 25/121 repository-computed; handoff prompt supplied 26/121
- files with peak >=0.99: 30/121

Speech-region RMS:

- RMS mean: 0.08466
- std: 0.04689
- min: 0.00548
- Q25: 0.04895
- median: 0.08033
- Q75: 0.11629
- max: 0.21339
- RMS dB mean: -23.09
- std: 5.89
- min: -45.22
- Q25: -26.20
- median: -21.90
- Q75: -18.69
- max: -13.42

PROJECT DECISION / DESIGN INTENT:

- Do not blindly apply positive gain.
- Use conservative gain with clipping protection.
- Background should be mixed by target SNR rather than raw amplitude.

## Speaker-Independent Split

AUTHORITATIVE / FROZEN ARTIFACT:

- `dataset\processed\split_manifest.csv`
- SHA-256: `0989224592842A34D9958198E8218B7CE0114989F09B05FF5E81CE51C6F09CE8`

VERIFIED FACT: `split_manifest.csv` columns:

- `ParticipantID`
- `NoteType`
- `SequenceNumber`
- `Filename`
- `FullPath`
- `Split`

VERIFIED FACT: split counts:

- train recordings: 121
- validation recordings: 26
- test recordings: 26
- percentages from `split_audit.txt`: train 69.94%, val 15.03%, test 15.03%

VERIFIED FACT: speakers:

- train speakers: 25
- val speakers: 6
- test speakers: 6
- speaker overlap: 0

VERIFIED FACT: mandatory/optional by split:

- train: 86 mandatory / 35 optional
- val: 26 mandatory / 0 optional
- test: 20 mandatory / 6 optional

VERIFIED FACT: train participant IDs:

- `02c153b8-8fb9-4528-8158-8049e257db88`
- `15c65680-cfe5-44e4-ac9b-f8ccd940653d`
- `1c231685-30ef-4e9a-a881-ed8b63c6f1b6`
- `2a24ad90-f292-4321-9baf-081583734e42`
- `2a69429e-6de8-4410-8267-e2185fe3809f`
- `2b017ac3-bbec-4db3-8402-8f942b24becd`
- `2c214cfb-1bf1-4660-b74b-d804bbfc73d4`
- `2d70cdb6-2e97-4ac9-ae60-daebf846ea60`
- `2e2da424-76f9-4afd-ae75-12a27e00ac73`
- `335b6d7c-5ded-4be0-b876-09b9508d2265`
- `4234fb4d-064c-484d-88aa-ef5be9c7eabf`
- `52499bbe-68bc-4e40-86d7-6121e9dfeb24`
- `5da6fe17-5ed0-43eb-bb94-8a61cc1c37f1`
- `73b51835-9a81-46f6-968e-6d3f2d58a323`
- `8e25f738-67de-47c3-8986-885244e13042`
- `a1f3bee7-082d-458e-8208-b3e808172cee`
- `aa91f7e3-e010-44eb-89ae-877dae296ad8`
- `baa8bfbe-be1f-4849-a601-67812dfe2775`
- `d30246ec-2c35-481e-a9e5-155d77fdd772`
- `d9fa7fff-de38-4adc-b1e3-c27fa4626dfa`
- `ddd732c0-fb06-4989-a4ff-e9664d7a7087`
- `ef518ed6-56c1-4a88-8afe-158ff08a2964`
- `efc91795-fdcf-4347-b99e-efaf84c2f9be`
- `f70d467e-8781-426e-80e9-a74519d645ea`
- `fe398b09-8619-4849-b515-8b11c24fc7e7`

VERIFIED FACT: validation participant IDs:

- `263875b0-b8cb-4e7f-a43a-d6e551ee48ed`
- `2d80737c-f830-4ec6-9dfe-a6fcc805d9ee`
- `47f64db2-16aa-4904-94c9-bab3f0095a28`
- `64de123c-53a4-4059-957c-92bf29ac468b`
- `943069ad-2757-493d-8f94-d7a295137152`
- `f5f4bb28-973c-4278-9967-38fb81c6a050`

VERIFIED FACT: test participant IDs:

- `054b38c0-8e14-4bd8-a649-2e2c5f3d389e`
- `2c574cb7-1d2b-440f-bf2f-6428bfe6bd07`
- `81fdbf2e-3612-4678-bf20-f0113528505c`
- `aca6354d-db66-444b-b95a-b2e6df9361ef`
- `c7d753f1-f749-4a37-8bd5-53dc9bab4fea`
- `da3b85a1-387b-40d5-a8af-ce10db3ac2c8`

VERIFIED FACT: `verify_split_manifest.py` checks that flattened WAVs exist under `dataset\processed\positive` and prints `STATUS: PASS`.

PROJECT DECISION:

- `split_manifest.csv` is authoritative and MUST NOT be regenerated casually.

## External Speech Commands Data

VERIFIED FACT:

- source dataset directory: `dataset\raw\external\speech_commands_v0.02`
- archive present: `dataset\raw\external\speech_commands_v0.02.tar.gz`
- total WAV files under extracted dataset: 105,835
- official validation list size: 9,981 lines
- official testing list size: 11,005 lines
- word-class directories: 35

VERIFIED FACT: word-class directories:

`backward`, `bed`, `bird`, `cat`, `dog`, `down`, `eight`, `five`, `follow`, `forward`, `four`, `go`, `happy`, `house`, `learn`, `left`, `marvin`, `nine`, `no`, `off`, `on`, `one`, `right`, `seven`, `sheila`, `six`, `stop`, `three`, `tree`, `two`, `up`, `visual`, `wow`, `yes`, `zero`

VERIFIED FACT: final selected unknown labels in `select_unknown_and_background.py`:

`backward`, `bed`, `bird`, `cat`, `dog`, `down`, `eight`, `five`, `four`, `go`, `happy`, `house`, `learn`, `left`, `marvin`, `nine`, `no`, `off`, `on`, `one`, `right`, `seven`, `sheila`, `six`, `stop`, `three`, `tree`, `two`, `up`, `yes`, `zero`

VERIFIED FACT: labels present in Speech Commands but excluded from the final 31-label unknown selection:

`follow`, `forward`, `visual`, `wow`

AUTHORITATIVE UNKNOWN MANIFEST:

- `dataset\processed\unknown_selection_manifest.csv`
- SHA-256: `060604817BA087EB2039BC38FE69A3905C2F12529CC10D7A013FD6EA1CC0626B`
- rows: 173
- columns: `src_path`, `label`, `speaker_id`, `split`

VERIFIED FACT: final unknown selection counts:

- train: 121
- val: 26
- test: 26
- train speakers: 114
- val speakers: 25
- test speakers: 23
- train/val speaker overlap: 0
- train/test speaker overlap: 0
- val/test speaker overlap: 0

VERIFIED FACT: deterministic seeds in `select_unknown_and_background.py`:

- default base seed: `26172`
- unknown train seed: `26172 + 101`
- unknown val seed: `26172 + 202`
- unknown test seed: `26172 + 303`

VERIFIED FACT: `select_unknown_speech.py` is an older/experimental script. It created `dataset\raw\unknown_speech` with 1,000 WAVs and `selection_manifest.csv` across all 35 words, including `follow`, `forward`, `visual`, and `wow`. This is NOT the authoritative final unknown selection.

## Background Data

VERIFIED FACT: original Speech Commands background sources:

- `dataset\raw\external\speech_commands_v0.02\_background_noise_\doing_the_dishes.wav`
- `dataset\raw\external\speech_commands_v0.02\_background_noise_\dude_miaowing.wav`
- `dataset\raw\external\speech_commands_v0.02\_background_noise_\exercise_bike.wav`
- `dataset\raw\external\speech_commands_v0.02\_background_noise_\pink_noise.wav`
- `dataset\raw\external\speech_commands_v0.02\_background_noise_\running_tap.wav`
- `dataset\raw\external\speech_commands_v0.02\_background_noise_\white_noise.wav`

VERIFIED FACT: preprocessed background segment directory:

- `dataset\raw\background_noise`
- 360 WAV files
- sources and filename prefixes: `bike`, `dishes`, `miaow`, `pink`, `tap`, `white`
- 60 sequential segments per source: indices `000` through `059`
- naming convention: `<source>_<segment_index>.wav`, e.g. `pink_033.wav`

AUTHORITATIVE BACKGROUND MANIFEST:

- `dataset\processed\background_selection_manifest.csv`
- SHA-256: `A89278620DEAECBFEEDBE8A67E349EBA5D1A13275641BC4B944132EFB44D2B76`
- rows: 173
- columns: `src_path`, `source`, `segment_index`, `split`

VERIFIED FACT: temporal split strategy encoded in `select_unknown_and_background.py`:

- 60 segments per source.
- first 70% -> train: indices `000` to `041` inclusive.
- next 15% -> val: indices `042` to `050` inclusive.
- final 15% -> test: indices `051` to `059` inclusive.
- Random subsampling happens inside each temporal block after the temporal split.

DESIGN INTENT:

- The temporal split deliberately reduces leakage from adjacent segments of the same continuous background recording.

VERIFIED FACT: final background selected counts:

- train: 121
- val: 26
- test: 26

VERIFIED FACT: background source distribution:

- train: `bike=20`, `dishes=20`, `miaow=20`, `pink=21`, `tap=20`, `white=20`
- val: `bike=4`, `dishes=5`, `miaow=5`, `pink=4`, `tap=4`, `white=4`
- test: `bike=5`, `dishes=5`, `miaow=4`, `pink=4`, `tap=4`, `white=4`

VERIFIED FACT: deterministic seeds in `select_unknown_and_background.py`:

- background train seed: `26172 + 404`
- background val seed: `26172 + 505`
- background test seed: `26172 + 606`

## Real-Only Final Dataset

AUTHORITATIVE BASELINE / PRESERVE:

- `dataset\final\`

VERIFIED FACT: exact structure:

```text
dataset\final\
  train\
    positive\
    unknown\
    background\
  val\
    positive\
    unknown\
    background\
  test\
    positive\
    unknown\
    background\
```

VERIFIED FACT: counts:

- train positive: 121
- train unknown: 121
- train background: 121
- val positive: 26
- val unknown: 26
- val background: 26
- test positive: 26
- test unknown: 26
- test background: 26
- total real-only WAV files: 519

VERIFIED FACT: audio integrity for `dataset\final`:

- total files: 519
- unreadable/problems: 0
- sample rate: 16,000 Hz for all 519
- channels: 1 for all 519
- sample width: 16-bit PCM for all 519
- min frames: 6,826
- max frames: 49,131

VERIFIED FACT: `dataset\final\background` exists but contains no files; actual split-specific background files are under `dataset\final\train\background`, `dataset\final\val\background`, and `dataset\final\test\background`.

PROJECT DECISION:

- The real-only final dataset is the baseline reference dataset and should remain preserved.

## Synthetic Data Strategy

PROJECT DECISION:

- For the first synthetic-data experiment, synthetic data is added ONLY to train.
- Validation and test remain completely real-only.
- Synthetic derivatives of training samples remain in training only to avoid validation/test contamination.

Target additions:

- +242 synthetic positive
- +242 synthetic unknown
- +242 synthetic background
- +726 synthetic total

Combined train target:

- 363 positive
- 363 unknown
- 363 background
- 1,089 total

## Synthetic Positive

Script: `generate_synthetic_positive.py`

VERIFIED FACT: configuration:

- `SEED = 26172`
- `SAMPLE_RATE = 16000`
- `TARGET_SECONDS = 3.0`
- `TARGET_SAMPLES = 48000`
- `N_VARIANTS_PER_SOURCE = 2`
- source directory: `dataset\final\train\positive`
- background directory: `dataset\final\train\background`
- output directory: `dataset\synthetic\train\positive`
- manifest: `dataset\synthetic\train\synthetic_positive_manifest.csv`

VERIFIED FACT: script enforces:

- exactly 121 training positive source WAVs
- exactly 121 training background WAVs
- exactly 242 generated positive WAVs
- exactly 242 metadata rows

VERIFIED FACT: manifest columns:

- `synthetic_filename`
- `source_filename`
- `background_filename`
- `variant`
- `gain_db`
- `snr_db`
- `shift_ms`
- `reverb`
- `seed`

VERIFIED FACT: variants:

- `gain_noise`: gain sampled uniformly from -3.0 to +3.0 dB; SNR sampled uniformly from 15.0 to 25.0 dB; deterministic pairing with one training background; safe peak limit 0.95.
- `shift_reverb`: shift sampled uniformly from -120.0 to +120.0 ms; shift implemented with `np.roll`; mild reverb with delays `[11, 23, 37, 53]` ms and gains `[0.18, 0.11, 0.07, 0.04]`; safe peak limit 0.95; no background.

VERIFIED FACT: `prepare_audio` converts to mono if needed, tiles short audio to 3 seconds, and truncates longer audio to 3 seconds.

VERIFIED FACT: output WAVs are written with `subtype="PCM_16"`.

PROJECT DECISION:

- no pitch shifting initially
- no TTS initially
- maximum two transformations per synthetic sample
- originals untouched
- reproducible seed

KNOWN CAVEAT:

- Timing shift uses `np.roll`, which is circular. This was accepted for the current frozen experiment but may be revisited because circular shifting can wrap audio from end to beginning.

VERIFIED FACT: synthetic positive audit:

- files: 242
- sample rate: 16 kHz
- channels: mono
- frames: 48,000 each
- duration: 3.0 s each
- peak mean: 0.6571
- peak min: 0.0539
- peak median: 0.7325
- peak max: 0.950012
- peak >=0.95: 32 files
- RMS mean: 0.0653
- RMS min: 0.0042
- RMS max: 0.1651
- RMS >0.5: 0

VERIFIED FACT:

- `dataset\synthetic\train\synthetic_positive_manifest.csv`
- SHA-256: `F3BA72A27AEA8361C2EDEF8789510B5D1E6E466ACEC8C64A114BCCF0A732984C`

## Synthetic Unknown

Script: `generate_synthetic_unknown.py`

VERIFIED FACT: configuration:

- `SEED = 26172`
- `TARGET_SR = 16000`
- `TARGET_SECONDS = 3`
- `TARGET_SAMPLES = 48000`
- `N_VARIANTS_PER_SOURCE = 2`
- source directory: `dataset\final\train\unknown`
- background directory: `dataset\final\train\background`
- output directory: `dataset\synthetic\train\unknown`
- manifest: `dataset\synthetic\train\synthetic_unknown_manifest.csv`

VERIFIED FACT: script enforces:

- exactly 121 training unknown source WAVs
- exactly 121 training background WAVs

VERIFIED FACT: unknown clips are placed inside a 3-second zero-padded window by `place_in_window(audio, shift_ms)`. They are not tiled/repeated unless they are longer than 3 seconds, in which case they are truncated.

VERIFIED FACT: variants:

- `gain_noise`: gain sampled uniformly from -3.0 to +3.0 dB; placement shift sampled approximately +/-120 ms but manifest writes `shift_ms = 0.0` for variant 1; background SNR sampled uniformly from 15.0 to 25.0 dB; background is tiled/truncated to 3 seconds; safe peak limit 0.95.
- `shift_reverb`: placement shift sampled uniformly from -120.0 to +120.0 ms; mild reverb with same delays/gains as positive; no background; safe peak limit 0.95.

VERIFIED FACT: manifest columns:

- `synthetic_filename`
- `source_filename`
- `background_filename`
- `variant`
- `gain_db`
- `snr_db`
- `shift_ms`
- `reverb`
- `seed`

VERIFIED FACT: `seed` column is always `26172` in the unknown manifest, unlike synthetic positive where the seed column varies by source/variant.

PROJECT DECISION:

- no pitch shifting
- no TTS
- short unknown clips are not tiled/repeated

VERIFIED FACT: synthetic unknown audit:

- files: 242
- sample rate: 16 kHz
- channels: mono
- frames: 48,000 each
- exact duration: 3.0 s each
- RMS min: 0.003892
- RMS mean: 0.040921
- RMS max: 0.129425
- peak min: 0.025452
- peak mean: 0.530619
- peak max: 0.950012
- files with peak >=0.95: 9
- RMS >0.5: 0

VERIFIED FACT:

- `dataset\synthetic\train\synthetic_unknown_manifest.csv`
- SHA-256: `608AC8CE76AF7E62B244255A43A8D6326962BC94C014AE074A605753D54F172F`

## Synthetic Background

Script: `generate_synthetic_background.py`

VERIFIED FACT: configuration:

- `SEED = 26172`
- `TARGET_SR = 16000`
- `TARGET_SECONDS = 3`
- `TARGET_SAMPLES = 48000`
- `N_VARIANTS_PER_SOURCE = 2`
- source directory: `dataset\final\train\background`
- output directory: `dataset\synthetic\train\background`
- manifest: `dataset\synthetic\train\synthetic_background_manifest.csv`

VERIFIED FACT: script enforces exactly 121 training background source WAVs.

VERIFIED FACT: all source backgrounds are tiled to 3 seconds by `tile_to_length`.

VERIFIED FACT: variants:

- `gain`: gain sampled uniformly from -3.0 to +3.0 dB; no shift.
- `gain_shift`: gain sampled uniformly from -3.0 to +3.0 dB; circular shift sampled uniformly from -250.0 to +250.0 ms.

VERIFIED FACT: circular shift implemented by `np.roll` in `circular_shift`.

VERIFIED FACT: safe peak limit = 0.95.

VERIFIED FACT: manifest columns:

- `synthetic_filename`
- `source_filename`
- `variant`
- `gain_db`
- `shift_ms`
- `seed`

VERIFIED FACT: synthetic background audit:

- files: 242
- sample rate: 16 kHz
- channels: mono
- frames: 48,000 each
- duration: 3.0 s each
- RMS mean: 0.122340
- RMS std: 0.126035
- RMS min: 0.000795
- RMS max: 0.387777
- peak mean: 0.474717
- peak min: 0.007141
- peak max: 0.950012
- RMS <0.001: 2
- RMS >0.5: 0
- peak >=0.95: 42
- peak-limited: 42/242 = 17.36%
- variant counts reaching peak >=0.95: `v01 = 21`, `v02 = 21` (derived from filenames)

IMPORTANT TERMINOLOGY:

- Samples reaching 0.95 are peak-limited, NOT clipped.

KNOWN CAVEAT:

- Circular shift uses `np.roll`; review before production.

VERIFIED FACT:

- `dataset\synthetic\train\synthetic_background_manifest.csv`
- SHA-256: `D5D3A193DCD054DC69F26896B4D54EC6E5A267768C20BCDA8CEB208B970269A3`

## Cross-Class Integrity

VERIFIED FACT from `authoritative_dataset_manifest.csv`:

- real positive paths: 173 rows, 173 unique
- real unknown paths: 173 rows, 173 unique
- real background paths: 173 rows, 173 unique
- total real unique paths: 519
- cross-class real path collision: 0
- status: PASS

## Combined Training Dataset

AUTHORITATIVE TRAINING MANIFEST:

- `dataset\processed\combined_train_manifest.csv`
- SHA-256: `358CEEA2035510423C404D7352272138E477D5B806C914B7FC68C676C2E171E1`

VERIFIED FACT: columns:

- `split`
- `class`
- `source_type`
- `filename`
- `full_path`

VERIFIED FACT: exact structure:

- positive: 121 real + 242 synthetic = 363 total
- unknown: 121 real + 242 synthetic = 363 total
- background: 121 real + 242 synthetic = 363 total
- total rows: 1,089
- total real: 363
- total synthetic: 726
- unique paths: 1,089
- duplicate paths: 0

VERIFIED FACT: audio integrity for `combined_train_manifest.csv`:

- rows: 1,089
- missing paths: 0
- unreadable/bad: 0
- all sample rates: 16 kHz
- all channels: mono
- duration range: 0.426625 s to 3.040375 s

PROJECT NOTE FROM HANDOFF REQUEST, NOT VERIFIED FROM REPOSITORY:

- A first verification command incorrectly expected 121 samples for every class/source combination and reported FAIL.
- That was a verification-script error, not a dataset error.
- Correct condition: real per class = 121 and synthetic per class = 242.
- Corrected validation passed.
- Future AI must not "fix" the dataset based on that false FAIL.

## Authoritative Dataset Manifest

FROZEN / AUTHORITATIVE ARTIFACT:

- `dataset\processed\authoritative_dataset_manifest.csv`
- SHA-256: `04A3EFC9FBD9502EBD87723BC5676F1A5F7845BAC851FB995FACE99F6F73CD48`

VERIFIED FACT: columns:

- `split`
- `class`
- `source_type`
- `filename`
- `full_path`

VERIFIED FACT: exact contents:

- train: 1,089 rows; 363/class; 121 real/class; 242 synthetic/class
- val: 78 rows; 26/class; all real
- test: 78 rows; 26/class; all real
- total rows: 1,245

VERIFIED FACT: overall class totals:

- positive: 415
- unknown: 415
- background: 415

VERIFIED FACT: split x source:

- train: 363 real, 726 synthetic
- val: 78 real, 0 synthetic
- test: 78 real, 0 synthetic

VERIFIED FACT: split x class:

- train: 363 positive, 363 unknown, 363 background
- val: 26 positive, 26 unknown, 26 background
- test: 26 positive, 26 unknown, 26 background

VERIFIED FACT:

- duplicate paths: 0
- missing paths: 0
- unreadable/bad: 0
- all sample rates: 16 kHz
- all channels: mono
- duration range: 0.426625 s to 3.0706875 s
- status: PASS

## Important 3-Second vs 1-Second Pipeline Issue

CRITICAL TODO:

- The dataset files are mostly approximately 3 seconds.
- The current feature frontend expects exactly 1 second.
- Therefore: DO NOT simply run the current feature frontend over the 3-second WAVs.

VERIFIED FACT:

- `feature_frontend.py` pads if shorter than 1 second.
- `feature_frontend.py` truncates if longer than 1 second.
- A naive feature-extraction run on 3-second files would keep only the first 1 second.

WHY THIS MATTERS:

- Positive recordings may contain the keyword outside the first second.
- Training on first-second truncations could silently corrupt labels.

CORRECT NEXT TASK:

- Design and implement a deliberate 3-second audio -> 1-second KWS window policy.

NOT FINALIZED:

- positive window selection
- unknown window selection
- background window selection
- whether one or multiple 1-second windows should be generated per source
- how endpoint/speech information should be used
- how to preserve class semantics
- how to avoid introducing leakage
- how the policy interacts with `(40, 49, 1)` model input
- how final training sample count changes after windowing
- whether augmentation should occur before or after window extraction
- consistency between training and ESP32 inference

DO NOT begin feature extraction until this policy is explicitly decided.

## Current Dataset Status

| Component | Status |
| --- | --- |
| Real positive collection | COMPLETE for 173 reviewed/processed positives in this repo |
| Speaker-independent split | FROZEN |
| Speech Commands unknown selection | COMPLETE |
| Background selection | COMPLETE |
| Real-only baseline dataset | COMPLETE |
| Synthetic positive | COMPLETE/FROZEN |
| Synthetic unknown | COMPLETE/FROZEN |
| Synthetic background | COMPLETE/FROZEN |
| Combined train manifest | COMPLETE |
| Authoritative manifest | COMPLETE/FROZEN |
| Feature extraction | NOT STARTED |
| 3 s -> 1 s window policy | NOT FINALIZED |
| DS-CNN training | NOT STARTED / no training code found |
| Validation evaluation | NOT STARTED |
| Test evaluation | NOT STARTED |
| ESP32 deployment | NOT STARTED / no firmware project found |
| Quantization | NOT STARTED / no quantization artifact found |

## File and Directory Inventory

Key scripts:

- `feature_frontend.py`: extracts 1-second log-Mel features; outputs `(40, 49, 1)`; should not be blindly applied to 3-second files.
- `analyze_positive_endpoints.py`: analyzes `dataset\processed\positive` WAVs using 30 ms frames, 10 ms hop, 20th-percentile noise floor, threshold `noise_floor + 10 dB`, 100 ms smoothing, 80 ms minimum active region; outputs `dataset\processed\positive_endpoint_analysis.csv`. Output exists and is useful for windowing policy. Do not rerun casually unless preserving current CSV/hash/versioning.
- `find_split.py`: exploratory randomized speaker split search using `dataset\processed\dataset_manifest.csv`, seed 42, target val/test recording count 26 and 6 speakers each. Not authoritative.
- `freeze_split.py`: hard-coded speaker IDs for train/val/test; writes `dataset\processed\split_manifest.csv`. Do not rerun casually because the output is frozen.
- `verify_split_manifest.py`: checks `split_manifest.csv` rows have corresponding flattened positive WAVs in `dataset\processed\positive`; prints PASS.
- `select_unknown_and_background.py`: final authoritative unknown/background selection script; uses Speech Commands official validation/testing lists, selected 31 unknown labels, temporal background split, seed 26172 plus split offsets; writes `unknown_selection_manifest.csv` and `background_selection_manifest.csv`. Do not rerun casually without preserving current manifests.
- `select_unknown_speech.py`: older/experimental script that selected 1,000 unknown speech files across all 35 word classes into `dataset\raw\unknown_speech`. Not authoritative for final dataset.
- `generate_synthetic_positive.py`: creates 242 synthetic train positives and manifest; output frozen for current experiment.
- `generate_synthetic_unknown.py`: creates 242 synthetic train unknowns and manifest; output frozen for current experiment.
- `generate_synthetic_background.py`: creates 242 synthetic train backgrounds and manifest; output frozen for current experiment.
- `analysis.py`: prints basic stats from `dataset\raw\voice_notes.csv`; exploratory.
- `check_audio.py`: placeholder/manual script expecting `YOUR_FILE.webm`; not a reusable project audit.
- `check_sc_partitions.py`: prints Speech Commands label train/val/test counts for selected labels; exploratory.

Key manifests:

- `dataset\processed\dataset_manifest.csv`: positive raw WebM manifest; 173 rows.
- `dataset\processed\split_manifest.csv`: frozen positive split; 173 rows; authoritative.
- `dataset\processed\positive_endpoint_analysis.csv`: endpoint/speech-span analysis for 173 positives.
- `dataset\processed\positive_review_manifest.csv`: 173 rows with blank `decision`/`reason`.
- `dataset\processed\unknown_selection_manifest.csv`: authoritative unknown selection; 173 rows.
- `dataset\processed\background_selection_manifest.csv`: authoritative background selection; 173 rows.
- `dataset\processed\combined_train_manifest.csv`: train real+synthetic; 1,089 rows.
- `dataset\processed\authoritative_dataset_manifest.csv`: full train/val/test manifest; 1,245 rows; frozen.
- `dataset\synthetic\train\synthetic_positive_manifest.csv`: 242 rows.
- `dataset\synthetic\train\synthetic_unknown_manifest.csv`: 242 rows.
- `dataset\synthetic\train\synthetic_background_manifest.csv`: 242 rows.

Key directories:

- `dataset\raw\audio`: raw participant WebM recordings; 173 `.webm` files.
- `dataset\processed\audio_wav`: converted WAVs by participant/type; 173 WAV files.
- `dataset\processed\positive`: flattened converted positive WAVs; 173 WAV files.
- `dataset\raw\external\speech_commands_v0.02`: extracted Speech Commands dataset.
- `dataset\raw\background_noise`: 360 preprocessed background segments.
- `dataset\raw\unknown_speech`: older/experimental 1,000-file unknown selection.
- `dataset\final`: real-only baseline dataset; 519 WAVs.
- `dataset\synthetic\train`: 726 synthetic training WAVs.

## Reproducibility

Seeds:

- speaker exploratory split in `find_split.py`: `random.seed(42)`
- unknown/background selection base seed: `26172`
- unknown split seed offsets: train `+101`, val `+202`, test `+303`
- background split seed offsets: train `+404`, val `+505`, test `+606`
- synthetic positive seed: `26172`
- synthetic unknown seed: `26172`
- synthetic background seed: `26172`
- older `select_unknown_speech.py` seed: `26172`

Manifest hashes:

- `dataset\processed\split_manifest.csv`: `0989224592842A34D9958198E8218B7CE0114989F09B05FF5E81CE51C6F09CE8`
- `dataset\processed\unknown_selection_manifest.csv`: `060604817BA087EB2039BC38FE69A3905C2F12529CC10D7A013FD6EA1CC0626B`
- `dataset\processed\background_selection_manifest.csv`: `A89278620DEAECBFEEDBE8A67E349EBA5D1A13275641BC4B944132EFB44D2B76`
- `dataset\synthetic\train\synthetic_positive_manifest.csv`: `F3BA72A27AEA8361C2EDEF8789510B5D1E6E466ACEC8C64A114BCCF0A732984C`
- `dataset\synthetic\train\synthetic_unknown_manifest.csv`: `608AC8CE76AF7E62B244255A43A8D6326962BC94C014AE074A605753D54F172F`
- `dataset\synthetic\train\synthetic_background_manifest.csv`: `D5D3A193DCD054DC69F26896B4D54EC6E5A267768C20BCDA8CEB208B970269A3`
- `dataset\processed\combined_train_manifest.csv`: `358CEEA2035510423C404D7352272138E477D5B806C914B7FC68C676C2E171E1`
- `dataset\processed\authoritative_dataset_manifest.csv`: `04A3EFC9FBD9502EBD87723BC5676F1A5F7845BAC851FB995FACE99F6F73CD48`

Fixed parameters:

- sample rate: 16 kHz
- channels: mono
- processed/final/synthetic WAV subtype: 16-bit PCM where scripts write synthetic data
- classes: `positive`, `unknown`, `background`
- feature tensor: `(40, 49, 1)`
- frontend duration: 1 second
- synthetic output duration: 3 seconds / 48,000 frames
- synthetic gain ranges: -3 dB to +3 dB
- synthetic positive/unknown target SNR: 15 dB to 25 dB
- positive/unknown shift range: approximately +/-120 ms
- background shift range: approximately +/-250 ms
- safe peak limit: 0.95

Immutable/frozen artifacts:

- `dataset\processed\split_manifest.csv`
- `dataset\final\`
- `dataset\processed\unknown_selection_manifest.csv`
- `dataset\processed\background_selection_manifest.csv`
- `dataset\synthetic\train\`
- `dataset\processed\combined_train_manifest.csv`
- `dataset\processed\authoritative_dataset_manifest.csv`
- raw recordings under `dataset\raw\audio`

## Known Pitfalls and Corrections

1. Do not assume 3-second WAVs can be directly fed into the current 1-second frontend.
2. Do not regenerate the frozen speaker split.
3. Do not mix synthetic validation/test data into the first experiment.
4. Do not modify original recordings.
5. Do not interpret peak limiting at 0.95 as clipping.
6. Do not blindly apply positive gain.
7. Background mixing should use target SNR.
8. Synthetic unknown clips should not be tiled/repeated.
9. PROJECT NOTE, NOT VERIFIED FROM REPOSITORY: the combined manifest's first FAIL was caused by an incorrect verification condition, not incorrect dataset balance.
10. Windows Python path strings can interpret sequences such as `\f` and `\v` as escape characters. Prefer `pathlib.Path` or correctly escaped/raw paths.
11. Do not infer source participant identity from final sequential filenames such as `0000.wav`. Use authoritative manifests and metadata.
12. `np.roll` is currently used for circular timing shifts in synthetic positive/background and should be treated as a known caveat for future production-quality review.
13. `feature_frontend.py` comments/docstring disagree with actual shape; actual output is `(40, 49, 1)`.
14. `select_unknown_speech.py` and `dataset\raw\unknown_speech` are older/experimental and include labels excluded from the final unknown selection.
15. `participants.csv` contains PII. Do not paste participant names, phone numbers, auth IDs, secrets, or credentials into model prompts or public files.

## Decisions Log

- PROJECT DECISION: use 3-class classification: `positive`, `unknown`, `background`.
  - Reason: separates target phrase, other speech, and nonspeech/background.
  - Status: implemented in dataset manifests.
  - Frozen: yes for current dataset.
  - Revisit if adding more classes such as silence/noise subtypes.

- PROJECT DECISION / USER-SUPPLIED: DS-CNN architecture.
  - Reason: common efficient KWS architecture for edge deployment.
  - Status: not implemented in repository.
  - Frozen: not verifiable.
  - Revisit when model code is created.

- PROJECT DECISION: speaker-independent positive split.
  - Reason: evaluate generalization to unseen speakers.
  - Status: implemented in `split_manifest.csv`.
  - Frozen: yes.
  - Revisit only with explicit dataset versioning.

- PROJECT DECISION: Speech Commands unknown selection.
  - Reason: use real spoken words distinct from `activate orbit`.
  - Status: implemented in `unknown_selection_manifest.csv`.
  - Frozen: yes for current dataset.
  - Revisit if unknown class definition changes.

- PROJECT DECISION: background temporal split.
  - Reason: reduce temporal leakage from continuous background recordings.
  - Status: implemented in `background_selection_manifest.csv`.
  - Frozen: yes for current dataset.
  - Revisit only with explicit dataset versioning.

- PROJECT DECISION: synthetic data only in training for first experiment.
  - Reason: keep validation/test real-only.
  - Status: implemented in `authoritative_dataset_manifest.csv`.
  - Frozen: yes for current experiment.
  - Revisit in later experiments after baseline results.

- PROJECT DECISION: 2 synthetic variants per training source.
  - Reason: controlled augmentation scale.
  - Status: implemented.
  - Frozen: yes for current experiment.
  - Revisit after evaluation.

- PROJECT DECISION: no pitch shift initially.
  - Reason: avoid unrealistic or uncontrolled speaker artifacts in first experiment.
  - Status: implemented by omission.
  - Frozen: yes for current experiment.
  - Revisit after baseline.

- PROJECT DECISION: no TTS initially.
  - Reason: keep first synthetic experiment derived from real samples.
  - Status: implemented by omission.
  - Frozen: yes for current experiment.
  - Revisit after baseline.

- PROJECT DECISION: conservative gain and clipping/peak protection.
  - Reason: real positives already include high peaks/clipping risk.
  - Status: implemented with -3 dB to +3 dB gain and 0.95 peak limiting.
  - Frozen: yes for current experiment.
  - Revisit with more audio-level analysis.

- PROJECT DECISION: mix background by target SNR.
  - Reason: raw-amplitude mixing would be inconsistent.
  - Status: implemented for `gain_noise` variants.
  - Frozen: yes for current experiment.
  - Revisit if SNR curriculum is introduced.

- PROJECT DECISION: 3-second synthetic outputs.
  - Reason: align synthetic files with current real recording duration before windowing.
  - Status: implemented.
  - Frozen: yes for current experiment.
  - Revisit only together with 3 s -> 1 s policy.

- PROJECT DECISION: preserve originals.
  - Reason: auditability and reproducibility.
  - Status: raw/processed/final/synthetic artifacts are separated.
  - Frozen: yes.
  - Revisit never without explicit archival/versioning.

- PROJECT DECISION: manifest SHA-256 hashing.
  - Reason: reproducibility.
  - Status: hashes computed and recorded above.
  - Frozen: yes for current artifacts.
  - Revisit if manifests are intentionally versioned.

- VERIFIED FACT / CURRENT IMPLEMENTATION: current frontend is 1 second, `(40, 49, 1)`.
  - Reason: code uses `target_length = SAMPLE_RATE`.
  - Status: implemented.
  - Frozen: no; should be revisited after window policy.

- TODO / NOT YET DONE: 3-second-to-1-second windowing problem.
  - Reason: current data duration conflicts with current frontend duration.
  - Status: unresolved.
  - Frozen: no.
  - Revisit now; it is the exact next task.

## Exact Next Task

Design the 3-second -> 1-second KWS windowing policy before feature extraction.

Explicit constraints:

- Feature extraction has NOT started.
- The current frontend must not be blindly applied to the 3-second files.
- No training should begin until the windowing policy is settled.
- The authoritative dataset manifests should remain unchanged.
- The policy must define how positives, unknown speech, and background are converted to 1-second windows.
- The policy must preserve speaker/data split boundaries and avoid leakage.
- The policy must produce inputs compatible with the repository-verified frontend tensor shape `(40, 49, 1)`.
