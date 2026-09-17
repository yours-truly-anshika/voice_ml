import random
from pathlib import Path

ROOT = Path(r"C:\Users\Anshika\voice_ml\dataset\raw\external\speech_commands_v0.02")
OUT = Path(r"C:\Users\Anshika\voice_ml\dataset\raw\unknown_speech")

random.seed(26172)

# Read official validation and test paths
validation = set(
    x.strip().replace("\\", "/")
    for x in (ROOT / "validation_list.txt").read_text().splitlines()
    if x.strip()
)

testing = set(
    x.strip().replace("\\", "/")
    for x in (ROOT / "testing_list.txt").read_text().splitlines()
    if x.strip()
)

excluded = validation | testing

# All spoken-word WAV files
files = []
for word_dir in ROOT.iterdir():
    if not word_dir.is_dir() or word_dir.name.startswith("_"):
        continue

    for wav in word_dir.glob("*.wav"):
        rel = wav.relative_to(ROOT).as_posix()
        if rel not in excluded:
            files.append((word_dir.name, wav))

# Group by word
by_word = {}
for word, path in files:
    by_word.setdefault(word, []).append(path)

# Target: approximately equal representation across 35 words
words = sorted(by_word)
target_total = 1000
per_word = target_total // len(words)
remainder = target_total % len(words)

selected = []

for i, word in enumerate(words):
    candidates = by_word[word].copy()
    random.shuffle(candidates)

    target = per_word + (1 if i < remainder else 0)

    # Prefer speaker diversity: max 2 clips per speaker for this word
    chosen = []
    speaker_counts = {}

    for path in candidates:
        speaker = path.stem.split("_nohash_")[0]
        count = speaker_counts.get(speaker, 0)

        if count < 2:
            chosen.append(path)
            speaker_counts[speaker] = count + 1

        if len(chosen) >= target:
            break

    selected.extend((word, p) for p in chosen)

# Create output directory
OUT.mkdir(parents=True, exist_ok=True)

# Copy selected files with unique names
manifest = []

for index, (word, src) in enumerate(selected, start=1):
    speaker = src.stem.split("_nohash_")[0]
    dst = OUT / f"unknown_{index:04d}_{word}_{speaker}.wav"

    dst.write_bytes(src.read_bytes())

    manifest.append(
        f"{dst.name},{word},{speaker},{src.relative_to(ROOT).as_posix()}"
    )

(OUT / "selection_manifest.csv").write_text(
    "filename,word,speaker,source_path\n" + "\n".join(manifest),
    encoding="utf-8"
)

print(f"Selected: {len(selected)}")
print(f"Words: {len(words)}")
print(f"Output: {OUT}")
