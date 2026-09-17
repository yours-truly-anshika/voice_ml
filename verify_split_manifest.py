import pandas as pd
from pathlib import Path

MANIFEST = Path(r"dataset\processed\split_manifest.csv")
POSITIVE_DIR = Path(r"dataset\processed\positive")

df = pd.read_csv(MANIFEST)

missing = []
found = []

for _, row in df.iterrows():
    wav_name = Path(row["Filename"]).stem + ".wav"
    
    # Positive files use the flattened naming convention:
    # ParticipantID_NoteType_SequenceNumber.wav
    expected = POSITIVE_DIR / (
        f"{row['ParticipantID']}_{row['NoteType']}_{int(row['SequenceNumber'])}.wav"
    )

    if expected.exists():
        found.append(expected)
    else:
        missing.append(str(expected))

print("=" * 60)
print("FROZEN SPLIT / WAV INTEGRITY CHECK")
print("=" * 60)

print(f"\nManifest rows: {len(df)}")
print(f"WAV files found: {len(found)}")
print(f"Missing WAV files: {len(missing)}")

if missing:
    print("\nMISSING FILES:")
    for path in missing:
        print(path)
    raise SystemExit("\nERROR: Some manifest recordings are missing.")

print("\nSplit counts:")
print(df["Split"].value_counts().sort_index().to_string())

print("\nSpeaker counts:")
print(
    df.groupby("Split")["ParticipantID"]
    .nunique()
    .sort_index()
    .to_string()
)

print("\nSTATUS: PASS")