import pandas as pd

df = pd.read_csv("dataset/raw/voice_notes.csv")

print("Total rows:", len(df))
print("Unique participants:", df["participant_id"].nunique())

print("\nNOTE TYPES:")
print(df["note_type"].value_counts(dropna=False))

print("\nVARIATION TAGS:")
print(df["variation_tag"].value_counts(dropna=False))

print("\nDURATION:")
print("Min:", df["duration_ms"].min(), "ms")
print("Max:", df["duration_ms"].max(), "ms")
print("Mean:", df["duration_ms"].mean(), "ms")

print("\nSEQUENCE NUMBERS:")
print(df["sequence_number"].describe())

print("\nMissing values:")
print(df.isna().sum())