import pandas as pd
import random

manifest = r"dataset\processed\dataset_manifest.csv"

df = pd.read_csv(manifest)

speaker_counts = df.groupby("ParticipantID").size()
speakers = list(speaker_counts.index)
counts = speaker_counts.to_dict()

random.seed(42)
random.shuffle(speakers)

target_val = 26
target_test = 26


def choose_group(available, target, group_size):
    best_group = None
    best_error = float("inf")

    for _ in range(5000):
        candidate_pool = available.copy()
        random.shuffle(candidate_pool)

        group = candidate_pool[:group_size]
        count = sum(counts[x] for x in group)

        error = abs(count - target)

        if error < best_error:
            best_error = error
            best_group = group

    return best_group


val = choose_group(speakers, target_val, 6)

remaining = [x for x in speakers if x not in val]

test = choose_group(remaining, target_test, 6)

train = [x for x in remaining if x not in test]


# Assign split labels only in memory
split_map = {}

for x in train:
    split_map[x] = "train"

for x in val:
    split_map[x] = "val"

for x in test:
    split_map[x] = "test"

df["Split"] = df["ParticipantID"].map(split_map)


print("=" * 65)
print("FINAL SPLIT AUDIT")
print("=" * 65)

print("\nRECORDING COUNTS:")
print(df["Split"].value_counts().sort_index().to_string())

print("\nPERCENTAGES:")
print(
    (df["Split"].value_counts(normalize=True).sort_index() * 100)
    .round(2)
    .to_string()
)

print("\nMANDATORY / OPTIONAL BY SPLIT:")
print(
    pd.crosstab(df["Split"], df["NoteType"])
    .reindex(["train", "val", "test"])
    .fillna(0)
    .to_string()
)

print("\nSPEAKERS PER SPLIT:")
print(df.groupby("Split")["ParticipantID"].nunique().to_string())

print("\nSPEAKER OVERLAP CHECK:")

train_set = set(train)
val_set = set(val)
test_set = set(test)

print("TRAIN & VAL:", len(train_set & val_set))
print("TRAIN & TEST:", len(train_set & test_set))
print("VAL & TEST:", len(val_set & test_set))

print("\nSPEAKER COUNTS WITHIN EACH SPLIT:")

for split in ["train", "val", "test"]:
    print(f"\n{split.upper()}:")
    print(
        df[df["Split"] == split]
        .groupby("ParticipantID")
        .size()
        .sort_values()
        .to_string()
    )