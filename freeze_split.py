import pandas as pd

MANIFEST = r"dataset\processed\dataset_manifest.csv"
OUTPUT = r"dataset\processed\split_manifest.csv"

train = {
    "15c65680-cfe5-44e4-ac9b-f8ccd940653d",
    "ef518ed6-56c1-4a88-8afe-158ff08a2964",
    "4234fb4d-064c-484d-88aa-ef5be9c7eabf",
    "1c231685-30ef-4e9a-a881-ed8b63c6f1b6",
    "a1f3bee7-082d-458e-8208-b3e808172cee",
    "efc91795-fdcf-4347-b99e-efaf84c2f9be",
    "d30246ec-2c35-481e-a9e5-155d77fdd772",
    "2a24ad90-f292-4321-9baf-081583734e42",
    "aa91f7e3-e010-44eb-89ae-877dae296ad8",
    "73b51835-9a81-46f6-968e-6d3f2d58a323",
    "fe398b09-8619-4849-b515-8b11c24fc7e7",
    "baa8bfbe-be1f-4849-a601-67812dfe2775",
    "8e25f738-67de-47c3-8986-885244e13042",
    "2b017ac3-bbec-4db3-8402-8f942b24becd",
    "2e2da424-76f9-4afd-ae75-12a27e00ac73",
    "2a69429e-6de8-4410-8267-e2185fe3809f",
    "f70d467e-8781-426e-80e9-a74519d645ea",
    "2d70cdb6-2e97-4ac9-ae60-daebf846ea60",
    "52499bbe-68bc-4e40-86d7-6121e9dfeb24",
    "d9fa7fff-de38-4adc-b1e3-c27fa4626dfa",
    "02c153b8-8fb9-4528-8158-8049e257db88",
    "335b6d7c-5ded-4be0-b876-09b9508d2265",
    "5da6fe17-5ed0-43eb-bb94-8a61cc1c37f1",
    "2c214cfb-1bf1-4660-b74b-d804bbfc73d4",
    "ddd732c0-fb06-4989-a4ff-e9664d7a7087",
}

val = {
    "943069ad-2757-493d-8f94-d7a295137152",
    "47f64db2-16aa-4904-94c9-bab3f0095a28",
    "f5f4bb28-973c-4278-9967-38fb81c6a050",
    "263875b0-b8cb-4e7f-a43a-d6e551ee48ed",
    "64de123c-53a4-4059-957c-92bf29ac468b",
    "2d80737c-f830-4ec6-9dfe-a6fcc805d9ee",
}

test = {
    "c7d753f1-f749-4a37-8bd5-53dc9bab4fea",
    "da3b85a1-387b-40d5-a8af-ce10db3ac2c8",
    "81fdbf2e-3612-4678-bf20-f0113528505c",
    "054b38c0-8e14-4bd8-a649-2e2c5f3d389e",
    "aca6354d-db66-444b-b95a-b2e6df9361ef",
    "2c574cb7-1d2b-440f-bf2f-6428bfe6bd07",
}

df = pd.read_csv(MANIFEST)

all_speakers = set(df["ParticipantID"])

assert train.isdisjoint(val)
assert train.isdisjoint(test)
assert val.isdisjoint(test)
assert train | val | test == all_speakers

def assign_split(pid):
    if pid in train:
        return "train"
    if pid in val:
        return "val"
    if pid in test:
        return "test"
    raise ValueError(f"Unknown participant: {pid}")

df["Split"] = df["ParticipantID"].apply(assign_split)

df.to_csv(OUTPUT, index=False)

print("SPLIT MANIFEST CREATED")
print()
print(df["Split"].value_counts().sort_index())
print()
print("SPEAKERS:")
print(df.groupby("Split")["ParticipantID"].nunique().sort_index())
print()
print("OUTPUT:", OUTPUT)