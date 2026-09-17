from pathlib import Path

root = Path(r"dataset\raw\external\speech_commands_v0.02")

labels = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight",
    "five", "four", "go", "happy", "house", "learn", "left",
    "marvin", "nine", "no", "off", "on", "one", "right", "seven",
    "sheila", "six", "stop", "three", "tree", "two", "up",
    "yes", "zero"
]

val = set((root / "validation_list.txt").read_text().splitlines())
test = set((root / "testing_list.txt").read_text().splitlines())

print(f"{'LABEL':<22}{'TRAIN':>7}{'VAL':>7}{'TEST':>7}")
print("-" * 43)

for label in labels:
    files = list((root / label).glob("*.wav"))

    train = 0
    validation = 0
    testing = 0

    for p in files:
        rel = f"{label}/{p.name}"

        if rel in val:
            validation += 1
        elif rel in test:
            testing += 1
        else:
            train += 1

    print(f"{label:<22}{train:>7}{validation:>7}{testing:>7}")