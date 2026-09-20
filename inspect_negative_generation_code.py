from pathlib import Path

ROOT = Path(r"C:\Users\Anshika\voice_ml")

KEYWORDS = [
    "unknown",
    "background",
    "synthetic",
    "negative",
]

print("=" * 70)
print("POSSIBLE NEGATIVE-DATA GENERATION SCRIPTS")
print("=" * 70)

matches = []

for path in sorted(ROOT.rglob("*.py")):
    if any(part in {"venv", ".venv", "__pycache__"} for part in path.parts):
        continue

    name = path.name.lower()

    if any(keyword in name for keyword in KEYWORDS):
        matches.append(path)

if not matches:
    print("\nNo matching Python files found.")
else:
    print(f"\nFound {len(matches)} possible scripts:\n")

    for path in matches:
        print(path.relative_to(ROOT))

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)