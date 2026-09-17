import argparse
import csv
import random
import re
from collections import defaultdict, Counter
from pathlib import Path


AUDIO_EXT = ".wav"

# Deliberately chosen Speech Commands labels for the first
# real-data "unknown speech" baseline.
#
# These are ordinary spoken words/commands/digits and are
# distinct from the target phrase "activate orbit".
UNKNOWN_LABELS = [
    "backward",
    "bed",
    "bird",
    "cat",
    "dog",
    "down",
    "eight",
    "five",
    "four",
    "go",
    "happy",
    "house",
    "learn",
    "left",
    "marvin",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "seven",
    "sheila",
    "six",
    "stop",
    "three",
    "tree",
    "two",
    "up",
    "yes",
    "zero",
]


def load_split_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def parse_speaker_id(filename):
    """
    Speech Commands filenames look like:

        00176480_nohash_0.wav

    The hexadecimal prefix identifies the speaker.
    """
    m = re.match(r"^([0-9a-f]+)_", filename.lower())
    if not m:
        raise ValueError(f"Could not parse speaker ID: {filename}")
    return m.group(1)


def collect_unknown_pool(root):
    val_set = load_split_file(root / "validation_list.txt")
    test_set = load_split_file(root / "testing_list.txt")

    discovered = sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and not p.name.startswith("_")
    )

    print(f"Discovered {len(discovered)} Speech Commands word classes.")
    print("Using", len(UNKNOWN_LABELS), "labels for unknown speech.")

    missing = sorted(set(UNKNOWN_LABELS) - set(discovered))
    if missing:
        raise RuntimeError(
            f"Requested unknown labels not found: {missing}"
        )

    pool = {
        "train": defaultdict(list),
        "val": defaultdict(list),
        "test": defaultdict(list),
    }

    for label in UNKNOWN_LABELS:
        for wav in sorted((root / label).glob("*.wav")):

            rel = f"{label}/{wav.name}"
            speaker = parse_speaker_id(wav.name)

            if rel in val_set:
                split = "val"
            elif rel in test_set:
                split = "test"
            else:
                split = "train"

            pool[split][label].append(
                {
                    "src_path": str(wav.resolve()),
                    "label": label,
                    "speaker_id": speaker,
                }
            )

    return pool


def select_unknown(pool, split, target_n, seed):
    """
    Select approximately equal numbers from each label.

    Every label gets either floor(target/31) or ceil(target/31)
    samples, with a deterministic random seed.
    """

    rng = random.Random(seed)

    labels = [
        label for label in UNKNOWN_LABELS
        if pool[split][label]
    ]

    if len(labels) != len(UNKNOWN_LABELS):
        raise RuntimeError(
            f"{split}: only {len(labels)}/{len(UNKNOWN_LABELS)} "
            f"unknown labels have candidates."
        )

    base = target_n // len(labels)
    remainder = target_n % len(labels)

    allocation = {
        label: base for label in labels
    }

    extra_labels = rng.sample(labels, remainder)

    for label in extra_labels:
        allocation[label] += 1

    selected = []

    for label in labels:
        candidates = list(pool[split][label])
        rng.shuffle(candidates)

        if len(candidates) < allocation[label]:
            raise RuntimeError(
                f"Not enough candidates for {split}/{label}: "
                f"need {allocation[label]}, "
                f"have {len(candidates)}"
            )

        selected.extend(candidates[:allocation[label]])

    rng.shuffle(selected)

    return selected


def parse_background_filename(filename):
    """
    Expected:

        bike_000.wav
        dishes_041.wav
        pink_059.wav

    Returns:

        source, segment_index
    """

    m = re.match(r"^(.+)_(\d+)\.wav$", filename)

    if not m:
        raise ValueError(
            f"Background filename does not match SOURCE_INDEX.wav: "
            f"{filename}"
        )

    source = m.group(1)
    index = int(m.group(2))

    return source, index


def collect_background(root):
    by_source = defaultdict(list)

    files = sorted(root.glob("*.wav"))

    for wav in files:
        source, index = parse_background_filename(wav.name)

        by_source[source].append(
            {
                "src_path": str(wav.resolve()),
                "source": source,
                "segment_index": index,
            }
        )

    for source in by_source:
        by_source[source].sort(
            key=lambda x: x["segment_index"]
        )

    print("\nBackground sources:")

    for source in sorted(by_source):
        indices = [
            x["segment_index"]
            for x in by_source[source]
        ]

        print(
            f"  {source:<10} "
            f"{len(indices)} segments "
            f"[{min(indices)}..{max(indices)}]"
        )

        if indices != list(range(len(indices))):
            raise RuntimeError(
                f"{source}: segment indices are not contiguous."
            )

    if len(by_source) != 6:
        raise RuntimeError(
            f"Expected 6 background sources, "
            f"found {len(by_source)}."
        )

    return by_source


def split_background_by_time(by_source):
    """
    60 segments/source:

        000-041 -> TRAIN
        042-050 -> VAL
        051-059 -> TEST

    No source's temporal regions overlap between splits.
    """

    blocks = {
        "train": defaultdict(list),
        "val": defaultdict(list),
        "test": defaultdict(list),
    }

    for source, segments in by_source.items():

        n = len(segments)

        train_end = int(n * 0.70)
        val_end = train_end + int(n * 0.15)

        blocks["train"][source] = segments[:train_end]
        blocks["val"][source] = segments[train_end:val_end]
        blocks["test"][source] = segments[val_end:]

        print(
            f"{source:<10}: "
            f"train={len(blocks['train'][source])}, "
            f"val={len(blocks['val'][source])}, "
            f"test={len(blocks['test'][source])}"
        )

    return blocks


def select_background(blocks, split, target_n, seed):
    """
    Balance selection across the six independent noise sources.
    """

    rng = random.Random(seed)

    sources = sorted(blocks[split])

    base = target_n // len(sources)
    remainder = target_n % len(sources)

    allocation = {
        source: base for source in sources
    }

    extra_sources = rng.sample(sources, remainder)

    for source in extra_sources:
        allocation[source] += 1

    selected = []

    for source in sources:
        candidates = list(blocks[split][source])
        rng.shuffle(candidates)

        if len(candidates) < allocation[source]:
            raise RuntimeError(
                f"Not enough background candidates for "
                f"{split}/{source}"
            )

        selected.extend(
            candidates[:allocation[source]]
        )

    rng.shuffle(selected)

    return selected


def write_csv(path, rows, fields):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Wrote {len(rows)} rows -> {path}"
    )


def audit_unknown(rows):

    print("\nUNKNOWN AUDIT")

    split_counts = Counter(
        r["split"] for r in rows
    )

    print("Split counts:", dict(split_counts))

    for split in ["train", "val", "test"]:

        subset = [
            r for r in rows
            if r["split"] == split
        ]

        labels = Counter(
            r["label"] for r in subset
        )

        speakers = {
            r["speaker_id"]
            for r in subset
        }

        print(
            f"{split}: "
            f"samples={len(subset)}, "
            f"speakers={len(speakers)}, "
            f"labels={len(labels)}"
        )

        print(
            "  label counts:",
            dict(sorted(labels.items()))
        )


def audit_background(rows):

    print("\nBACKGROUND AUDIT")

    split_counts = Counter(
        r["split"] for r in rows
    )

    print("Split counts:", dict(split_counts))

    for split in ["train", "val", "test"]:

        subset = [
            r for r in rows
            if r["split"] == split
        ]

        sources = Counter(
            r["source"] for r in subset
        )

        print(
            f"{split}: "
            f"samples={len(subset)}, "
            f"sources={dict(sorted(sources.items()))}"
        )

        for source in sorted(sources):

            indices = sorted(
                r["segment_index"]
                for r in subset
                if r["source"] == source
            )

            if split == "train":
                allowed = range(0, 42)
            elif split == "val":
                allowed = range(42, 51)
            else:
                allowed = range(51, 60)

            if not all(i in allowed for i in indices):
                raise RuntimeError(
                    f"TIME LEAKAGE detected: "
                    f"{split}/{source}"
                )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--speech_commands_root",
        required=True
    )

    parser.add_argument(
        "--background_dir",
        required=True
    )

    parser.add_argument(
        "--out_dir",
        required=True
    )

    parser.add_argument(
        "--n_train",
        type=int,
        default=121
    )

    parser.add_argument(
        "--n_val",
        type=int,
        default=26
    )

    parser.add_argument(
        "--n_test",
        type=int,
        default=26
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=26172
    )

    args = parser.parse_args()

    sc_root = Path(args.speech_commands_root)
    bg_root = Path(args.background_dir)
    out_dir = Path(args.out_dir)

    print("=" * 60)
    print("SIH26172 DATASET SELECTION")
    print("=" * 60)

    print("\nSeed:", args.seed)
    print(
        "Targets:",
        args.n_train,
        args.n_val,
        args.n_test
    )

    # --------------------------------------------------
    # UNKNOWN SPEECH
    # --------------------------------------------------

    pool = collect_unknown_pool(sc_root)

    unknown_rows = []

    for split, target in [
        ("train", args.n_train),
        ("val", args.n_val),
        ("test", args.n_test),
    ]:

        rows = select_unknown(
            pool,
            split,
            target,
            args.seed + {"train": 101, "val": 202, "test": 303}[split]
        )

        for row in rows:
            row["split"] = split

        unknown_rows.extend(rows)

    audit_unknown(unknown_rows)

    write_csv(
        out_dir / "unknown_selection_manifest.csv",
        unknown_rows,
        [
            "src_path",
            "label",
            "speaker_id",
            "split",
        ]
    )

    # --------------------------------------------------
    # BACKGROUND
    # --------------------------------------------------

    by_source = collect_background(bg_root)

    blocks = split_background_by_time(
        by_source
    )

    background_rows = []

    for split, target in [
        ("train", args.n_train),
        ("val", args.n_val),
        ("test", args.n_test),
    ]:

        rows = select_background(
            blocks,
            split,
            target,
            args.seed + {"train": 404, "val": 505, "test": 606}[split]
        )

        for row in rows:
            row["split"] = split

        background_rows.extend(rows)

    audit_background(background_rows)

    write_csv(
        out_dir / "background_selection_manifest.csv",
        background_rows,
        [
            "src_path",
            "source",
            "segment_index",
            "split",
        ]
    )

    print("\n" + "=" * 60)
    print("SELECTION COMPLETE")
    print("NO AUDIO FILES WERE COPIED OR MODIFIED.")
    print("=" * 60)


if __name__ == "__main__":
    main()