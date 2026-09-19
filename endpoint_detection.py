"""
endpoint_detection.py

Runs ffmpeg silencedetect across every WAV file in a directory to estimate
the true speech onset/offset (endpoint) for each recording, then reports
the distribution of speech-span durations across the whole set.

Purpose (SIH26172 / activate_orbit KWS):
    Decide the correct fixed model-input window length for the positive
    "activate orbit" class, instead of blindly cropping to 1 second and
    risking cutting the keyword.

Usage:
    python endpoint_detection.py \
        --input_dir "C:\\Users\\Anshika\\voice_ml\\dataset\\processed\\positive" \
        --output_csv "C:\\Users\\Anshika\\voice_ml\\dataset\\processed\\positive_endpoints.csv" \
        --noise_db -35 \
        --min_silence_dur 0.2

Requires: ffmpeg and ffprobe on PATH. Python 3.8+, no third-party
dependencies beyond the standard library (stats printed manually;
pandas is used only if available, otherwise falls back to stdlib).
"""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

SILENCE_START_RE = re.compile(r"silence_start:\s*([0-9.]+)")
SILENCE_END_RE = re.compile(r"silence_end:\s*([0-9.]+)")


def get_duration(path: Path) -> float:
    """Return duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def get_silence_intervals(path: Path, noise_db: float, min_dur: float):
    """
    Run ffmpeg silencedetect and parse silence_start/silence_end pairs
    from stderr. Returns a list of (start, end) tuples, sorted.
    Note: if the file ends while still "silent", ffmpeg may not emit a
    matching silence_end -- that case is handled by the caller using
    total duration as an implicit closing boundary.
    """
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    stderr = result.stderr

    starts = [float(m) for m in SILENCE_START_RE.findall(stderr)]
    ends = [float(m) for m in SILENCE_END_RE.findall(stderr)]

    intervals = []
    # Pair them in order; if the last start has no matching end,
    # treat the file's end as the implicit end (handled by caller).
    for i, s in enumerate(starts):
        if i < len(ends):
            intervals.append((s, ends[i]))
        else:
            intervals.append((s, None))  # unresolved, closed later
    return intervals


def compute_onset_offset(intervals, duration, edge_tolerance=0.05):
    """
    Given silence intervals and total duration, compute:
      onset  = time the actual speech begins (end of leading silence, else 0)
      offset = time the actual speech ends (start of trailing silence, else duration)
    Internal silence intervals (pauses between words, e.g. "activate" <pause> "orbit")
    are NOT used to trim -- they are part of the phrase and must be preserved.
    """
    # Resolve any unresolved trailing interval to file duration
    resolved = [(s, e if e is not None else duration) for s, e in intervals]
    resolved.sort(key=lambda x: x[0])

    onset = 0.0
    offset = duration

    if resolved:
        first_s, first_e = resolved[0]
        if first_s <= edge_tolerance:
            onset = first_e

        last_s, last_e = resolved[-1]
        if last_e >= duration - edge_tolerance:
            offset = last_s

    # Safety: never let offset <= onset
    if offset <= onset:
        onset, offset = 0.0, duration

    return onset, offset, resolved


def percentile(sorted_vals, pct):
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def summarize(name, values):
    vals = sorted(v for v in values if v == v)  # drop NaN
    if not vals:
        print(f"{name}: no data")
        return
    n = len(vals)
    mean = sum(vals) / n
    print(f"\n{name} (n={n})")
    print(f"  min    : {vals[0]:.3f}")
    print(f"  p50    : {percentile(vals, 50):.3f}")
    print(f"  mean   : {mean:.3f}")
    print(f"  p90    : {percentile(vals, 90):.3f}")
    print(f"  p95    : {percentile(vals, 95):.3f}")
    print(f"  p99    : {percentile(vals, 99):.3f}")
    print(f"  max    : {vals[-1]:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input_dir", required=True, help="Directory of positive WAV files")
    ap.add_argument("--output_csv", required=True, help="Path to write per-file results CSV")
    ap.add_argument("--noise_db", type=float, default=-35.0,
                     help="silencedetect noise threshold in dB (default -35)")
    ap.add_argument("--min_silence_dur", type=float, default=0.2,
                     help="minimum silence duration in seconds to count as a gap (default 0.2)")
    ap.add_argument("--pattern", default="*.wav", help="glob pattern for input files")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob(args.pattern))
    if not files:
        print(f"No files matched {args.pattern} in {input_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    failures = []

    for i, f in enumerate(files, 1):
        try:
            duration = get_duration(f)
            intervals = get_silence_intervals(f, args.noise_db, args.min_silence_dur)
            onset, offset, resolved = compute_onset_offset(intervals, duration)
            span = offset - onset
            internal_gaps = max(0, len(resolved) - (
                (1 if resolved and resolved[0][0] <= 0.05 else 0)
                + (1 if resolved and resolved[-1][1] >= duration - 0.05 else 0)
            ))
            rows.append({
                "filename": f.name,
                "duration_s": round(duration, 3),
                "onset_s": round(onset, 3),
                "offset_s": round(offset, 3),
                "speech_span_s": round(span, 3),
                "num_silence_intervals": len(resolved),
                "num_internal_pauses": internal_gaps,
            })
        except Exception as e:
            failures.append((f.name, str(e)))

        if i % 20 == 0 or i == len(files):
            print(f"Processed {i}/{len(files)}...", file=sys.stderr)

    # Write CSV
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "filename", "duration_s", "onset_s", "offset_s",
            "speech_span_s", "num_silence_intervals", "num_internal_pauses",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {out_path}")
    if failures:
        print(f"\n{len(failures)} files failed processing:")
        for name, err in failures:
            print(f"  {name}: {err}")

    # Summary distributions -- this is what decides your model input window
    summarize("duration_s (raw file length)", [r["duration_s"] for r in rows])
    summarize("speech_span_s (onset->offset, keeps internal pauses)",
              [r["speech_span_s"] for r in rows])
    summarize("onset_s (leading silence trimmed)", [r["onset_s"] for r in rows])
    summarize("offset_s", [r["offset_s"] for r in rows])

    spans = sorted(r["speech_span_s"] for r in rows)
    if spans:
        p95 = percentile(spans, 95)
        print(f"\nSuggested fixed model-input window (p95 speech_span, rounded up to nearest 0.1s):")
        import math
        suggested = math.ceil(p95 * 10) / 10.0
        print(f"  -> {suggested:.1f} s")
        print("  (Use this instead of 1.0 s for windowing/padding logic, "
              "then recompute spectrogram frame count accordingly.)")


if __name__ == "__main__":
    main()