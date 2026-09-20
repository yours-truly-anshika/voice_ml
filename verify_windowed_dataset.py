import csv
import sys
import wave
import struct
import math
import statistics
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DURATIONS = [1.0, 1.5, 2.0]

def verify_duration(dur):
    manifest_path = ROOT_DIR / "dataset" / f"windowed_{dur}s" / f"windowed_dataset_manifest_{dur}s.csv"
    if not manifest_path.exists():
        print(f"\n--- {dur}s: Error: Manifest {manifest_path} does not exist. ---")
        return False
        
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
        
    generated_files = set()
    source_to_splits = {}
    valid = True
    
    # Coverage metrics
    total_positive = 0
    fits_inside_count = 0
    truncation_count = 0
    speech_overlaps = []
    speech_spans = []
    
    expected_frames = int(dur * 16000)
    
    for row in reader:
        win_path = ROOT_DIR / row['window_path']
        if not win_path.exists():
            print(f"Error: File missing {win_path}")
            valid = False
            continue
            
        generated_files.add(str(win_path))
        
        # Audio check
        try:
            with wave.open(str(win_path), 'rb') as w:
                if w.getframerate() != 16000:
                    print(f"Error: WRONG SR {win_path}")
                    valid = False
                if w.getnchannels() != 1:
                    print(f"Error: WRONG CHANNELS {win_path}")
                    valid = False
                if w.getnframes() != expected_frames:
                    print(f"Error: WRONG FRAMES {win_path} (got {w.getnframes()}, expected {expected_frames})")
                    valid = False
        except Exception as e:
            print(f"Error reading {win_path}: {e}")
            valid = False
            
        # Splits
        src_path = row['source_path']
        split = row['split']
        if src_path not in source_to_splits:
            source_to_splits[src_path] = set()
        source_to_splits[src_path].add(split)
        
        if row['class'] == 'positive':
            total_positive += 1
            if row['fits_inside'] == 'True':
                fits_inside_count += 1
            else:
                truncation_count += 1
                
            if row['speech_fraction']:
                speech_overlaps.append(float(row['speech_fraction']))
                
            if row['speech_start_s'] and row['speech_end_s']:
                span = float(row['speech_end_s']) - float(row['speech_start_s'])
                speech_spans.append(span)
        
    # Check if any source crosses splits
    for src_path, splits in source_to_splits.items():
        if len(splits) > 1:
            print(f"Error: Source {src_path} crosses splits: {splits}")
            valid = False
            
    # Manifest row count vs generated files
    if len(reader) != len(generated_files):
        print(f"Error: Manifest has {len(reader)} rows but {len(generated_files)} unique files.")
        valid = False
        
    print(f"\n--- {dur}s Verification ---")
    if valid:
        print("PASS")
    else:
        print("FAIL")
        
    print(f"Total windows generated: {len(reader)}")
    
    if total_positive > 0:
        pct_complete = (fits_inside_count / total_positive) * 100
        mean_overlap = sum(speech_overlaps) / len(speech_overlaps) if speech_overlaps else 0.0
        min_span = min(speech_spans) if speech_spans else 0.0
        max_span = max(speech_spans) if speech_spans else 0.0
        median_span = statistics.median(speech_spans) if speech_spans else 0.0
        
        print("\nCoverage Summary:")
        print(f"Total positive views: {total_positive}")
        print(f"Fits completely inside window: {fits_inside_count} ({pct_complete:.1f}%)")
        print(f"Requires truncation: {truncation_count}")
        print(f"Mean speech-span overlap: {mean_overlap:.4f}")
        print(f"Speech span (min/median/max): {min_span:.4f}s / {median_span:.4f}s / {max_span:.4f}s")
        
    return valid

def verify():
    all_valid = True
    for dur in DURATIONS:
        if not verify_duration(dur):
            all_valid = False
            
    print("\nOVERALL VERIFICATION:", "PASS" if all_valid else "FAIL")

if __name__ == "__main__":
    verify()
