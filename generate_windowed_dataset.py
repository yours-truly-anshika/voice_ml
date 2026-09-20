import os
import csv
import math
import wave
import struct
import hashlib
import random
from pathlib import Path

# CONFIGURATION
SAMPLE_RATE = 16000
DURATIONS = [1.0, 1.5, 2.0]

ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
MANIFEST_PATH = DATASET_DIR / "processed" / "authoritative_dataset_manifest.csv"
ENDPOINT_PATH = DATASET_DIR / "processed" / "positive_endpoint_analysis.csv"
SYNTH_UNKNOWN_MANIFEST = DATASET_DIR / "synthetic" / "train" / "synthetic_unknown_manifest.csv"

def read_audio(path):
    """
    Reads a mono 16kHz WAV file.
    Returns (samples_array, duration_sec)
    Returns None if file is missing, corrupted, or an LFS pointer.
    """
    try:
        # Check if it's an LFS stub by reading first few bytes
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(100)
            if 'git-lfs' in content:
                return None
                
        with wave.open(str(path), 'rb') as w:
            n_frames = w.getnframes()
            frames = w.readframes(n_frames)
            samples = struct.unpack(f"<{n_frames}h", frames)
            return list(samples), n_frames / w.getframerate()
    except Exception:
        return None

def write_audio(path, samples, window_samples):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # ensure exactly window_samples
    if len(samples) > window_samples:
        samples = samples[:window_samples]
    elif len(samples) < window_samples:
        samples = samples + [0] * (window_samples - len(samples))
        
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        frames = struct.pack(f"<{len(samples)}h", *[int(s) for s in samples])
        w.writeframes(frames)
        
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def extract_window(samples, start_sec, duration_sec):
    window_samples = int(duration_sec * SAMPLE_RATE)
    start_sample = int(start_sec * SAMPLE_RATE)
    end_sample = start_sample + window_samples
    start_sample = max(0, min(start_sample, len(samples)))
    end_sample = max(0, min(end_sample, len(samples)))
    
    win = samples[start_sample:end_sample]
    # pad if necessary
    if len(win) < window_samples:
        win = win + [0]*(window_samples - len(win))
    return win, start_sample, end_sample

def get_hash(path):
    try:
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except:
        return ""

def main():
    if not MANIFEST_PATH.exists():
        print(f"Manifest not found: {MANIFEST_PATH}")
        return
        
    endpoints = {}
    if ENDPOINT_PATH.exists():
        with open(ENDPOINT_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                endpoints[row['filename']] = {
                    'onset': float(row['speech_onset_sec']),
                    'offset': float(row['speech_offset_sec']),
                    'duration': float(row['duration_sec'])
                }
                
    synth_unknown = {}
    if SYNTH_UNKNOWN_MANIFEST.exists():
        with open(SYNTH_UNKNOWN_MANIFEST, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                synth_unknown[row['synthetic_filename']] = float(row['shift_ms'])
                
    dataset_manifest_path = DATASET_DIR / "processed" / "split_manifest.csv"
    positive_map = {}
    if dataset_manifest_path.exists():
        with open(dataset_manifest_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ep_key = f"{row['ParticipantID']}_{row['NoteType']}_{row['SequenceNumber']}.wav"
                positive_map[row['Filename']] = ep_key
                
    out_rows_by_dur = {d: [] for d in DURATIONS}
    
    processed_count = 0
    skipped_count = 0
    
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row['split']
            cls = row['class']
            src_type = row['source_type']
            filename = row['filename']
            
            raw_path = row['full_path'].replace('\\', '/')
            if 'Anshika/voice_ml' in raw_path:
                raw_path = raw_path.split('Anshika/voice_ml/')[-1]
            full_path = ROOT_DIR / raw_path
            
            res = read_audio(full_path)
            if res is None:
                skipped_count += 1
                continue
                
            samples, duration = res
            src_sha256 = get_hash(full_path)
            processed_count += 1
            
            # For positive class, get endpoints
            if cls == 'positive':
                if src_type == 'real':
                    ep_key = positive_map.get(filename, filename)
                else:
                    ep_key = None
                ep = endpoints.get(ep_key, {'onset': 1.0, 'offset': 2.0, 'duration': duration})
                onset = ep['onset']
                offset = ep['offset']
            
            for dur in DURATIONS:
                out_dir = DATASET_DIR / f"windowed_{dur}s"
                out_dir.mkdir(parents=True, exist_ok=True)
                
                windows_to_gen = []
                if cls == 'positive':
                    if split == 'train':
                        start_early = max(0.0, onset - 0.1)
                        windows_to_gen.append(('positive_early_partial', start_early, 0))
                        
                        center = (onset + offset) / 2
                        start_center = max(0.0, center - dur/2)
                        windows_to_gen.append(('positive_center', start_center, 1))
                        
                        end_near_end = min(duration, offset + 0.1)
                        start_near_end = max(0.0, end_near_end - dur)
                        windows_to_gen.append(('positive_near_end', start_near_end, 2))
                    else:
                        end_eval = min(duration, offset + 0.1)
                        start_eval = max(0.0, end_eval - dur)
                        windows_to_gen.append(('positive_eval_near_end', start_eval, 0))
                        
                elif cls == 'unknown':
                    if src_type == 'real':
                        windows_to_gen.append(('unknown_real_pass', 0.0, 0))
                    else:
                        shift_ms = synth_unknown.get(filename, 0.0)
                        center = 1.5 + (shift_ms / 1000.0)
                        start = max(0.0, center - dur/2)
                        windows_to_gen.append(('unknown_synthetic_center', start, 0))
                        
                elif cls == 'background':
                    random.seed(filename)
                    max_start = max(0.0, duration - dur)
                    start = random.uniform(0, max_start)
                    windows_to_gen.append(('background_seeded_crop', start, 0))
                    
                for strategy, start_sec, idx in windows_to_gen:
                    win_samples, s_start, s_end = extract_window(samples, start_sec, dur)
                    win_filename = f"{Path(filename).stem}_w{idx}.wav"
                    out_path = out_dir / split / cls / win_filename
                    
                    win_sha256 = write_audio(out_path, win_samples, int(dur * SAMPLE_RATE))
                    
                    speech_fraction = ""
                    speech_start_s = ""
                    speech_end_s = ""
                    fits_inside = ""
                    
                    if cls == 'positive':
                        w_start_s = start_sec
                        w_end_s = start_sec + dur
                        ep_onset = onset
                        ep_offset = offset
                        speech_start_s = round(ep_onset, 4)
                        speech_end_s = round(ep_offset, 4)
                        
                        overlap_start = max(w_start_s, ep_onset)
                        overlap_end = min(w_end_s, ep_offset)
                        if overlap_end > overlap_start:
                            speech_fraction = (overlap_end - overlap_start) / (ep_offset - ep_onset)
                            speech_fraction = round(min(1.0, speech_fraction), 4)
                        else:
                            speech_fraction = 0.0
                            
                        # Complete annotated speech fits inside window
                        fits_inside = "True" if (w_start_s <= ep_onset and w_end_s >= ep_offset) else "False"
                            
                    out_rows_by_dur[dur].append({
                        'window_id': f"{Path(filename).stem}_{idx}",
                        'source_file': filename,
                        'source_path': str(full_path.relative_to(ROOT_DIR)).replace('\\', '/'),
                        'window_path': str(out_path.relative_to(ROOT_DIR)).replace('\\', '/'),
                        'split': split,
                        'class': cls,
                        'source_duration_s': round(duration, 4),
                        'window_duration_s': dur,
                        'window_start_sample': s_start,
                        'window_end_sample': s_start + len(win_samples),
                        'window_start_s': round(start_sec, 4),
                        'window_end_s': round(start_sec + dur, 4),
                        'window_strategy': strategy,
                        'window_index': idx,
                        'source_sha256': src_sha256,
                        'window_sha256': win_sha256,
                        'speech_start_s': speech_start_s,
                        'speech_end_s': speech_end_s,
                        'speech_fraction': speech_fraction,
                        'fits_inside': fits_inside
                    })
                    
    fieldnames = ['window_id', 'source_file', 'source_path', 'window_path', 'split', 'class', 
                  'source_duration_s', 'window_duration_s', 'window_start_sample', 'window_end_sample',
                  'window_start_s', 'window_end_s', 'window_strategy', 'window_index',
                  'source_sha256', 'window_sha256', 'speech_start_s', 'speech_end_s', 'speech_fraction', 'fits_inside']
                  
    for dur in DURATIONS:
        manifest_dir = DATASET_DIR / f"windowed_{dur}s"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"windowed_dataset_manifest_{dur}s.csv"
        with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in out_rows_by_dur[dur]:
                writer.writerow(row)
                
    print(f"Processed: {processed_count}")
    print(f"Skipped (LFS/missing): {skipped_count}")
    for dur in DURATIONS:
        print(f"Views for {dur}s: {len(out_rows_by_dur[dur])}")

if __name__ == "__main__":
    main()
