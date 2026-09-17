import librosa

# Replace with your actual file name
audio_file = "YOUR_FILE.webm" 

y, sr = librosa.load(audio_file, sr=None)
duration = librosa.get_duration(y=y, sr=sr)

print("Sample Rate:", sr)
print("Channels:", 1 if y.ndim == 1 else y.shape[0])
print("Duration (seconds):", duration)
print("Total Samples:", len(y) if y.ndim == 1 else y.shape[1])