import numpy as np
import librosa


SAMPLE_RATE = 16000

N_MELS = 40

FRAME_LENGTH_MS = 30
FRAME_STEP_MS = 20

FRAME_LENGTH = int(SAMPLE_RATE * FRAME_LENGTH_MS / 1000)
FRAME_STEP = int(SAMPLE_RATE * FRAME_STEP_MS / 1000)

N_FFT = 512


def extract_log_mel(audio):
    """
    Convert 1 second of 16-kHz mono audio into
    a 40 x 49 x 1 log-Mel feature tensor.
    """

    audio = np.asarray(audio, dtype=np.float32)

    # Dataset contract: every input must already be exactly 1 second.
    if len(audio) != SAMPLE_RATE:
        raise ValueError(
            f"Expected exactly {SAMPLE_RATE} samples, got {len(audio)}"
        )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=FRAME_STEP,
        win_length=FRAME_LENGTH,
        window="hann",
        center=False,
        n_mels=N_MELS,
        fmin=20,
        fmax=SAMPLE_RATE // 2,
        power=2.0,
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Expected: (40, 49)
    print("Feature shape:", log_mel.shape)

    return log_mel[..., np.newaxis]


if __name__ == "__main__":

    # One second of dummy audio.
    audio = np.zeros(
        SAMPLE_RATE,
        dtype=np.float32
    )

    features = extract_log_mel(audio)

    print("Final tensor shape:", features.shape)