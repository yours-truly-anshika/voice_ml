import numpy as np
import librosa


SAMPLE_RATE = 16000

N_MELS = 40

FRAME_LENGTH_MS = 30
FRAME_STEP_MS = 20

FRAME_LENGTH = int(SAMPLE_RATE * FRAME_LENGTH_MS / 1000)
FRAME_STEP = int(SAMPLE_RATE * FRAME_STEP_MS / 1000)

N_FFT = 512

SUPPORTED_SAMPLE_COUNTS = (16000, 24000, 32000)


def extract_log_mel(audio):
    """
    Convert supported 16-kHz mono audio into a log-Mel tensor.

    Supported input lengths are 1.0s, 1.5s, and 2.0s, producing
    feature shapes (40, 49, 1), (40, 74, 1), and (40, 99, 1).
    """

    audio = np.asarray(audio, dtype=np.float32)

    # Dataset contract: every input must be exactly 1.0s, 1.5s, or 2.0s
    if len(audio) not in SUPPORTED_SAMPLE_COUNTS:
        raise ValueError(
            "Expected exactly 16000, 24000, or 32000 samples, "
            f"got {len(audio)}"
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

    return log_mel[..., np.newaxis]


if __name__ == "__main__":

    # One second of dummy audio.
    audio = np.zeros(
        SAMPLE_RATE,
        dtype=np.float32
    )

    features = extract_log_mel(audio)

    print("Final tensor shape:", features.shape)