import wave

import numpy as np


def save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    """
    Export as 16-bit PCM WAV.
    """
    with wave.open(path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        f.writeframes(pcm.tobytes())