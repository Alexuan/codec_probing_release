"""Shared mono+resample loader for the codec wrappers."""

import numpy as np
import soundfile as sf


def load_mono_resampled(path: str, target_sr: int) -> np.ndarray:
    """Read audio with soundfile, average to mono, resample to ``target_sr``."""
    import librosa

    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)
