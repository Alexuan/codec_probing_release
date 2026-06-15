"""Optionally pre-cut word-level audio segments from LibriSpeech wavs.

This is *not* required for the probing experiments — the feature extractor
slices on the frame grid directly. Use these helpers only if you want
standalone ``.wav`` files for one-off auditing.
"""

import os
from typing import List

import pandas as pd


def cut_segments(
    df: pd.DataFrame,
    output_dir: str,
    target_sr: int = 24000,
    overwrite: bool = False,
) -> List[str]:
    """Cut each ``(path, start, finish, text)`` row into ``output_dir/<text>/<row_id>.wav``.

    Returns the list of written paths in the same order as ``df`` rows.
    """
    import librosa
    import soundfile as sf

    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for idx, row in df.iterrows():
        word_dir = os.path.join(output_dir, row.text)
        os.makedirs(word_dir, exist_ok=True)
        out = os.path.join(word_dir, f"row{idx:08d}.wav")
        if overwrite or not os.path.exists(out):
            audio, _ = librosa.load(
                row.path,
                sr=target_sr,
                offset=float(row.start),
                duration=max(float(row.finish) - float(row.start), 0.01),
            )
            sf.write(out, audio, target_sr)
        paths.append(out)
    return paths
