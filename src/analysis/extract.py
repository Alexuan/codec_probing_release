"""Per-word codec-feature extraction with **audio slicing** + time pooling.

Mirrors the verified word-pair pipeline (``extract_features*.py`` with
``slice=True``): for each MFA-aligned word we slice the *audio* to
``[start, finish]``, encode that word's audio with the codec, and pool over all
of its frames into ``[L, D]`` for the row.

This is NOT the same as encoding the whole utterance once and slicing the
*features* afterwards — because of the encoder's receptive field / padding the
two give materially different numbers, so feature-slicing is intentionally not
used here.
"""

import os
import tempfile

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..codecs.base import BaseCodec


def _pool(feats: np.ndarray, pool: str) -> np.ndarray:
    """Pool a single layer's ``[T, D]`` frames over time -> ``[D]``."""
    if pool == "mean":
        return feats.mean(axis=0)
    if pool == "center":
        return feats[len(feats) // 2]
    raise ValueError(f"unknown pool {pool!r}")


def extract_codec_features(
    codec: BaseCodec,
    df: pd.DataFrame,
    pool: str = "mean",
    feat_column: str = "feat",
    progress: bool = True,
) -> pd.DataFrame:
    """Attach a per-row pooled feature ``feat_column`` of shape ``[L, D]``.

    For each row the audio is read at its native sample rate, sliced to the
    word's ``[start, finish]`` window, written to a temporary wav, and encoded
    by ``codec`` (which loads + resamples it exactly as it would any audio
    file). Rows whose file is unreadable or whose slice yields no frames are
    skipped (the dev pipeline's "filtered due to empty tokens" behaviour).
    """
    import soundfile as sf

    out = df.copy()
    out[feat_column] = None

    audio_cache: dict = {}   # path -> (waveform[native sr], sr) or None
    n_skipped = 0

    rows = out.itertuples()
    if progress:
        rows = tqdm(rows, total=len(out), desc=f"{codec.name}: slice+encode")

    for row in rows:
        path = row.path
        if path not in audio_cache:
            try:
                wav, sr = sf.read(path, dtype="float32")
                if getattr(wav, "ndim", 1) > 1:
                    wav = wav.mean(axis=1)   # to mono
                audio_cache[path] = (wav, sr)
            except Exception as e:
                audio_cache[path] = None
                print(f"  skip (cannot read {path}): {type(e).__name__}: {str(e)[:80]}")
        rec = audio_cache[path]
        if rec is None:
            n_skipped += 1
            continue

        wav, sr = rec
        a = max(int(float(row.start) * sr), 0)
        b = min(int(float(row.finish) * sr), len(wav))
        seg = wav[a:b]
        if seg.size == 0:
            n_skipped += 1
            continue

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp = tf.name
        try:
            sf.write(tmp, seg, sr)
            feats, T, _D = codec.extract_hidden_states(tmp)   # [L, T, D] of the word's audio
        except Exception as e:
            n_skipped += 1
            print(f"  skip (encode failed, {type(e).__name__}) {os.path.basename(path)} [{row.start:.2f},{row.finish:.2f}]")
            continue
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

        if T == 0:
            n_skipped += 1
            continue
        out.at[row.Index, feat_column] = np.stack(
            [_pool(feats[l], pool) for l in range(feats.shape[0])], axis=0
        )  # [L, D]

    if n_skipped:
        print(f"skipped {n_skipped}/{len(out)} rows (unreadable file / empty slice / zero frames)")
    return out


def pair_layer_distances(
    df_with_feat: pd.DataFrame,
    pairs: list,
    feat_column: str = "feat",
) -> np.ndarray:
    """Compute ``[N_pairs, L]`` Euclidean distances for a list of ``(idx_a, idx_b)``."""
    rows = []
    for a, b in pairs:
        fa = df_with_feat.at[a, feat_column]
        fb = df_with_feat.at[b, feat_column]
        if fa is None or fb is None:
            continue
        rows.append(np.linalg.norm(fa - fb, axis=1))
    return np.asarray(rows)
