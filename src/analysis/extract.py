"""Per-row codec-feature extraction with word-boundary slicing and pooling.

Closely mirrors ``extract_features.py`` in the juice500ml reference repo,
adapted to neural codecs: each unique ``path`` in the DataFrame is encoded
once, producing ``[L, T, D]`` hidden states; each row is then sliced by its
``(start, finish)`` seconds on the codec's own frame grid and pooled into
``[L, D]`` per row.
"""

from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..codecs.base import BaseCodec


def _frame_index(second: float, total_frames: int, audio_duration_s: float) -> int:
    """Convert a wall-clock time to a frame index on the codec's grid."""
    if audio_duration_s <= 0:
        return 0
    idx = int(round(second / audio_duration_s * total_frames))
    return min(max(idx, 0), total_frames - 1)


def _pool(feats: np.ndarray, pool: str) -> np.ndarray:
    if pool == "mean":
        return feats.mean(axis=0)
    if pool == "center":
        return feats[len(feats) // 2]
    raise ValueError(f"unknown pool {pool!r}")


def _slice_and_pool(
    feats_layered: np.ndarray, start_s: float, finish_s: float, audio_duration_s: float, pool: str
) -> np.ndarray:
    """Slice ``[L, T, D]`` features by time and pool over time → ``[L, D]``."""
    L, T, _D = feats_layered.shape
    a = _frame_index(start_s, T, audio_duration_s)
    b = _frame_index(finish_s, T, audio_duration_s)
    if b <= a:
        b = a + 1
    return np.stack([_pool(feats_layered[l, a:b], pool) for l in range(L)], axis=0)


def _audio_duration(path: str) -> float:
    import soundfile as sf

    info = sf.info(path)
    return info.frames / float(info.samplerate)


def extract_codec_features(
    codec: BaseCodec,
    df: pd.DataFrame,
    pool: str = "mean",
    feat_column: str = "feat",
    progress: bool = True,
) -> pd.DataFrame:
    """Encode each unique audio path with ``codec`` and attach a per-row pooled
    feature ``feat_column`` of shape ``[L, D]`` to a copy of ``df``."""
    out = df.copy()
    out[feat_column] = None

    paths = sorted(out.path.unique())
    iterator = tqdm(paths, desc=f"{codec.name}: encode") if progress else paths
    for path in iterator:
        feats, _T, _D = codec.extract_hidden_states(path)
        duration = _audio_duration(path)
        for idx, row in out[out.path == path].iterrows():
            out.at[idx, feat_column] = _slice_and_pool(
                feats, float(row.start), float(row.finish), duration, pool
            )
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
