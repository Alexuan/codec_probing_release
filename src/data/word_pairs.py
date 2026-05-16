"""LibriSpeech word-pair construction.

The heavy lifting (DataFrame construction, synonym / homophone maps, samplers)
is a direct port of https://github.com/juice500ml/phonetic_semantic_probing
under Apache 2.0 — see :mod:`src.data._juice500ml`. This module simply
re-exports those entry points plus two ergonomic helpers used by the
release's notebooks.
"""

from typing import List, Optional, Tuple

import pandas as pd

from ._juice500ml import (  # noqa: F401 — re-exported
    build_librispeech_dataframe,
    build_wordmaps,
    filter_df,
    phonetic_dist,
    samplers,
)


def sample_pairs_from_map(
    df: pd.DataFrame,
    pair_map: dict,
    n_pairs: int,
    seed: int = 0,
) -> List[Tuple[int, int]]:
    """Random sample of ``n_pairs`` ``(row_a, row_b)`` occurrences where
    ``row_b.text`` is in ``pair_map[row_a.text]``.

    Used in the same way for synonym and homophone maps.
    """
    import random

    rng = random.Random(seed)
    by_word = {t: g.index.tolist() for t, g in df.groupby("text")}
    keys = [w for w, vs in pair_map.items() if vs and w in by_word]
    if not keys:
        return []

    pairs: List[Tuple[int, int]] = []
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 20:
        attempts += 1
        w = rng.choice(keys)
        v = rng.choice(list(pair_map[w]))
        if v not in by_word:
            continue
        pairs.append((rng.choice(by_word[w]), rng.choice(by_word[v])))
    return pairs


def sample_random_pairs(df: pd.DataFrame, n_pairs: int, seed: int = 0) -> List[Tuple[int, int]]:
    """Random baseline: ``n_pairs`` occurrences with different surface words."""
    import random

    rng = random.Random(seed)
    indices = df.index.tolist()
    pairs: List[Tuple[int, int]] = []
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 20:
        attempts += 1
        a, b = rng.sample(indices, 2)
        if df.loc[a].text == df.loc[b].text:
            continue
        pairs.append((a, b))
    return pairs


__all__ = [
    "build_librispeech_dataframe",
    "build_wordmaps",
    "filter_df",
    "phonetic_dist",
    "samplers",
    "sample_pairs_from_map",
    "sample_random_pairs",
]
