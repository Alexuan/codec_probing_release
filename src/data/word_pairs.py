"""LibriSpeech word-pair construction for semantic vs. phonetic probing.

From LibriSpeech audio + Montreal Forced Aligner (MFA) alignments this module
builds:

* a per-word-occurrence ``DataFrame`` with columns
  ``text / start / finish / path / phones / synonyms / speaker``
  (:func:`build_librispeech_dataframe`), and
* **synonym** and **near-homophone** word-pair maps
  (:func:`build_wordmaps`): two words are *synonyms* if they share a WordNet
  sense yet differ phonetically (phoneme-level edit distance above a
  threshold), and *near-homophones* if their CMU-dict phoneme sequences are
  within that threshold while not being synonyms.

The synonym / near-homophone contrast follows the word-pair probing
methodology of Choi et al., *"Self-Supervised Speech Representations are More
Phonetic than Semantic"* (Interspeech 2024) — see the README references.
"""

from __future__ import annotations

import random
import re
from functools import partial
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


# --------------------------------------------------------------------------- #
# Phonetic distance + sampling-frame filtering
# --------------------------------------------------------------------------- #
def phonetic_dist(x: Sequence[str], y: Sequence[str]) -> float:
    """Phoneme-level word error rate between two phoneme sequences."""
    from jiwer import wer  # lazy: only needed for Tier-2 word-map construction

    ref, hyp = (x, y) if len(x) > len(y) else (y, x)
    return wer(reference=" ".join(ref), hypothesis=" ".join(hyp))


def filter_df(df: pd.DataFrame, speaker: Optional[str], n_sample: Optional[int],
              language_uniform: bool, seed: int) -> pd.DataFrame:
    """Subset the occurrence DataFrame by speaker / random sample (seeded)."""
    print(f"Original size: {len(df)}")
    if language_uniform:
        assert speaker is None
        langs = df["language"].unique()
        df = pd.concat([df[df.language == lang].sample(n_sample, random_state=seed) for lang in langs])
    else:
        if speaker is not None:
            df = df[df.speaker == speaker]
        if n_sample is not None:
            df = df.sample(n_sample, random_state=seed)
    print(f"Filtered size: {len(df)}")
    return df


# --------------------------------------------------------------------------- #
# Per-word occurrence DataFrame from MFA TextGrids
# --------------------------------------------------------------------------- #
def _phones_for(word: str, cmu_cache) -> Optional[List[str]]:
    """CMU-dict phoneme sequence for ``word`` (stress markers stripped)."""
    if word not in cmu_cache:
        return None
    return [re.sub(r"\d+", "", p) for p in cmu_cache[word][0]]


def _synonyms_for(word: str, lang: str = "eng") -> set:
    """WordNet synonyms (lemma names) for ``word``, excluding the word itself."""
    import nltk
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    from nltk.corpus import wordnet

    return {
        syn
        for synsets in wordnet.synsets(word, lang=lang)
        for syn in synsets.lemma_names("eng")
        if syn != word
    }


def build_librispeech_dataframe(
    dataset_path: Path, textgrid_path: Path,
    splits: Sequence[str] = ("dev-clean", "test-clean"),
) -> pd.DataFrame:
    """One row per MFA-aligned word occurrence that has CMU phones + >=1 synonym."""
    import cmudict
    from textgrids import TextGrid

    dataset_path, textgrid_path = Path(dataset_path), Path(textgrid_path)
    cmu_cache = cmudict.dict()

    rows = []
    for split in splits:
        for p in tqdm(list(textgrid_path.glob(f"{split}/*/*/*.TextGrid")), desc=split):
            grid = TextGrid(p)
            for word in grid["words"]:
                phones = _phones_for(word.text, cmu_cache)
                synonyms = list(_synonyms_for(word.text))
                if phones is not None and synonyms:
                    rows.append({
                        "text": str(word.text),
                        "start": word.xmin,
                        "finish": word.xmax,
                        "path": str((dataset_path / p.relative_to(p.parents[3]).with_suffix(".flac")).absolute()),
                        "phones": phones,
                        "synonyms": synonyms,
                        "speaker": p.parents[1].name,
                    })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Synonym / near-homophone maps
# --------------------------------------------------------------------------- #
def _build_synonym_map(words, df, text2phones, threshold: float = 0.4) -> dict:
    synonym_map = {}
    for index in tqdm(df.reset_index().groupby(["text"])["index"].min()):
        row = df.loc[index]
        synonyms = []
        for s in row.synonyms:
            if threshold < 0:
                synonyms.append(s)
            elif (s in words) and (phonetic_dist(row.phones, text2phones[s]) > threshold):
                synonyms.append(s)
        synonyms = set(synonyms).intersection(words)
        if synonyms:
            synonym_map[row.text] = synonyms
    return synonym_map


def _near_homophones_for_word(df, indices, text2phones, synonym_map, word, threshold: float = 0.4) -> set:
    homophones = []
    for index in indices:
        row = df.loc[index]
        if (row.text != word
                and row.text not in synonym_map.get(word, set())
                and word not in synonym_map.get(row.text, set())
                and 0.0 < phonetic_dist(text2phones[word], row.phones) <= threshold):
            homophones.append(row.text)
    return set(homophones)


def _build_homophone_map(words, synonym_map, df, text2phones, num_workers: int) -> dict:
    words = list(words)
    chunksize = max(1, len(words) // max(num_workers * 4, 1))
    indices = df.reset_index().groupby(["text"])["index"].min()
    homophones_list = process_map(
        partial(_near_homophones_for_word, df[["text", "phones"]], indices, text2phones, synonym_map),
        words, max_workers=num_workers, chunksize=chunksize,
    )
    return {w: hs for w, hs in zip(words, homophones_list) if hs}


def build_wordmaps(
    df: pd.DataFrame, threshold: float = 0.4, num_workers: int = 64,
    speaker: Optional[str] = None, n_sample: Optional[int] = None, seed: int = 0,
) -> dict:
    """Return ``{'synonym_map': {word: {...}}, 'homophone_map': {word: {...}}}``."""
    filtered = filter_df(df, speaker, n_sample, language_uniform=False, seed=seed)
    words = set(filtered.text.unique())
    text2phones = {row.text: tuple(row.phones) for row in filtered.itertuples()}

    synonym_map = _build_synonym_map(words, filtered, text2phones, threshold=threshold)
    print(f"Synonym pairs: {sum(len(v) for v in synonym_map.values())}")

    # Unfiltered synonym set is used only to exclude pairs from the homophone search.
    unfiltered_synonyms = _build_synonym_map(words, filtered, text2phones, threshold=-1)
    homophone_map = _build_homophone_map(words, unfiltered_synonyms, filtered, text2phones, num_workers)
    print(f"Near-homophone pairs: {sum(len(v) for v in homophone_map.values())}")

    return {"synonym_map": synonym_map, "homophone_map": homophone_map}


# --------------------------------------------------------------------------- #
# Pair samplers
# --------------------------------------------------------------------------- #
def _random_sampler(df, wordmap):
    l = df.index.to_numpy(); r = l.copy()
    np.random.default_rng(seed=42).shuffle(r)
    for a, b in zip(l, r):
        if a != b:
            yield a, b


def _synonym_sampler(df, wordmap):
    for a in df.index.to_numpy():
        syn = set(wordmap["synonym_map"].get(df.loc[a].text, set()))
        for b in df[df.text.isin(syn)].index:
            if a != b:
                yield a, b


def _homophone_sampler(df, wordmap):
    for a in df.index.to_numpy():
        hom = wordmap["homophone_map"].get(df.loc[a].text, set())
        for b in df[df.text.isin(hom)].index:
            if a != b:
                yield a, b


samplers = {"random": _random_sampler, "synonym": _synonym_sampler, "homophone": _homophone_sampler}


def sample_pairs_from_map(df: pd.DataFrame, pair_map: dict, n_pairs: int, seed: int = 0) -> List[Tuple[int, int]]:
    """Sample ``n_pairs`` ``(row_a, row_b)`` occurrences with ``b.text in pair_map[a.text]``."""
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
        if v in by_word:
            pairs.append((rng.choice(by_word[w]), rng.choice(by_word[v])))
    return pairs


def sample_random_pairs(df: pd.DataFrame, n_pairs: int, seed: int = 0) -> List[Tuple[int, int]]:
    """Random baseline: ``n_pairs`` occurrences with different surface words."""
    rng = random.Random(seed)
    indices = df.index.tolist()
    pairs: List[Tuple[int, int]] = []
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 20:
        attempts += 1
        a, b = rng.sample(indices, 2)
        if df.loc[a].text != df.loc[b].text:
            pairs.append((a, b))
    return pairs


__all__ = [
    "build_librispeech_dataframe", "build_wordmaps", "filter_df", "phonetic_dist",
    "sample_pairs_from_map", "sample_random_pairs", "samplers",
]
