# Vendored from https://github.com/juice500ml/phonetic_semantic_probing
# (Apache License 2.0). Only the LibriSpeech-specific code paths are kept;
# the multilingual / commonvoice / FSC / SNIPS parsers in the upstream
# `dataset_cleanup.py` are not needed for this release.
#
# Original authors: Juice Choi (juice500ml) and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Verbatim port of juice500ml's LibriSpeech word-pair construction.

The two public entry points are:

* :func:`build_librispeech_dataframe` — matches ``_librispeech`` in
  ``dataset_cleanup.py``. Parses MFA TextGrids into a per-word DataFrame
  with columns ``text / start / finish / path / phones / synonyms / speaker``.

* :func:`build_wordmaps` — matches the ``__main__`` block of
  ``extract_synonyms_homophones.py``. Returns
  ``{'synonym_map': {word: set(...)}, 'homophone_map': {word: set(...)}}``.

The helper functions (``phonetic_dist``, ``filter_df``, ``_get_synonym_map``,
``_get_homophone``, ``_get_homophone_map``) are kept under their original
names to make it easy to diff against the upstream repo.
"""

from __future__ import annotations

import re
from functools import partial
from pathlib import Path
from typing import Sequence

import pandas as pd
from jiwer import wer
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


# ---------------------------------------------------------------- utils.py


def phonetic_dist(x, y):  # noqa: D401 — match upstream signature exactly
    """Phoneme-level WER. Direct port of ``utils.phonetic_dist``."""
    ref, hyp = (x, y) if len(x) > len(y) else (y, x)
    return wer(reference=" ".join(ref), hypothesis=" ".join(hyp))


def filter_df(df, speaker, n_sample, language_uniform, seed):
    """Direct port of ``utils.filter_df``."""
    print(f"Original size: {len(df)}")
    if language_uniform:
        langs = df["language"].unique()
        assert speaker is None
        df = pd.concat([
            df[df.language == lang].sample(n_sample, random_state=seed)
            for lang in langs
        ])
    else:
        if speaker is not None:
            df = df[df.speaker == speaker]
        if n_sample is not None:
            df = df.sample(n_sample, random_state=seed)
    print(f"Filtered size: {len(df)}")
    return df


# ---------------------------------------------------------------- dataset_cleanup.py


def _cmudict(word, cache=None):
    """Direct port. The default ``cache`` is computed lazily so importing this
    module does not pay the cmudict load cost."""
    if cache is None:
        import cmudict
        cache = cmudict.dict()
    if word not in cache:
        return None
    return [re.sub(r"\d+", "", p) for p in cache[word][0]]


def _wordnet(word, lang="eng"):
    """Direct port. Lazily loads the WordNet corpus."""
    import nltk
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    from nltk.corpus import wordnet

    synonyms = [
        syn
        for synsets in wordnet.synsets(word, lang=lang)
        for syn in synsets.lemma_names("eng")
        if syn != word
    ]
    return set(synonyms)


def build_librispeech_dataframe(
    dataset_path: Path, textgrid_path: Path,
    splits: Sequence[str] = ("dev-clean", "test-clean"),
) -> pd.DataFrame:
    """Direct port of ``_librispeech`` in ``dataset_cleanup.py``.

    Only addition: ``splits`` is exposed as a keyword instead of hard-coded.
    """
    import cmudict
    from textgrids import TextGrid

    dataset_path = Path(dataset_path)
    textgrid_path = Path(textgrid_path)
    cmu_cache = cmudict.dict()

    rows = []
    for split in splits:
        for p in tqdm(list(textgrid_path.glob(f"{split}/*/*/*.TextGrid"))):
            grid = TextGrid(p)
            for word in grid["words"]:
                phones = _cmudict(word.text, cache=cmu_cache)
                synonyms = list(_wordnet(word.text))
                if phones is not None and len(synonyms) > 0:
                    rows.append({
                        "text": word.text,
                        "start": word.xmin,
                        "finish": word.xmax,
                        "path": str(
                            (dataset_path / p.relative_to(p.parents[3]).with_suffix(".flac")).absolute()
                        ),
                        "phones": phones,
                        "synonyms": synonyms,
                        "speaker": p.parents[1].name,
                    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------- extract_synonyms_homophones.py


def _get_synonym_map(words, df, text2phones, threshold=0.4):
    """Direct port of ``_get_synonym_map``."""
    synonym_map = {}
    for index in tqdm(df.reset_index().groupby(["text"])["index"].min()):
        row = df.loc[index]
        synonyms = []
        for s in row.synonyms:
            if threshold < 0:
                synonyms.append(s)
            else:
                if (s in words) and (phonetic_dist(row.phones, text2phones[s]) > threshold):
                    synonyms.append(s)
        synonyms = set(synonyms).intersection(words)
        if len(synonyms) > 0:
            synonym_map[row.text] = synonyms
    return synonym_map


def _get_homophone(df, indices, text2phones, synonym_map, word, threshold=0.4):
    """Direct port of ``_get_homophone``."""
    homophones = []
    for index in indices:
        row = df.loc[index]
        if (row.text != word) and \
            (row.text not in synonym_map.get(word, set())) and \
            (word not in synonym_map.get(row.text, set())):
            if 0.0 < phonetic_dist(text2phones[word], row.phones) <= threshold:
                homophones.append(row.text)
    return set(homophones)


def _get_homophone_map(words, synonym_map, df, text2phones, num_workers):
    """Direct port of ``_get_homophone_map``."""
    words = list(words)
    chunksize = max(1, len(words) // max(num_workers * 4, 1))
    tqdm_args = dict(max_workers=num_workers, chunksize=chunksize)
    indices = df.reset_index().groupby(["text"])["index"].min()

    homophones_list = process_map(
        partial(_get_homophone, df[["text", "phones"]], indices, text2phones, synonym_map),
        words, **tqdm_args)
    return {w: hs for w, hs in zip(words, homophones_list) if len(hs) > 0}


def build_wordmaps(
    df: pd.DataFrame,
    threshold: float = 0.4,
    num_workers: int = 64,
    speaker: str | None = None,
    n_sample: int | None = None,
    seed: int = 0,
) -> dict:
    """Run the upstream ``extract_synonyms_homophones.py`` ``__main__`` flow.

    Returns ``{'synonym_map': ..., 'homophone_map': ...}``. The flow follows
    the upstream English / non-language_uniform branch exactly: a filtered
    synonym map at ``threshold``, an unfiltered synonym map used solely to
    exclude pairs from the homophone search, then the homophone map.
    """
    filtered_df = filter_df(df, speaker, n_sample, language_uniform=False, seed=seed)
    words = set(filtered_df.text.unique())
    text2phones = {row.text: tuple(row.phones) for row in filtered_df.itertuples()}

    synonym_map = _get_synonym_map(words, filtered_df, text2phones, threshold=threshold)
    print(f"Synonym pairs: {sum(len(v) for v in synonym_map.values())}")

    not_filtered_synonym_map = _get_synonym_map(words, filtered_df, text2phones, threshold=-1)
    homophone_map = _get_homophone_map(
        words, not_filtered_synonym_map, filtered_df, text2phones, num_workers,
    )
    print(f"Near-homophone pairs: {sum(len(v) for v in homophone_map.values())}")

    return {"synonym_map": synonym_map, "homophone_map": homophone_map}


# ---------------------------------------------------------------- samplers (utils.samplers)


def _random_sampler(df, wordmap):
    import numpy as np
    l_indices = df.index.to_numpy()
    r_indices = l_indices.copy()
    np.random.default_rng(seed=42).shuffle(r_indices)
    for l, r in zip(l_indices, r_indices):
        if l != r:
            yield l, r


def _synonym_sampler(df, wordmap):
    l_indices = df.index.to_numpy()
    for l in l_indices:
        syn = set(wordmap["synonym_map"].get(df.loc[l].text, set()))
        for r in df[df.text.isin(syn)].index:
            if l != r:
                yield l, r


def _homophone_sampler(df, wordmap):
    l_indices = df.index.to_numpy()
    for l in l_indices:
        hom = wordmap["homophone_map"].get(df.loc[l].text, set())
        for r in df[df.text.isin(hom)].index:
            if l != r:
                yield l, r


samplers = {
    "random": _random_sampler,
    "synonym": _synonym_sampler,
    "homophone": _homophone_sampler,
}
