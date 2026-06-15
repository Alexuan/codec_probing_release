"""Build the shipped Fig 2 cache (``data/euclidean_cache/{codec}.pkl``).

Release-prep provenance tool. It converts the authors' raw per-seed word-pair
distance outputs (one ``.dist.pkl`` per codec/seed, each a dict of
``[n_layers, n_pairs]`` arrays keyed by pair group) into the compact per-codec
cache the figure notebooks consume: a dict ``{synonym, homophone, random} ->
[n_seeds, n_layers]`` where each row is that seed's mean distance per layer.

This matches the dev aggregation (``generate_layerwise_figures_seeds.py``):
mean over pairs within a seed, then ``notebook 04``'s ``mean(axis=0)`` averages
across the 5 seeds. Readers reproducing from scratch regenerate the same cache
by running ``notebooks/01_semantic_phonetic_probing.ipynb``; this script is kept
to document how the shipped cache was produced.

Usage::

    python scripts/build_fig2_cache.py \
        --dist-dir <dir with *_dist-euclidean_dist.dist.pkl> \
        --output-dir data/euclidean_cache
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
from pathlib import Path

import numpy as np

# Map the codec tag used in the raw filenames -> the short name used for the
# shipped cache file (and shown as the figure's per-panel label).
CODEC_TAGS = {
    "encodec_24k_12bps": "encodec",
    "dac_24k": "dac",
    "mimi": "mimi",
    "mimo": "mimo",
}

GROUPS = ("synonym", "homophone", "random")


def _seed_files(dist_dir: Path, codec_tag: str) -> list:
    pat = str(dist_dir / f"*model-{codec_tag}_*everyone_size-10000_*dist-euclidean_dist.dist.pkl")
    return sorted(glob.glob(pat))


def build_one(dist_dir: Path, codec_tag: str) -> dict | None:
    files = _seed_files(dist_dir, codec_tag)
    if not files:
        print(f"  [skip] no .dist.pkl for {codec_tag}")
        return None

    per_seed: dict = {g: [] for g in GROUPS}  # each entry: per-layer mean over pairs
    n_layers = None
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        for g in GROUPS:
            arr = np.asarray(d[g], dtype=np.float32)  # raw shape: [n_layers, n_pairs]
            if n_layers is None:
                n_layers = arr.shape[0]
            per_seed[g].append(arr.mean(axis=1))      # -> [n_layers]

    out = {g: np.stack(per_seed[g], axis=0) for g in GROUPS}  # [n_seeds, n_layers]
    print(f"  {codec_tag:18s} L={n_layers}  seeds={len(files)}  shape={out['random'].shape}")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dist-dir", type=Path, required=True,
                   help="Directory containing *_dist-euclidean_dist.dist.pkl files.")
    p.add_argument("--output-dir", type=Path, default=Path("data/euclidean_cache"))
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for codec_tag, short in CODEC_TAGS.items():
        out = build_one(args.dist_dir, codec_tag)
        if out is None:
            continue
        with open(args.output_dir / f"{short}.pkl", "wb") as fh:
            pickle.dump(out, fh)
    print(f"done -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
