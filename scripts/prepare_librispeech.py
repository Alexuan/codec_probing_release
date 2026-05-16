"""Build the LibriSpeech word-occurrence DataFrame and synonym / homophone maps.

The underlying functions are a verbatim port (Apache 2.0) of
https://github.com/juice500ml/phonetic_semantic_probing — see
``src/data/_juice500ml.py``. This wrapper just exposes a single CLI entry
point and writes the two pickles consumed by the probing notebooks.

Outputs (under ``--output-dir``):
  - ``librispeech.df.pkl``    — one row per MFA-aligned word occurrence
  - ``librispeech.wordmap.pkl`` — dict with ``synonym_map`` / ``homophone_map``

Example::

    python scripts/prepare_librispeech.py \
        --librispeech-dir /data/xuanshi/DATA/LibriSpeech \
        --textgrid-dir    /data/xuanshi/DATA/LibriSpeech_MFA \
        --output-dir      data/word_pairs \
        --splits dev-clean test-clean \
        --threshold 0.4
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--librispeech-dir", type=Path, required=True,
                   help="Root of LibriSpeech wavs (contains dev-clean/, test-clean/, ...).")
    p.add_argument("--textgrid-dir", type=Path, required=True,
                   help="Root of MFA TextGrids (same layout as LibriSpeech).")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write .df.pkl and .wordmap.pkl.")
    p.add_argument("--splits", nargs="+", default=["dev-clean", "test-clean"],
                   help="LibriSpeech splits to include.")
    p.add_argument("--threshold", type=float, default=0.4,
                   help="Phonetic-distance threshold for synonym / homophone classification.")
    p.add_argument("--num-workers", type=int, default=64,
                   help="Parallel workers for the homophone search.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-df", action="store_true",
                   help="Reuse the existing .df.pkl and only rebuild the wordmap.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df_path = args.output_dir / "librispeech.df.pkl"
    map_path = args.output_dir / "librispeech.wordmap.pkl"

    from src.data import build_librispeech_dataframe, build_wordmaps

    if args.skip_df and df_path.exists():
        import pandas as pd
        df = pd.read_pickle(df_path)
        print(f"reusing existing DataFrame: {df_path} ({len(df)} rows)")
    else:
        df = build_librispeech_dataframe(
            dataset_path=args.librispeech_dir,
            textgrid_path=args.textgrid_dir,
            splits=tuple(args.splits),
        )
        df.to_pickle(df_path)
        print(f"wrote {df_path}  ({len(df)} word occurrences, "
              f"{df.text.nunique()} unique words)")

    wordmaps = build_wordmaps(
        df,
        threshold=args.threshold,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    with open(map_path, "wb") as f:
        pickle.dump(wordmaps, f)
    print(f"wrote {map_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
