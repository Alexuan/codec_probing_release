"""Build the LibriSpeech word-occurrence DataFrame and synonym / homophone maps.

Writes the two pickles consumed by the probing notebooks. The DataFrame stores
audio paths **relative to the LibriSpeech root** (e.g. ``dev-clean/<spk>/...``),
so it is portable: at run time the notebooks join it with ``LIBRISPEECH_DIR``
(see ``src.data.resolve_audio_paths``). The df is built from the TextGrid tree
alone; ``--librispeech-dir`` is optional and only used to sanity-check that the
audio actually exists.

Outputs (under ``--output-dir``):
  - ``librispeech.df.pkl``    — one row per MFA-aligned word occurrence
  - ``librispeech.wordmap.pkl`` — dict with ``synonym_map`` / ``homophone_map``

Example::

    python scripts/prepare_librispeech.py \
        --textgrid-dir    /path/to/librispeech_alignments \
        --output-dir      data/word_pairs \
        --splits dev-clean test-clean \
        --librispeech-dir /path/to/LibriSpeech   # optional: verify audio exists
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
    p.add_argument("--textgrid-dir", type=Path, required=True,
                   help="Root of MFA TextGrids, laid out as <split>/<spk>/<chapter>/<utt>.TextGrid.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write .df.pkl and .wordmap.pkl.")
    p.add_argument("--librispeech-dir", type=Path, default=None,
                   help="Optional: LibriSpeech root, used only to verify the audio exists.")
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
            textgrid_path=args.textgrid_dir,
            splits=tuple(args.splits),
        )
        df.to_pickle(df_path)
        print(f"wrote {df_path}  ({len(df)} word occurrences, "
              f"{df.text.nunique()} unique words; paths relative to the LibriSpeech root)")

    # Optional sanity check: do the relative paths resolve under --librispeech-dir?
    if args.librispeech_dir is not None:
        sample = df.path.iloc[0]
        full = os.path.join(str(args.librispeech_dir), sample)
        ok = os.path.exists(full)
        print(f"audio check: {full} -> {'found' if ok else 'MISSING'}")
        if not ok:
            print("  (set LIBRISPEECH_DIR to the root that contains dev-clean/ test-clean/ when running notebooks)")

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
