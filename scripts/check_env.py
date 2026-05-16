"""Quickly verify the environment can load the four codecs and the analysis utilities.

Run::

    python scripts/check_env.py
    python scripts/check_env.py --mimo-checkpoint /path/to/MiMo-Audio-Tokenizer

A clean run prints one ``OK`` line per codec that loads and one ``SKIP`` line
per codec that the local environment is missing (e.g. ``descript-audio-codec``
not installed, or ``flash_attn`` missing for MIMO). Use this as the first
sanity check before opening any of the notebooks.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import warnings

# Make `src` importable when running from the repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mimo-checkpoint", default=None,
                   help="Path to MiMo-Audio-Tokenizer checkpoint directory.")
    p.add_argument("--device", default="auto", help="cuda / cpu / auto")
    return p.parse_args()


def main() -> int:
    warnings.filterwarnings("ignore")
    args = _parse_args()

    import numpy as np
    import soundfile as sf
    import torch

    device = (
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        ("cuda" if args.device == "cuda" else "cpu")
    )

    print(f"python       {sys.version.split()[0]}")
    print(f"torch        {torch.__version__}  cuda={torch.cuda.is_available()}")
    print(f"device       {device}")

    # Module imports
    from src.codecs import load_dac, load_encodec, load_mimi, load_mimo  # noqa: F401
    from src.analysis import (  # noqa: F401
        cka_with_baseline,
        compute_codec_vs_vtd_similarity,
        extract_codec_features,
        linear_cka,
        plot_similarity,
        rsa_similarity,
        solve_cca,
    )
    from src.data import (  # noqa: F401
        build_librispeech_dataframe,
        build_wordmaps,
        phonetic_dist,
    )
    from src.vtd import calculate_vocal_tract_distance  # noqa: F401
    print("imports      OK")

    # 2-sec dummy audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, 0.01 * np.random.randn(48000).astype("float32"), 24000)
        audio_path = tmp.name

    loaders = [
        ("encodec", lambda: load_encodec(bandwidth=12.0, device=device)),
        ("dac",     lambda: load_dac(device=device)),
        ("mimi",    lambda: load_mimi(device=device)),
    ]
    if args.mimo_checkpoint:
        loaders.append(("mimo", lambda: load_mimo(checkpoint=args.mimo_checkpoint, device=device)))

    failed = 0
    for name, fn in loaders:
        try:
            codec = fn()
            feats, T, D = codec.extract_hidden_states(audio_path)
            print(f"OK   {name:8s} sr={codec.sampling_rate}  feats={feats.shape}  (L,T,D)")
        except Exception as e:
            print(f"SKIP {name:8s} {type(e).__name__}: {str(e)[:160]}")
            failed += 1

    os.unlink(audio_path)
    return 0 if failed < len(loaders) else 1


if __name__ == "__main__":
    sys.exit(main())
