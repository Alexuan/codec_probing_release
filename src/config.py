"""Per-experiment configuration dataclasses.

These replace the Hydra YAML configs used during development; each notebook
imports the relevant config at the top so every path / hyperparameter lives in
one visible place. Machine-specific locations (the SPAN rtMRI speech corpus and
the MiMo-Audio-Tokenizer checkpoint) are read from environment variables so the
notebooks contain no hard-coded absolute paths:

    export SPAN_SPEECH_DIR=/path/to/SPAN          # parent of <sub_id>/2drt/audio/...
    export MIMO_CHECKPOINT=/path/to/MiMo-Audio-Tokenizer

Defaults point under ``./data`` where the shipped caches live, so the
figure-reproduction notebook runs with no configuration.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Placeholders shown when a machine-specific path has not been set.
_MIMO_DEFAULT = os.environ.get("MIMO_CHECKPOINT", "<path-to-MiMo-Audio-Tokenizer>")
_SPAN_DEFAULT = os.environ.get("SPAN_SPEECH_DIR", "<path-to-SPAN-speech-corpus>")


@dataclass
class SemanticPhoneticConfig:
    """Exp 1 (Sec 2.3): word-pair Euclidean distance probing -> Fig 2."""
    word_pairs_dir: str = "./data/word_pairs"
    cache_dir: str = "./data/euclidean_cache"
    pool: str = "mean"
    n_pairs: int = 2000          # raise to 5-10k for the final figure
    seed: int = 0


@dataclass
class ArticulatoryConfig:
    """Exp 2 (Sec 2.4): codec vs VTD PWCCA -> Fig 3 & 4."""
    vtd_dir: str = "./data/output_vtd"
    speech_dir: str = field(default_factory=lambda: _SPAN_DEFAULT)
    cache_dir: str = "./data/output_stats_cache"
    use_cca: bool = True
    use_rsa: bool = False        # paper reports PWCCA only; RSA kept available in code
    skip_track_substrings: tuple = ("vcv", "bvt")


@dataclass
class SpeechTextCKAConfig:
    """Exp 3 (Sec 2.5): CKA between codec final-layer features and LLM text embeddings."""
    word_pairs_dir: str = "./data/word_pairs"
    cache_path: str = "./data/cka_results.pkl"
    # LLM paired with each codec (the checkpoints used to produce the paper's CKA table).
    text_models: Dict[str, str] = field(default_factory=lambda: {
        "mimi": "kyutai/moshiko-pytorch-bf16",
        "mimo": "XiaomiMiMo/MiMo-Audio-7B-Base",
    })
    n_words: int = 500
    n_permutations: int = 100
    seed: int = 0


@dataclass
class ReleaseConfig:
    """Top-level container so notebooks can import one config object."""
    device: str = "cuda"
    codecs: List[str] = field(default_factory=lambda: ["encodec", "dac", "mimi", "mimo"])
    mimo_checkpoint: Optional[str] = field(default_factory=lambda: _MIMO_DEFAULT)
    semantic_phonetic: SemanticPhoneticConfig = field(default_factory=SemanticPhoneticConfig)
    articulatory: ArticulatoryConfig = field(default_factory=ArticulatoryConfig)
    speech_text_cka: SpeechTextCKAConfig = field(default_factory=SpeechTextCKAConfig)
