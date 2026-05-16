"""Per-experiment configuration dataclasses.

These replace the Hydra YAML configs used in the source repos. Each notebook
instantiates the relevant config at the top so all hyperparameters live in one
visible place.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SemanticPhoneticConfig:
    """Exp 1 (Sec 2.3): word-pair Euclidean distance probing."""
    librispeech_dir: str = "./data/librispeech"
    textgrid_dir: str = "./data/librispeech_alignments"
    word_segments_dir: str = "./data/word_segments"
    homophone_threshold: float = 0.4
    pairs_per_word: int = 1
    random_pair_count: int = 10000
    seed: int = 0


@dataclass
class ArticulatoryConfig:
    """Exp 2 (Sec 2.4): codec vs VTD PWCCA."""
    vtd_dir: str = "./data/output_vtd"
    speech_dir: str = "./data/75speaker_speech"
    cache_dir: str = "./data/output_stats_cache"
    use_cca: bool = True
    use_rsa: bool = False  # paper uses PWCCA only; RSA is kept available in code
    skip_track_substrings: tuple = ("vcv", "bvt")


@dataclass
class SpeechTextCKAConfig:
    """Exp 3 (Sec 2.5): CKA between codec latents and text-token latents."""
    librispeech_dir: str = "./data/librispeech"
    text_model: str = "kyutai/moshi-7b"  # MIMI LLM
    word_segments_dir: str = "./data/word_segments"
    n_permutations: int = 100
    seed: int = 0


@dataclass
class ReleaseConfig:
    """Top-level container so notebooks can import one config object."""
    device: str = "cuda"
    codecs: List[str] = field(default_factory=lambda: ["encodec", "dac", "mimi", "mimo"])
    mimo_checkpoint: Optional[str] = None
    semantic_phonetic: SemanticPhoneticConfig = field(default_factory=SemanticPhoneticConfig)
    articulatory: ArticulatoryConfig = field(default_factory=ArticulatoryConfig)
    speech_text_cka: SpeechTextCKAConfig = field(default_factory=SpeechTextCKAConfig)
