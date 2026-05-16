from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class BaseCodec(nn.Module, ABC):
    """Common interface for the four codecs probed in the paper.

    All codecs expose one method, :meth:`extract_hidden_states`, that returns
    the *accumulated decoded features* per the paper's Sec 2.3 definition:
    ``feats[k]`` is the sum of the decoded outputs of quantizer layers
    ``0..k``.
    """

    def __init__(self, name: str, sampling_rate: int):
        super().__init__()
        self.name = name
        self.sampling_rate = sampling_rate

    @abstractmethod
    def extract_hidden_states(self, audio_path: str) -> Tuple[np.ndarray, int, int]:
        """Return ``(hidden_states[L, T, D], T, D)`` for a single audio file."""
