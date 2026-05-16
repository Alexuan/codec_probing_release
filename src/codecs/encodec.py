"""EnCodec codec, loaded from HuggingFace ``facebook/encodec_24khz``.

The paper uses EnCodec at 12 kbps (16 RVQ codebooks at 75 Hz). Each layer's
output is the cumulative sum of decoded codebooks 0..k, matching the paper's
Sec 2.3 feature definition.
"""

from typing import Tuple

import numpy as np
import torch
from transformers import AutoProcessor, EncodecModel

from ._io import load_mono_resampled
from .base import BaseCodec


class EnCodec(BaseCodec):
    def __init__(
        self,
        model_name: str = "facebook/encodec_24khz",
        bandwidth: float = 12.0,
        device: str = "cuda",
    ):
        super().__init__(name=f"encodec_24khz_{bandwidth}bps", sampling_rate=24000)
        self.device = device
        self.bandwidth = bandwidth
        self.model = EncodecModel.from_pretrained(model_name).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def extract_hidden_states(self, audio_path: str) -> Tuple[np.ndarray, int, int]:
        audio_np = load_mono_resampled(audio_path, self.sampling_rate)
        inputs = self.processor(
            raw_audio=audio_np, sampling_rate=self.sampling_rate, return_tensors="pt"
        )
        input_values = inputs["input_values"].to(self.device)

        encoded = self.model.encode(input_values, bandwidth=self.bandwidth)
        # audio_codes: [num_chunks, B, n_q, T]; we use single-chunk (chunk_length_s=None)
        codes = encoded.audio_codes[0]  # [B, n_q, T]
        n_q = codes.shape[1]

        per_layer = torch.stack(
            [self.model.quantizer.layers[i].decode(codes[:, i, :]) for i in range(n_q)],
            dim=0,
        )  # [n_q, B, D, T]
        accumulated = torch.cumsum(per_layer, dim=0)  # [n_q, B, D, T]
        # Drop batch dim (B=1), move to [n_q, T, D]
        feats = accumulated.squeeze(1).transpose(1, 2).contiguous().cpu().numpy()
        return feats, feats.shape[1], feats.shape[2]
