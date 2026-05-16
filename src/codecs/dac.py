"""DAC codec (Descript Audio Codec) at 24 kHz.

Requires ``pip install descript-audio-codec`` plus ``pip install audiotools``.
Per-layer accumulated features follow Sec 2.3 of the paper: ``feats[k]`` is the
cumulative sum of decoded residual codebooks ``0..k``.
"""

from typing import Tuple

import numpy as np
import torch

from .base import BaseCodec


class DAC(BaseCodec):
    def __init__(self, model_type: str = "24khz", device: str = "cuda"):
        try:
            import dac  # type: ignore
        except ImportError as e:
            raise ImportError(
                "DAC requires `descript-audio-codec`. Install with "
                "`pip install descript-audio-codec audiotools`."
            ) from e
        sampling_rate = {"24khz": 24000, "44khz": 44100, "16khz": 16000}[model_type]
        super().__init__(name=f"dac_{model_type}", sampling_rate=sampling_rate)
        self.device = device

        model_path = dac.utils.download(model_type=model_type)
        self.model = dac.DAC.load(model_path).to(device).eval()

    @torch.no_grad()
    def extract_hidden_states(self, audio_path: str) -> Tuple[np.ndarray, int, int]:
        from audiotools import AudioSignal  # type: ignore

        signal = AudioSignal(audio_path)
        signal.resample(self.sampling_rate)
        signal.to(self.device)

        compressed = self.model.compress(signal)
        codes = compressed.codes  # [B, n_q, T]

        _, z_p, _ = self.model.quantizer.from_codes(codes)
        # z_p: [B, n_q * D_proj, T] -> [n_q, D_proj, T] after squeezing batch
        z_p = z_p.squeeze(0)
        n_codebooks = self.model.quantizer.n_codebooks
        if z_p.shape[0] % n_codebooks != 0:
            raise ValueError(f"DAC z_p shape {z_p.shape} not divisible by n_codebooks={n_codebooks}")
        d_proj = z_p.shape[0] // n_codebooks
        z_p = z_p.view(n_codebooks, d_proj, z_p.shape[1])  # [n_q, D_proj, T]

        accumulated = torch.cumsum(z_p, dim=0)  # [n_q, D_proj, T]
        feats = accumulated.transpose(1, 2).contiguous().cpu().numpy()  # [n_q, T, D_proj]
        return feats, feats.shape[1], feats.shape[2]
