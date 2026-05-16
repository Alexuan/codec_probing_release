"""MIMO codec (MiMo-Audio-Tokenizer, Xiaomi).

Requires the vendored ``mimo_audio_tokenizer`` package and its checkpoint
directory. The package itself needs ``flash_attn`` at runtime. Install via::

    pip install -e /path/to/MiMo_Audio_Tokenizer
    pip install flash-attn --no-build-isolation
"""

from typing import Tuple

import numpy as np
import torch

from .base import BaseCodec


class MIMO(BaseCodec):
    def __init__(self, checkpoint: str, device: str = "cuda"):
        try:
            import mimo_audio_tokenizer  # type: ignore
        except ImportError as e:
            raise ImportError(
                "MIMO requires the vendored `mimo_audio_tokenizer` package on "
                "sys.path and a working `flash_attn` install. See the project "
                f"README for setup instructions (underlying error: {e})."
            ) from e

        model = mimo_audio_tokenizer.load_model(checkpoint).bfloat16().to(device).eval()
        super().__init__(name="mimo", sampling_rate=int(model.config.sampling_rate))
        self.device = device
        self.model = model
        self._mimo = mimo_audio_tokenizer

    @torch.no_grad()
    def extract_hidden_states(self, audio_path: str) -> Tuple[np.ndarray, int, int]:
        wav = self._mimo.load_audio(audio_path, self.sampling_rate)
        mels = self._mimo.mel_spectrogram(wav, self.model.config)
        mels, mels_lens = self._mimo.padding([mels])
        mels = mels.to(self.device)
        mels_lens = mels_lens.to(self.device)

        codes, codes_lens, _ = self.model.encode(mels, mels_lens)
        # decode_vq_layer returns the final embedding plus a per-layer list of
        # accumulated decoded features (each is [1, T, D]).
        hidden_states, hidden_states_list = self.model.encoder.decode_vq_layer(codes, codes_lens)
        feats = np.concatenate([h.cpu().float().numpy() for h in hidden_states_list], axis=0)  # [L, T, D]
        return feats, hidden_states.shape[1], hidden_states.shape[2]
