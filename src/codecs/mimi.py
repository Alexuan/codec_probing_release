"""MIMI codec, loaded from HuggingFace ``kyutai/mimi``.

MIMI has a semantic-acoustic split: the first quantizer is distilled from
WavLM, the remaining are acoustic RVQ. We return ``[L, T, D]`` with::

    feats[0]                    = decoded semantic (WavLM-distilled)
    feats[1..L-1]               = decoded semantic + cumsum(acoustic_1..k)

Set ``add_semantic_to_acoustic=False`` to expose the bare semantic vs bare
accumulated-acoustic split used in Fig 4 of the paper.
"""

from typing import Tuple

import numpy as np
import torch
from transformers import AutoFeatureExtractor, MimiModel

from ._io import load_mono_resampled
from .base import BaseCodec


class MIMI(BaseCodec):
    def __init__(
        self,
        model_name: str = "kyutai/mimi",
        device: str = "cuda",
        add_semantic_to_acoustic: bool = True,
    ):
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        super().__init__(name="mimi", sampling_rate=feature_extractor.sampling_rate)
        self.device = device
        self.add_semantic_to_acoustic = add_semantic_to_acoustic
        self.feature_extractor = feature_extractor
        self.model = MimiModel.from_pretrained(model_name).to(device).eval()

    @torch.no_grad()
    def extract_hidden_states(self, audio_path: str) -> Tuple[np.ndarray, int, int]:
        audio_np = load_mono_resampled(audio_path, self.sampling_rate)
        inputs = self.feature_extractor(
            raw_audio=audio_np,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        encoder_outputs = self.model.encode(**inputs)
        codes = encoder_outputs[0]  # [1, N_total, T]

        n_semantic = self.model.quantizer.num_semantic_quantizers
        chunks = []

        if n_semantic > 0:
            sem = self.model.quantizer.semantic_residual_vector_quantizer.decode(
                codes[:, :n_semantic]
            )
            chunks.append(sem)  # [1, D, T]

        if codes.shape[1] > n_semantic:
            acoustic = codes[:, n_semantic:].transpose(0, 1)  # [N_ac, 1, T]
            running = torch.tensor(0.0, device=acoustic.device)
            acoustic_list = []
            for i, indices in enumerate(acoustic):
                running = running + self.model.quantizer.acoustic_residual_vector_quantizer.layers[i].decode(indices)
                acoustic_list.append(running)
            acoustic_stack = torch.cat(acoustic_list, dim=0)  # [N_ac, D, T]
            acoustic_stack = self.model.quantizer.acoustic_residual_vector_quantizer.output_proj(
                acoustic_stack
            )
            chunks.append(acoustic_stack)

        embeddings = torch.cat(chunks, dim=0)  # [N_total, D, T]

        if self.add_semantic_to_acoustic and n_semantic > 0 and embeddings.shape[0] > 1:
            base = embeddings[0:1]
            embeddings = embeddings.clone()
            embeddings[1:] = embeddings[1:] + base

        upsampled = self.model.upsample(embeddings)  # [N, D, T_up]
        feats = upsampled.transpose(1, 2).contiguous().cpu().numpy()  # [N, T, D]
        return feats, feats.shape[1], feats.shape[2]
