"""Per-layer CCA / RSA similarity between codec hidden states and VTD.

Refactored from ``vocal_tract_distance/scripts/03_vtd_statistical_analysis.py``.
The main entry point is :func:`compute_codec_vs_vtd_similarity`.
"""

import glob
import json
import os
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
import tqdm

from ..codecs.base import BaseCodec
from ..vtd.postprocess import vtd_interpolation_frame, vtd_normalize, vtd_trim_interpolation
from .pwcca import solve_cca
from .rsa import rsa_similarity


def _speech_path_for(speech_dir: str, vtd_file: str) -> str:
    sub_id = os.path.basename(vtd_file).split("_")[0]
    trackname = os.path.basename(vtd_file).replace("_vtd.npy", "")
    return os.path.join(speech_dir, sub_id, "2drt", "audio", f"{trackname}_audio.wav")


def _flush_bucket(
    vtd_bucket: List[np.ndarray],
    hidden_bucket: List[np.ndarray],
    cca_acc: dict,
    rsa_acc: dict,
    use_cca: bool,
    use_rsa: bool,
):
    vtd_array = np.concatenate(vtd_bucket, axis=0)            # (T_total, grid)
    hidden_array = np.concatenate(hidden_bucket, axis=1)       # (L, T_total, D)
    for i in range(hidden_array.shape[0]):
        if use_cca:
            cca = solve_cca(vtd_array.T, hidden_array[i].T)
            cca_acc[i].append(cca["pwcca_sim_x"])
        if use_rsa:
            rsa_acc[i].append(rsa_similarity(vtd_array, hidden_array[i]))


def compute_codec_vs_vtd_similarity(
    codec: BaseCodec,
    vtd_dir: str,
    speech_dir: str,
    cache_path: Optional[str] = None,
    use_cca: bool = True,
    use_rsa: bool = True,
    skip_track_substrings: Tuple[str, ...] = ("vcv", "bvt"),
) -> dict:
    """Compute per-layer PWCCA / RSA between ``codec`` hidden states and VTD.

    The output dict mirrors the JSON cache format used by the original scripts,
    so cached results from prior runs remain readable.
    """
    if not (use_cca or use_rsa):
        raise ValueError("At least one of use_cca / use_rsa must be True")

    vtd_files = sorted(glob.glob(os.path.join(vtd_dir, "*_vtd.npy")))

    cca_acc: dict = defaultdict(list)
    rsa_acc: dict = defaultdict(list)

    vtd_bucket: List[np.ndarray] = []
    hidden_bucket: List[np.ndarray] = []
    bucket_len = 0

    for vtd_file in tqdm.tqdm(vtd_files):
        trackname = os.path.basename(vtd_file).replace("_vtd.npy", "").lower()
        if any(s in trackname for s in skip_track_substrings):
            continue

        speech_file = _speech_path_for(speech_dir, vtd_file)
        if not os.path.exists(speech_file):
            continue

        hidden, T, D = codec.extract_hidden_states(speech_file)

        vtd = np.load(vtd_file)
        vtd = vtd_trim_interpolation(vtd)
        vtd = vtd_normalize(vtd)
        vtd = vtd_interpolation_frame(vtd, T)  # (T, grid_dim)

        if bucket_len < D:
            vtd_bucket.append(vtd)
            hidden_bucket.append(hidden)
            bucket_len += T
        else:
            _flush_bucket(vtd_bucket, hidden_bucket, cca_acc, rsa_acc, use_cca, use_rsa)
            vtd_bucket, hidden_bucket, bucket_len = [], [], 0

    if bucket_len > 0 and vtd_bucket:
        _flush_bucket(vtd_bucket, hidden_bucket, cca_acc, rsa_acc, use_cca, use_rsa)

    cca_means = [np.mean(cca_acc[i]) for i in sorted(cca_acc)] if use_cca else None
    rsa_means = [np.mean(rsa_acc[i]) for i in sorted(rsa_acc)] if use_rsa else None
    num_layers = len(cca_means) if cca_means is not None else (len(rsa_means) if rsa_means else 0)

    payload = {
        "codec_model": codec.name,
        "use_cca": use_cca,
        "use_rsa": use_rsa,
        "num_layers": int(num_layers),
        "cca_means": cca_means,
        "rsa_means": rsa_means,
        "cca_raw": {str(k): v for k, v in cca_acc.items()} if use_cca else None,
        "rsa_raw": {str(k): v for k, v in rsa_acc.items()} if use_rsa else None,
    }

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    return payload


def plot_similarity(
    cca_means: Optional[List[float]],
    rsa_means: Optional[List[float]],
    num_layers: int,
    codec_model: str,
    output_path: Optional[str] = None,
    ax=None,
):
    """Line plot of per-layer correlation, matching the paper's figure style."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.8, 3.2))
    else:
        fig = ax.figure

    if cca_means is not None:
        ax.plot(cca_means, marker="o", label="PWCCA", linewidth=2.5, markersize=4)
    if rsa_means is not None:
        ax.plot(rsa_means, marker="s", label="RSA", linewidth=2.5, markersize=4)

    ax.set_xlabel("Code Index", fontsize=20)
    ax.set_ylabel("Correlation", fontsize=20)
    ax.set_xticks(list(range(0, num_layers, 5)))
    ax.tick_params(labelsize=16)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=14, frameon=False)
    ax.set_title(codec_model)
    fig.tight_layout(pad=0.2)

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    return fig, ax
