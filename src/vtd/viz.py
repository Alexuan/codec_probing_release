"""Visualization helpers for vocal tract boundaries on rtMRI frames."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections as mc


def plot_vocal_tract_boundaries(
    img: np.ndarray,
    grid,
    lower_boundary: np.ndarray,
    upper_boundary: np.ndarray,
    output_path: Optional[str] = None,
):
    """Overlay grid lines, boundary points, and connecting segments on an rtMRI frame."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)

    lower_grid = grid[0][0][4]
    upper_grid = grid[0][0][5]
    for i in range(len(lower_grid)):
        ax.add_collection(mc.LineCollection([[lower_grid[i], upper_grid[i]]], colors="y", linewidths=0.5))

    for i in range(lower_grid.shape[0]):
        lb, ub = lower_boundary[i], upper_boundary[i]
        if np.any(np.isnan(lb)) or np.any(np.isnan(ub)):
            continue
        ax.scatter(lb[0], lb[1], s=50, c="cyan", marker="x", zorder=3)
        ax.scatter(ub[0], ub[1], s=50, c="magenta", marker="x", zorder=3)
        label = "Vocal Tract Boundary" if i == 0 else ""
        ax.add_collection(
            mc.LineCollection([[lb, ub]], colors="lime", linewidths=2, zorder=2, label=label)
        )
    ax.legend()
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig, ax
