"""Post-processing of raw per-frame VTD arrays before CCA/RSA analysis."""

import numpy as np

DEFAULT_GRID_DIM = 120


def vtd_trim_interpolation(vtd_data: np.ndarray, grid_dim: int = DEFAULT_GRID_DIM) -> np.ndarray:
    """Drop all-NaN grid columns, fill remaining NaNs with 0, and interpolate
    back to ``grid_dim`` columns. Input shape: ``(n_frames, grid_dim)``."""
    frame_num, _ = vtd_data.shape

    all_nan = np.all(np.isnan(vtd_data), axis=0)
    keep = ~all_nan
    kept_idx = np.where(keep)[0]

    trimmed = vtd_data[:, keep].astype(float)
    trimmed[np.isnan(trimmed)] = 0.0

    if kept_idx.size == grid_dim:
        return trimmed

    orig_idx = np.arange(grid_dim)
    out = np.zeros((frame_num, grid_dim), dtype=float)
    for i in range(frame_num):
        out[i] = np.interp(orig_idx, kept_idx, trimmed[i])
    return out


def vtd_interpolation_frame(vtd_data: np.ndarray, target_frame_length: int) -> np.ndarray:
    """Resample the time axis of VTD to ``target_frame_length`` frames."""
    frame_length, grid_dim = vtd_data.shape
    orig_idx = np.arange(frame_length)
    target_idx = np.linspace(0, frame_length - 1, target_frame_length)

    out = np.zeros((target_frame_length, grid_dim), dtype=float)
    for i in range(grid_dim):
        out[:, i] = np.interp(target_idx, orig_idx, vtd_data[:, i])
    return out


def vtd_normalize(vtd_data: np.ndarray) -> np.ndarray:
    """Normalize each gridline independently to its per-clip maximum."""
    max_per_grid = np.nanmax(vtd_data, axis=0)
    max_per_grid[max_per_grid == 0] = 1.0
    return vtd_data / max_per_grid
