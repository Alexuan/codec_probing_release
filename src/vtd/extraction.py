"""Vocal Tract Distance (VTD) extraction from MATLAB-tracked articulator contours.

Reads ``trackdata`` .mat files produced by the upstream rtMRI segmentation
pipeline, selects articulator sub-contours, and intersects them with a fixed
grid of cross-sectional lines spanning lips to larynx. The per-frame distance
between the resulting upper and lower boundaries forms the VTD vector.
"""

from typing import List, Tuple

import numpy as np
from scipy import io


_JAW_IDS = [6, 1, 2, 3, 4]
_BACK_IDS = [4, 1, 2]
_NASAL_IDS = [5, 1, 2]


def _select_articulator(seg: dict, selected_ids: list) -> Tuple[np.ndarray, np.ndarray]:
    vs, idxs = [], []
    for sid in selected_ids:
        mask = np.isin(seg["i"], sid)
        vs.append(seg["v"][mask])
        idxs.append(seg["i"][mask])
    return np.vstack(vs), np.hstack(idxs)


def _read_segment(raw_seg) -> dict:
    seg = {
        "mu": np.squeeze(raw_seg["mu"][0, 0][0, 0]),
        "v": np.column_stack((raw_seg["v"][0, 0][:, 0] + 42, -raw_seg["v"][0, 0][:, 1] + 42)),
        "i": raw_seg["i"][0, 0].flatten(),
    }
    return seg


def parse_track_file(matfile: str) -> List[dict]:
    """Parse a MATLAB ``trackdata`` file into a list of per-frame contour dicts."""
    mat = io.loadmat(matfile)
    tracks = []
    for trackdata in mat["trackdata"][0]:
        track = {
            "frameNo": np.squeeze(trackdata["frameNo"][0, 0]),
            "template": np.squeeze(trackdata["template"][0, 0]),
            "segments": {},
        }
        jaw_seg, back_seg, nasal_seg, _contour = trackdata["contours"][0, 0]["segment"][0, 0][0]

        for raw_seg, key, ids in (
            (jaw_seg, "jaw", _JAW_IDS),
            (back_seg, "back", _BACK_IDS),
            (nasal_seg, "nasal", _NASAL_IDS),
        ):
            seg = _read_segment(raw_seg)
            seg["v"], seg["i"] = _select_articulator(seg, ids)
            track["segments"][key] = seg

        tracks.append(track)
    return tracks


def get_line_intersection(p1, p2, p3, p4):
    """Intersect segment ``p1-p2`` with segment ``p3-p4``; return ``(x, y)`` or ``None``."""
    s1_x, s1_y = p2[0] - p1[0], p2[1] - p1[1]
    s2_x, s2_y = p4[0] - p3[0], p4[1] - p3[1]
    denom = -s2_x * s1_y + s1_x * s2_y
    if denom == 0:
        return None

    s = (-s1_y * (p1[0] - p3[0]) + s1_x * (p1[1] - p3[1])) / denom
    t = (s2_x * (p1[1] - p3[1]) - s2_y * (p1[0] - p3[0])) / denom
    if 0 <= s <= 1 and 0 <= t <= 1:
        return (p1[0] + t * s1_x, p1[1] + t * s1_y)
    return None


def find_vocal_tract_boundary(grid, track: dict, position: str = "low") -> np.ndarray:
    """Find the intersection of articulator contours with grid lines.

    ``position='low'`` extracts the jaw-side boundary, ``'up'`` extracts the
    palate/pharyngeal side. Returns an ``(n_gridlines, 2)`` array of points
    (NaN where no intersection exists).
    """
    lower_grid = grid[0][0][4]
    upper_grid = grid[0][0][5]

    if position == "low":
        articulator = track["segments"]["jaw"]["v"]
        target_grid = upper_grid
    elif position == "up":
        nasal = track["segments"]["nasal"]["v"]
        back = track["segments"]["back"]["v"][::-1]
        articulator = np.vstack((nasal, back))
        target_grid = lower_grid
    else:
        raise ValueError(f"position must be 'low' or 'up', got {position!r}")

    intersections = np.full((len(lower_grid), 2), np.nan)
    for j in range(len(lower_grid)):
        p3, p4 = lower_grid[j], upper_grid[j]
        found = []
        for i in range(len(articulator) - 1):
            pt = get_line_intersection(articulator[i], articulator[i + 1], p3, p4)
            if pt is not None:
                found.append(pt)
        if len(found) == 1:
            intersections[j] = found[0]
        elif len(found) > 1:
            target = target_grid[j]
            dists = [np.linalg.norm(np.asarray(p) - np.asarray(target)) for p in found]
            intersections[j] = found[int(np.argmin(dists))]
    return intersections


def calculate_vocal_tract_distance(
    contour_predict_file: str,
    grid_file: str,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """Compute per-frame VTD from a tracked-contour ``.mat`` and a grid ``.mat``.

    Returns:
        vtd_array: ``(n_frames, n_gridlines)`` distance array.
        lower_boundary_list / upper_boundary_list: per-frame intersection arrays.
    """
    tracks = parse_track_file(contour_predict_file)
    grid_data = io.loadmat(grid_file)["grid"]

    vtd_list, lower_list, upper_list = [], [], []
    for track in tracks:
        lower = find_vocal_tract_boundary(grid_data, track, position="low")
        upper = find_vocal_tract_boundary(grid_data, track, position="up")
        vtd_list.append(np.linalg.norm(lower - upper, axis=1))
        lower_list.append(lower)
        upper_list.append(upper)

    return np.array(vtd_list), lower_list, upper_list
