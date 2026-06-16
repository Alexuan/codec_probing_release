"""Batch-compute Vocal Tract Distance (VTD) arrays from MATLAB-tracked contours.

Mirrors ``vocal_tract_distance/scripts/02_cal_vtd_all_75.py`` but with cleaner
argument handling and no production-debug ``ipdb`` breakpoints.

Inputs follow the 75-Speaker corpus layout::

    contour_dir/<sub_id>/track/<trackname>_track.mat
    grid_dir/<sub_id>.mat

Outputs::

    output_dir/<trackname>_vtd.npy           # (n_frames, 120) per track

Example::

    python scripts/compute_vtd_batch.py \
        --contour-dir /path/to/SPAN/span_75speakers_annot16 \
        --grid-dir    /path/to/SPAN/grids \
        --output-dir  data/output_vtd
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--contour-dir", type=Path, required=True)
    p.add_argument("--grid-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute even when the output .npy exists.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from src.vtd import calculate_vocal_tract_distance

    contour_files = sorted(glob.glob(str(args.contour_dir / "*/track/*.mat")))
    if not contour_files:
        print(f"no *.mat under {args.contour_dir}/*/track/", file=sys.stderr)
        return 1

    for contour_path in tqdm.tqdm(contour_files, desc="VTD"):
        trackname = os.path.basename(contour_path).replace("_track.mat", "")
        sub_id = trackname.split("_")[0]
        grid_path = args.grid_dir / f"{sub_id}.mat"
        if not grid_path.exists():
            print(f"missing grid for {sub_id}, skipping {trackname}", file=sys.stderr)
            continue

        out = args.output_dir / f"{trackname}_vtd.npy"
        if out.exists() and not args.overwrite:
            continue

        vtd, _lower, _upper = calculate_vocal_tract_distance(
            contour_predict_file=contour_path,
            grid_file=str(grid_path),
        )
        np.save(out, vtd)

    print(f"done -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
