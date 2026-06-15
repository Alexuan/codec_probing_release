"""Fetch the large derived data (``output_vtd/``, ~1.1 GB) that is not stored in git.

The vocal-tract-distance arrays are hosted on Zenodo. Set the record URL via
``--url`` or the ``CODEC_PROBING_VTD_URL`` env var (the DOI is listed in
``data/MANIFEST.md``), then::

    python scripts/download_data.py --output-dir data/output_vtd

The archive is verified against the sha256 in ``data/MANIFEST.md`` when present.
Needs the SPAN rtMRI *audio* corpus separately (license-gated; see README).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

DEFAULT_URL = os.environ.get("CODEC_PROBING_VTD_URL", "")  # set to the Zenodo file URL


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=DEFAULT_URL, help="Zenodo URL of output_vtd.tar.gz")
    p.add_argument("--output-dir", type=Path, default=Path("data/output_vtd"))
    p.add_argument("--sha256", default=None, help="Expected archive checksum (see data/MANIFEST.md).")
    args = p.parse_args()

    if not args.url:
        print("No URL. Pass --url or set CODEC_PROBING_VTD_URL "
              "(the Zenodo DOI is in data/MANIFEST.md).", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir.parent / "output_vtd.tar.gz"
    print(f"downloading {args.url} -> {archive}")
    urllib.request.urlretrieve(args.url, archive)

    if args.sha256:
        got = _sha256(archive)
        if got != args.sha256:
            print(f"checksum mismatch: {got} != {args.sha256}", file=sys.stderr)
            return 1
        print("checksum OK")

    print(f"extracting -> {args.output_dir}")
    with tarfile.open(archive) as tf:
        tf.extractall(args.output_dir.parent)
    n = len(list(args.output_dir.glob("*_vtd.npy")))
    print(f"done: {n} VTD files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
