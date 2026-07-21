from __future__ import annotations

import struct
from pathlib import Path

from real2sim_demo.colmap import _select_largest_model


def test_select_largest_colmap_model_by_registered_images(tmp_path: Path) -> None:
    sparse_root = tmp_path / "sparse"
    small = sparse_root / "0"
    large = sparse_root / "1"
    small.mkdir(parents=True)
    large.mkdir()
    (small / "images.bin").write_bytes(struct.pack("<Q", 5))
    (large / "images.bin").write_bytes(struct.pack("<Q", 17))

    assert _select_largest_model(sparse_root) == large
