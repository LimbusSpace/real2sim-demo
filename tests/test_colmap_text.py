from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from real2sim_demo.colmap_text import convert_colmap_text_to_hyworld


def test_colmap_text_is_converted_to_hyworld_dataset(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    source_images = tmp_path / "source_images"
    output_dir = tmp_path / "dataset"
    model_dir.mkdir()
    source_images.mkdir()
    (model_dir / "cameras.txt").write_text(
        "# Camera list\n1 PINHOLE 4 3 2 2 1.5 1.0\n", encoding="utf-8"
    )
    (model_dir / "images.txt").write_text(
        "# Image list\n"
        "1 1 0 0 0 0 0 0 1 frame_000001.jpg\n"
        "0 0 0 0 0 0\n"
        "2 1 0 0 0 -1 0 0 1 frame_000002.jpg\n"
        "0 0 0 0 0 0\n",
        encoding="utf-8",
    )
    (model_dir / "points3D.txt").write_text(
        "# Point list\n1 0 0 1 255 10 20 0.1 1 0\n"
        "2 1 0 1 10 255 20 0.2 2 0\n",
        encoding="utf-8",
    )
    for name in ("frame_000001.jpg", "frame_000002.jpg"):
        Image.new("RGB", (4, 3), (20, 30, 40)).save(source_images / name)

    provenance = convert_colmap_text_to_hyworld(model_dir, source_images, output_dir)

    assert provenance["image_count"] == 2
    assert (output_dir / "images/frame_000001.png").is_file()
    assert (output_dir / "points.ply").read_text(encoding="ascii").count("\n") > 10
    cameras = json.loads((output_dir / "cameras.json").read_text(encoding="utf-8"))
    assert cameras["frame_000001"]["intrinsic"] == [
        [2.0, 0.0, 1.5],
        [0.0, 2.0, 1.0],
        [0.0, 0.0, 1.0],
    ]
    assert cameras["frame_000002"]["extrinsic"][0][3] == -1.0
