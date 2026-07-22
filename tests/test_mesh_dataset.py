from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
from PIL import Image

from real2sim_demo.mesh_dataset import prepare_2dgs_dataset, read_colmap_image_names


def test_prepare_2dgs_dataset_copies_registered_images_and_binary_model(
    tmp_path: Path,
) -> None:
    run_dir = _stage1_run(tmp_path, stage="gaussian_trained", names=["a.png", "nested/b.png"])
    output = tmp_path / "stage2" / "dataset"

    provenance = prepare_2dgs_dataset(run_dir, output)

    assert provenance["registered_image_count"] == 2
    assert (output / "images" / "a.png").is_file()
    assert (output / "images" / "nested" / "b.png").is_file()
    assert (output / "sparse" / "0" / "cameras.bin").read_bytes() == b"camera"
    assert read_colmap_image_names(output / "sparse" / "0" / "images.bin") == [
        "a.png",
        "nested/b.png",
    ]


def test_prepare_2dgs_dataset_rejects_invalid_stage1_state(tmp_path: Path) -> None:
    run_dir = _stage1_run(tmp_path, stage="failed", names=["a.png", "b.png"])

    with pytest.raises(ValueError, match="prepared or gaussian_trained"):
        prepare_2dgs_dataset(run_dir, tmp_path / "dataset")


def test_read_colmap_image_names_rejects_truncated_points(tmp_path: Path) -> None:
    path = tmp_path / "images.bin"
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<I7dI", 1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1))
        handle.write(b"frame.png\0")
        handle.write(struct.pack("<Q", 1))

    with pytest.raises(ValueError, match="Truncated COLMAP points2D"):
        read_colmap_image_names(path)


def _stage1_run(tmp_path: Path, *, stage: str, names: list[str]) -> Path:
    run_dir = tmp_path / "stage1"
    source = run_dir / "colmap" / "undistorted"
    images = source / "images"
    model = source / "sparse"
    images.mkdir(parents=True)
    model.mkdir(parents=True)
    for index, name in enumerate(names):
        path = images.joinpath(*name.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (index * 30, 20, 40)).save(path)
    _write_images_bin(model / "images.bin", names)
    (model / "cameras.bin").write_bytes(b"camera")
    (model / "points3D.bin").write_bytes(b"points")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "real2sim.stage1.manifest.v1",
                "run_id": "input",
                "stage": stage,
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def _write_images_bin(path: Path, names: list[str]) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(names)))
        for image_id, name in enumerate(names, start=1):
            handle.write(
                struct.pack(
                    "<I7dI",
                    image_id,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1,
                )
            )
            handle.write(name.encode("utf-8") + b"\0")
            handle.write(struct.pack("<Q", 0))
