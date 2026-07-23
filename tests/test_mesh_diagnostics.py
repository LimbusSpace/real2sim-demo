from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageStat

from real2sim_demo.mesh_diagnostics import build_triptychs


def test_build_triptychs_exports_each_frame_and_contact_sheet(tmp_path: Path) -> None:
    gt_dir = tmp_path / "gt"
    render_dir = tmp_path / "renders"
    depth_dir = tmp_path / "vis"
    output_dir = tmp_path / "diagnostics"
    camera_map = tmp_path / "cameras.json"
    for path in (gt_dir, render_dir, depth_dir):
        path.mkdir()

    for index in range(2):
        ground_truth = np.zeros((16, 24, 3), dtype=np.uint8)
        ground_truth[:, :, index] = 180
        rendered = ground_truth.copy()
        rendered[:, 8:12, :] = 120
        depth = np.linspace(0.2 + index, 2.0 + index, 16 * 24, dtype=np.float32).reshape(16, 24)
        Image.fromarray(ground_truth).save(gt_dir / f"{index:05d}.png")
        Image.fromarray(rendered).save(render_dir / f"{index:05d}.png")
        Image.fromarray(depth, mode="F").save(depth_dir / f"depth_{index:05d}.tiff")
    camera_map.write_text(
        json.dumps([{"img_name": "frame_a"}, {"img_name": "frame_b"}]),
        encoding="utf-8",
    )

    report = build_triptychs(
        gt_dir,
        render_dir,
        depth_dir,
        output_dir,
        max_height=64,
        camera_map=camera_map,
    )

    assert report["frame_count"] == 2
    assert report["layout"] == ["ground_truth", "2dgs_rgb", "surf_depth"]
    assert report["frames"][0]["image_name"] == "frame_a"
    assert report["depth_percentile_98"] > report["depth_percentile_02"]
    assert report["metrics"]["mean_psnr_db"] > 10.0
    triptychs = sorted((output_dir / "triptychs").glob("*.jpg"))
    assert len(triptychs) == 2
    assert (output_dir / "contact_sheet.jpg").is_file()
    persisted = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert persisted["frame_count"] == 2
    with Image.open(triptychs[0]) as image:
        assert image.width == 76
        assert image.height == 44
        assert any(low != high for low, high in ImageStat.Stat(image).extrema)
