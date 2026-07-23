from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

FloatArray = NDArray[np.float32]

_DEPTH_COLORS = np.asarray(
    [
        [24, 24, 96],
        [25, 116, 179],
        [53, 183, 121],
        [246, 215, 70],
        [204, 54, 47],
    ],
    dtype=np.float32,
)


def build_triptychs(
    gt_dir: Path,
    render_dir: Path,
    depth_dir: Path,
    output_dir: Path,
    *,
    max_height: int = 480,
    camera_map: Path | None = None,
) -> dict[str, Any]:
    if max_height < 64:
        raise ValueError("max_height must be at least 64")
    gt_paths = sorted(gt_dir.glob("*.png"))
    if not gt_paths:
        raise FileNotFoundError(f"No ground-truth PNG images found in {gt_dir}")

    frames: list[tuple[Path, Path, Path]] = []
    for gt_path in gt_paths:
        render_path = render_dir / gt_path.name
        depth_path = depth_dir / f"depth_{gt_path.stem}.tiff"
        if not render_path.is_file() or not depth_path.is_file():
            raise FileNotFoundError(
                f"Missing diagnostic counterpart for {gt_path.name}: "
                f"render={render_path}, depth={depth_path}"
            )
        frames.append((gt_path, render_path, depth_path))

    image_names = _camera_names(camera_map, len(frames))

    depth_near, depth_far, valid_fraction = _depth_range([item[2] for item in frames])
    triptych_dir = output_dir / "triptychs"
    triptych_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, Any]] = []
    triptychs: list[Path] = []

    for index, (gt_path, render_path, depth_path) in enumerate(frames):
        with Image.open(gt_path) as source_image, Image.open(render_path) as rendered_image:
            ground_truth = source_image.convert("RGB")
            rendered = rendered_image.convert("RGB")
        if rendered.size != ground_truth.size:
            rendered = rendered.resize(ground_truth.size, Image.Resampling.LANCZOS)

        depth = _read_depth(depth_path)
        depth_image = _colorize_depth(depth, depth_near, depth_far)
        if depth_image.size != ground_truth.size:
            depth_image = depth_image.resize(ground_truth.size, Image.Resampling.BILINEAR)

        triptych = _compose_triptych(
            ground_truth,
            rendered,
            depth_image,
            max_height=max_height,
            image_name=image_names[index],
        )
        output_path = triptych_dir / f"{gt_path.stem}.jpg"
        triptych.save(output_path, quality=90, optimize=True, subsampling=0)
        triptychs.append(output_path)

        psnr, mae = _image_metrics(ground_truth, rendered)
        metrics.append(
            {
                "index": index,
                "image_name": image_names[index],
                "ground_truth": str(gt_path.resolve()),
                "render": str(render_path.resolve()),
                "depth": str(depth_path.resolve()),
                "triptych": str(output_path.resolve()),
                "psnr_db": round(psnr, 4),
                "mae": round(mae, 6),
            }
        )

    contact_sheet = output_dir / "contact_sheet.jpg"
    _write_contact_sheet(triptychs, contact_sheet)
    psnr_values = [float(item["psnr_db"]) for item in metrics]
    mae_values = [float(item["mae"]) for item in metrics]
    report: dict[str, Any] = {
        "schema": "real2sim.2dgs_diagnostics.v1",
        "frame_count": len(frames),
        "layout": ["ground_truth", "2dgs_rgb", "surf_depth"],
        "depth_units": "sfm_scene_units",
        "depth_percentile_02": depth_near,
        "depth_percentile_98": depth_far,
        "depth_valid_fraction": valid_fraction,
        "metrics": {
            "mean_psnr_db": round(float(np.mean(psnr_values)), 4),
            "min_psnr_db": round(float(np.min(psnr_values)), 4),
            "mean_mae": round(float(np.mean(mae_values)), 6),
            "max_mae": round(float(np.max(mae_values)), 6),
        },
        "triptych_dir": str(triptych_dir.resolve()),
        "contact_sheet": str(contact_sheet.resolve()),
        "frames": metrics,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _depth_range(paths: list[Path]) -> tuple[float, float, float]:
    samples: list[FloatArray] = []
    valid_count = 0
    pixel_count = 0
    for path in paths:
        depth = _read_depth(path)
        valid = depth[np.isfinite(depth) & (depth > 0)]
        valid_count += int(valid.size)
        pixel_count += int(depth.size)
        if valid.size:
            stride = max(1, math.ceil(valid.size / 50_000))
            samples.append(valid[::stride])
    if not samples:
        raise ValueError("2DGS diagnostic depth maps contain no finite positive values")
    combined = np.concatenate(samples)
    near, far = np.percentile(combined, [2.0, 98.0])
    if not np.isfinite(near) or not np.isfinite(far) or far <= near:
        raise ValueError(f"Invalid diagnostic depth range: near={near}, far={far}")
    return float(near), float(far), round(valid_count / pixel_count, 6)


def _read_depth(path: Path) -> FloatArray:
    with Image.open(path) as image:
        depth = np.asarray(image, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected a single-channel depth image: {path} shape={depth.shape}")
    return depth


def _colorize_depth(depth: FloatArray, near: float, far: float) -> Image.Image:
    valid = np.isfinite(depth) & (depth > 0)
    normalized = np.zeros_like(depth, dtype=np.float32)
    normalized[valid] = np.clip((depth[valid] - near) / (far - near), 0.0, 1.0)
    scaled = normalized * (_DEPTH_COLORS.shape[0] - 1)
    lower = np.floor(scaled).astype(np.intp)
    upper = np.clip(lower + 1, 0, _DEPTH_COLORS.shape[0] - 1)
    fraction = (scaled - lower)[..., None]
    colors = _DEPTH_COLORS[lower] * (1.0 - fraction) + _DEPTH_COLORS[upper] * fraction
    colors[~valid] = 0
    return Image.fromarray(np.asarray(np.rint(colors), dtype=np.uint8), mode="RGB")


def _compose_triptych(
    ground_truth: Image.Image,
    rendered: Image.Image,
    depth: Image.Image,
    *,
    max_height: int,
    image_name: str,
) -> Image.Image:
    target_height = min(max_height, ground_truth.height)
    target_width = max(1, round(ground_truth.width * target_height / ground_truth.height))
    size = (target_width, target_height)
    panels = [
        ground_truth.resize(size, Image.Resampling.LANCZOS),
        rendered.resize(size, Image.Resampling.LANCZOS),
        depth.resize(size, Image.Resampling.BILINEAR),
    ]
    labels = [f"Ground truth - {image_name}", "2DGS RGB", "Surf depth"]
    header_height = 28
    gap = 2
    canvas = Image.new(
        "RGB",
        (target_width * 3 + gap * 2, target_height + header_height),
        (20, 23, 28),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (label, panel) in enumerate(zip(labels, panels, strict=True)):
        left = index * (target_width + gap)
        draw.text((left + 10, 8), label, fill=(245, 245, 245), font=font)
        canvas.paste(panel, (left, header_height))
    return canvas


def _image_metrics(ground_truth: Image.Image, rendered: Image.Image) -> tuple[float, float]:
    gt = np.asarray(ground_truth, dtype=np.float32) / 255.0
    prediction = np.asarray(rendered, dtype=np.float32) / 255.0
    difference = prediction - gt
    mse = float(np.mean(np.square(difference)))
    mae = float(np.mean(np.abs(difference)))
    psnr = 99.0 if mse <= 1e-12 else min(99.0, 10.0 * math.log10(1.0 / mse))
    return psnr, mae


def _camera_names(path: Path | None, expected_count: int) -> list[str]:
    if path is None:
        return [f"view_{index:05d}" for index in range(expected_count)]
    if not path.is_file():
        raise FileNotFoundError(f"2DGS camera map does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != expected_count:
        raise ValueError(
            f"2DGS camera map count mismatch: expected={expected_count}, "
            f"actual={len(payload) if isinstance(payload, list) else 'not-a-list'}"
        )
    names: list[str] = []
    for index, camera in enumerate(payload):
        if not isinstance(camera, dict) or not camera.get("img_name"):
            raise ValueError(f"2DGS camera map entry {index} has no img_name")
        names.append(str(camera["img_name"]))
    return names


def _write_contact_sheet(paths: list[Path], output_path: Path) -> None:
    selection_count = min(12, len(paths))
    if selection_count == 1:
        selected = [paths[0]]
    else:
        selected = [
            paths[round(index * (len(paths) - 1) / (selection_count - 1))]
            for index in range(selection_count)
        ]
    thumbnails: list[Image.Image] = []
    for path in selected:
        with Image.open(path) as image:
            thumbnail = image.convert("RGB")
        thumbnail.thumbnail((1200, 260), Image.Resampling.LANCZOS)
        thumbnails.append(thumbnail)

    columns = min(2, len(thumbnails))
    rows = math.ceil(len(thumbnails) / columns)
    cell_width = max(image.width for image in thumbnails)
    cell_height = max(image.height for image in thumbnails)
    sheet = Image.new("RGB", (cell_width * columns, cell_height * rows), (20, 23, 28))
    for index, thumbnail in enumerate(thumbnails):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        sheet.paste(thumbnail, (x, y))
    sheet.save(output_path, quality=88, optimize=True, subsampling=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build 2DGS GT/render/depth diagnostics.")
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-map", type=Path)
    parser.add_argument("--max-height", type=int, default=480)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_triptychs(
        args.gt_dir,
        args.render_dir,
        args.depth_dir,
        args.output_dir,
        max_height=args.max_height,
        camera_map=args.camera_map,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
