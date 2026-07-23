from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MASt3R-SfM and export a COLMAP text model.")
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--scene-graph", choices=("swin", "logwin", "complete"), default="logwin")
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--cyclic", action="store_true")
    parser.add_argument("--shared-intrinsics", action="store_true")
    parser.add_argument("--coarse-iterations", type=int, default=300)
    parser.add_argument("--fine-iterations", type=int, default=300)
    parser.add_argument("--matching-confidence", type=float, default=5.0)
    parser.add_argument("--max-points", type=int, default=200_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository = args.repository.resolve()
    sys.path.insert(0, str(repository))

    import numpy as np  # noqa: I001
    import torch
    import mast3r.utils.path_to_dust3r  # noqa: F401
    from PIL import Image
    from scipy.spatial.transform import Rotation

    from dust3r.utils.image import load_images
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    from mast3r.image_pairs import make_pairs
    from mast3r.model import AsymmetricMASt3R

    image_paths = sorted(
        path.resolve()
        for path in args.images.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(image_paths) < 2:
        raise RuntimeError(f"Need at least two images, found {len(image_paths)}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    args.output.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output / "cache"
    model_dir = args.output / "model_txt"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    filelist = [str(path) for path in image_paths]
    images = load_images(filelist, size=args.image_size, verbose=True)
    scene_graph = _scene_graph_name(args.scene_graph, args.window_size, args.cyclic)
    pairs = make_pairs(images, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    print(f">> MASt3R-SfM graph={scene_graph} images={len(images)} directed_pairs={len(pairs)}")

    model = AsymmetricMASt3R.from_pretrained(str(args.weights.resolve()))
    model = model.to(args.device).eval()
    scene = sparse_global_alignment(
        filelist,
        pairs,
        str(cache_dir),
        model,
        lr1=0.07,
        niter1=args.coarse_iterations,
        lr2=0.01,
        niter2=args.fine_iterations,
        device=args.device,
        opt_depth=True,
        shared_intrinsics=args.shared_intrinsics,
        matching_conf_thr=args.matching_confidence,
    )

    cams2world = scene.get_im_poses().detach().cpu().numpy()
    intrinsics = scene.intrinsics.detach().cpu().numpy()
    processed_shapes = [image.shape[:2] for image in scene.imgs]
    registrations = _valid_registrations(
        image_paths,
        cams2world,
        intrinsics,
        processed_shapes,
        args.image_size,
        Image,
        Rotation,
        np,
    )
    points, colors = _collect_points(scene, args.max_points, np)
    _write_colmap_text(model_dir, registrations, points, colors)

    stats: dict[str, Any] = {
        "schema": "real2sim.mast3r_sfm.v1",
        "backend": "mast3r_sfm",
        "input_image_count": len(image_paths),
        "registered_image_count": len(registrations),
        "registration_success_rate": len(registrations) / len(image_paths),
        "sparse_point_count": len(points),
        "directed_pair_count": len(pairs),
        "scene_graph": scene_graph,
        "image_size": args.image_size,
        "shared_intrinsics": args.shared_intrinsics,
        "coarse_iterations": args.coarse_iterations,
        "fine_iterations": args.fine_iterations,
        "matching_confidence": args.matching_confidence,
        "weights": str(args.weights.resolve()),
        "repository": str(repository),
        "registered_images": [registration["name"] for registration in registrations],
    }
    (args.output / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(
        f">> registered {len(registrations)}/{len(image_paths)} images "
        f"({stats['registration_success_rate']:.2%}), sparse_points={len(points)}"
    )
    return 0


def _scene_graph_name(kind: str, window_size: int, cyclic: bool) -> str:
    if kind == "complete":
        return kind
    if window_size < 1:
        raise ValueError("window_size must be positive")
    suffix = "" if cyclic else "-noncyclic"
    return f"{kind}-{window_size}{suffix}"


def _valid_registrations(
    image_paths: list[Path],
    cams2world: Any,
    intrinsics: Any,
    processed_shapes: list[tuple[int, int]],
    image_size: int,
    image_module: Any,
    rotation_module: Any,
    np: Any,
) -> list[dict[str, Any]]:
    registrations: list[dict[str, Any]] = []
    for index, (path, cam2world, intrinsic, processed_shape) in enumerate(
        zip(image_paths, cams2world, intrinsics, processed_shapes, strict=True), start=1
    ):
        if not np.isfinite(cam2world).all() or not np.isfinite(intrinsic).all():
            continue
        if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
            continue
        with image_module.open(path) as image:
            original_width, original_height = image.size
        processed_height, processed_width = processed_shape
        resized_scale = image_size / max(original_width, original_height)
        resized_width = round(original_width * resized_scale)
        resized_height = round(original_height * resized_scale)
        crop_left = (resized_width - processed_width) / 2.0
        crop_top = (resized_height - processed_height) / 2.0
        scale_x = resized_width / original_width
        scale_y = resized_height / original_height

        world2cam = np.linalg.inv(cam2world)
        quat_xyzw = rotation_module.from_matrix(world2cam[:3, :3]).as_quat()
        registrations.append(
            {
                "id": index,
                "name": path.name,
                "width": original_width,
                "height": original_height,
                "fx": float(intrinsic[0, 0] / scale_x),
                "fy": float(intrinsic[1, 1] / scale_y),
                "cx": float((intrinsic[0, 2] + crop_left) / scale_x),
                "cy": float((intrinsic[1, 2] + crop_top) / scale_y),
                "qvec": (
                    float(quat_xyzw[3]),
                    float(quat_xyzw[0]),
                    float(quat_xyzw[1]),
                    float(quat_xyzw[2]),
                ),
                "tvec": tuple(float(value) for value in world2cam[:3, 3]),
            }
        )
    return registrations


def _collect_points(scene: Any, max_points: int, np: Any) -> tuple[Any, Any]:
    point_batches = [points.detach().cpu().numpy() for points in scene.get_sparse_pts3d()]
    color_batches = [np.asarray(colors) for colors in scene.get_pts3d_colors()]
    points = np.concatenate(point_batches, axis=0)
    colors = np.concatenate(color_batches, axis=0)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
    points = points[valid]
    colors = colors[valid]
    if max_points > 0 and len(points) > max_points:
        selected = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[selected]
        colors = colors[selected]
    colors = np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8)
    return points, colors


def _write_colmap_text(
    model_dir: Path,
    registrations: list[dict[str, Any]],
    points: Any,
    colors: Any,
) -> None:
    camera_rows = ["# Camera list with one line of data per camera:"]
    image_rows = ["# Image list with two lines of data per image:"]
    for registration in registrations:
        image_id = registration["id"]
        camera_rows.append(
            f"{image_id} PINHOLE {registration['width']} {registration['height']} "
            f"{registration['fx']:.12g} {registration['fy']:.12g} "
            f"{registration['cx']:.12g} {registration['cy']:.12g}"
        )
        qvec = " ".join(f"{value:.12g}" for value in registration["qvec"])
        tvec = " ".join(f"{value:.12g}" for value in registration["tvec"])
        image_rows.extend([f"{image_id} {qvec} {tvec} {image_id} {registration['name']}", ""])

    point_rows = ["# 3D point list with one line of data per point:"]
    for point_id, (point, color) in enumerate(zip(points, colors, strict=True), start=1):
        point_rows.append(
            f"{point_id} {point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
            f"{int(color[0])} {int(color[1])} {int(color[2])} 0"
        )
    (model_dir / "cameras.txt").write_text("\n".join(camera_rows) + "\n", encoding="ascii")
    (model_dir / "images.txt").write_text("\n".join(image_rows) + "\n", encoding="ascii")
    (model_dir / "points3D.txt").write_text("\n".join(point_rows) + "\n", encoding="ascii")


if __name__ == "__main__":
    raise SystemExit(main())
