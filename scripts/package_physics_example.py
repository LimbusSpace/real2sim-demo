from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from real2sim_demo.physics_validate import validate_mujoco_scene  # type: ignore[import-untyped]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Package a validated Stage 3 run as a portable repository example."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "real2sim.physics.v1":
        raise ValueError(f"Unsupported Stage 3 manifest schema: {manifest.get('schema')}")
    if manifest.get("stage") != "validated":
        raise ValueError("Stage 3 manifest must have stage='validated'")

    source_mjcf = args.run_dir / "mjcf"
    source_video = args.run_dir / "videos" / "scene_preview.mp4"
    source_video_report = args.run_dir / "videos" / "scene_preview.json"
    source_transform = args.run_dir / "source" / "transform.json"
    required = [
        source_mjcf / "scene.xml",
        source_mjcf / "scene.obj",
        source_mjcf / "scene.mtl",
        source_video,
        source_video_report,
        source_transform,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing package inputs: " + ", ".join(missing))
    collision_paths = sorted(source_mjcf.glob("scene_collision_*.obj"))
    if not collision_paths:
        raise ValueError("Validated Stage 3 run contains no collision meshes")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected_collision_names = {path.name for path in collision_paths}
    for stale in args.output_dir.glob("scene_collision_*.obj"):
        if stale.name not in expected_collision_names:
            stale.unlink()
    for source in [
        source_mjcf / "scene.xml",
        source_mjcf / "scene.obj",
        source_mjcf / "scene.mtl",
        *collision_paths,
    ]:
        shutil.copy2(source, args.output_dir / source.name)

    video_dir = args.output_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_video, video_dir / source_video.name)
    video_report: dict[str, Any] = json.loads(source_video_report.read_text(encoding="utf-8"))
    video_report["scene_xml"] = "scene.xml"
    video_report["output"] = "video/scene_preview.mp4"
    if isinstance(video_report.get("ffmpeg"), list) and video_report["ffmpeg"]:
        video_report["ffmpeg"][0] = "ffmpeg"
        video_report["ffmpeg"][-1] = "video/scene_preview.mp4"
    _write_json(video_dir / "scene_preview.json", video_report)

    transform: dict[str, Any] = json.loads(source_transform.read_text(encoding="utf-8"))
    transform["input_mesh"] = "stage2://mesh/post.ply"
    _write_json(args.output_dir / "transform.json", transform)

    validation = validate_mujoco_scene(
        args.output_dir / "scene.xml",
        args.output_dir / "validation.json",
        steps=10,
    )
    validation["scene_xml"] = "scene.xml"
    _write_json(args.output_dir / "validation.json", validation)

    package_manifest = {
        "schema": "real2sim.physics_package.v1",
        "source_manifest_sha256": _sha256(manifest_path),
        "source_input_mesh_sha256": manifest.get("input_stage2", {}).get("mesh_sha256"),
        "versions": manifest.get("versions", {}),
        "transform": {
            "target_extent_m": transform.get("target_extent_m"),
            "scale_m_per_unit": transform.get("scale_m_per_unit"),
            "support_plane_inlier_fraction": transform.get("support_plane", {}).get(
                "inlier_fraction"
            ),
        },
        "collision": {
            "count": len(collision_paths),
            "total_vertex_count": manifest.get("collision", {}).get("total_vertex_count"),
            "total_face_count": manifest.get("collision", {}).get("total_face_count"),
        },
        "validation": validation,
        "video": video_report,
        "files": _file_records(args.output_dir),
    }
    _write_json(args.output_dir / "package_manifest.json", package_manifest)
    print(json.dumps(package_manifest, indent=2))
    return 0


def _file_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if path.name == "package_manifest.json":
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    raise SystemExit(main())
