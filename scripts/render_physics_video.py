from __future__ import annotations

import argparse
import json
import subprocess
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from real2sim_demo.process import resolve_executable  # type: ignore[import-untyped]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a MuJoCo physics scene and temporary collision probes to MP4."
    )
    parser.add_argument("--scene", type=Path, required=True, help="Validated scene.xml path.")
    parser.add_argument("--output", type=Path, required=True, help="Output MP4 path.")
    parser.add_argument("--report", type=Path, help="Optional JSON render report path.")
    parser.add_argument("--manifest", type=Path, help="Optional Stage 3 manifest to update.")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg executable.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--probe-height", type=float, default=1.15)
    parser.add_argument("--no-probe", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.width <= 0 or args.height <= 0 or args.fps <= 0 or args.duration <= 0:
        raise ValueError("width, height, fps, and duration must be positive")
    if not args.scene.is_file():
        raise FileNotFoundError(args.scene)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_path = args.report or args.output.with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    import mujoco  # type: ignore[import-untyped]

    base_model = mujoco.MjModel.from_xml_path(str(args.scene))
    base_center = np.asarray(base_model.stat.center, dtype=np.float64).copy()
    base_extent = float(base_model.stat.extent)
    render_xml = _make_render_scene(
        args.scene,
        probe_height=args.probe_height,
        add_probes=not args.no_probe,
    )

    ffmpeg = resolve_executable(args.ffmpeg)
    frame_count = max(1, int(round(args.duration * args.fps)))
    command = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{args.width}x{args.height}",
        "-r",
        str(args.fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(args.output),
    ]
    model = mujoco.MjModel.from_xml_path(str(render_xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = base_center
    visual_radius = float(np.max(model.geom_rbound))
    camera.distance = max(visual_radius * 1.8, min(base_extent, 0.8))
    camera.elevation = -20.0

    contact_frames = 0
    max_contacts = 0
    probe_geom_ids = _probe_geom_ids(mujoco, model)
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        for frame_index in range(frame_count):
            if frame_index > 0:
                for _ in range(max(1, int(round(0.02 / model.opt.timestep)))):
                    mujoco.mj_step(model, data)
            current_contacts = _probe_contact_count(mujoco, data, probe_geom_ids)
            if current_contacts > 0:
                contact_frames += 1
            max_contacts = max(max_contacts, current_contacts)
            camera.azimuth = 35.0 + 360.0 * frame_index / frame_count
            renderer.update_scene(data, camera=camera)
            frame = np.asarray(renderer.render(), dtype=np.uint8)
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command, stdout, stderr)
    renderer.close()
    render_xml.unlink(missing_ok=True)

    report: dict[str, Any] = {
        "schema": "real2sim.physics_video.v1",
        "scene_xml": str(args.scene.resolve()),
        "output": str(args.output.resolve()),
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "duration_s": frame_count / args.fps,
        "frames": frame_count,
        "probe_enabled": not args.no_probe,
        "probe_contact_frames": contact_frames,
        "probe_max_contacts": max_contacts,
        "mujoco_version": mujoco.__version__,
        "ffmpeg": command,
        "created_at": datetime.now(UTC).isoformat(),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.manifest is not None:
        _update_manifest(args.manifest, args.output, report)
    print(json.dumps(report, indent=2))
    return 0


def _make_render_scene(scene: Path, *, probe_height: float, add_probes: bool) -> Path:
    tree = ET.parse(scene)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Scene XML has no worldbody")
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "video_skybox",
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.20 0.24 0.30",
            "rgb2": "0.025 0.03 0.04",
            "width": "512",
            "height": "3072",
        },
    )
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.attrib.update(
        {
            "ambient": "0.45 0.45 0.45",
            "diffuse": "0.85 0.85 0.85",
            "specular": "0.15 0.15 0.15",
        }
    )
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "video_key_light",
            "directional": "true",
            "pos": "0 -1 2",
            "dir": "0 0 -1",
            "diffuse": "0.8 0.8 0.8",
            "specular": "0.2 0.2 0.2",
        },
    )
    if not add_probes:
        ET.indent(tree, space="  ")
        temp_path = scene.with_name(f".{scene.stem}.video_render.xml")
        tree.write(temp_path, encoding="utf-8", xml_declaration=True)
        return temp_path
    positions = [
        (-0.28, -0.28),
        (0.04, -0.28),
        (0.36, -0.28),
        (-0.28, 0.04),
        (0.04, 0.04),
        (0.36, 0.04),
        (-0.28, 0.36),
        (0.04, 0.36),
        (0.36, 0.36),
    ]
    for index, (x, y) in enumerate(positions):
        body = ET.SubElement(
            worldbody,
            "body",
            {
                "name": f"video_probe_{index:02d}",
                "pos": f"{x} {y} {probe_height}",
            },
        )
        ET.SubElement(body, "freejoint")
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"video_probe_geom_{index:02d}",
                "type": "sphere",
                "size": "0.018",
                "mass": "0.04",
                "rgba": "0.95 0.12 0.03 1",
                "contype": "1",
                "conaffinity": "1",
            },
        )
    ET.indent(tree, space="  ")
    temp_path = scene.with_name(f".{scene.stem}.video_render.xml")
    tree.write(temp_path, encoding="utf-8", xml_declaration=True)
    return temp_path


def _probe_geom_ids(mujoco: Any, model: Any) -> set[int]:
    result: set[int] = set()
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name.startswith("video_probe_geom_"):
            result.add(geom_id)
    return result


def _probe_contact_count(mujoco: Any, data: Any, probe_geom_ids: set[int]) -> int:
    count = 0
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        if int(contact.geom1) in probe_geom_ids or int(contact.geom2) in probe_geom_ids:
            count += 1
    return count


def _update_manifest(manifest_path: Path, output: Path, report: dict[str, Any]) -> None:
    payload: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "real2sim.physics.v1":
        raise ValueError(f"Unsupported manifest schema: {payload.get('schema')}")
    payload.setdefault("artifacts", {})["video"] = str(output.resolve())
    payload["video"] = report
    payload["updated_at"] = datetime.now(UTC).isoformat()
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    provenance_path = manifest_path.with_name("provenance.json")
    if provenance_path.is_file():
        provenance: dict[str, Any] = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance.setdefault("artifacts", {})["video"] = str(output.resolve())
        provenance["video"] = report
        provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
