from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from .mesh_ply import validate_mesh_ply
from .physics_collision import build_coacd_command, collect_collision_stats, read_obj_stats
from .physics_config import Stage3Settings
from .physics_geometry import prepare_visual_mesh
from .physics_mjcf import assemble_mjcf, build_obj2mjcf_command, validate_mjcf_structure
from .physics_validate import validate_mujoco_scene
from .process import CommandResult, resolve_executable, run_command

EXPECTED_PACKAGE_VERSIONS = {
    "coacd": "1.0.11",
    "obj2mjcf": "0.0.25",
    "mujoco": "3.10.0",
    "trimesh": "4.12.2",
    "fast-simplification": "0.1.13",
}


def run_stage3(
    settings: Stage3Settings,
    *,
    stage: str = "all",
    dry_run: bool = False,
) -> dict[str, Any]:
    if stage not in {"prepare", "decompose", "mjcf", "validate", "all"}:
        raise ValueError(f"Unsupported Stage 3 stage: {stage}")
    settings.validate()
    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_or_create_manifest(settings)
    manifest.pop("error", None)
    manifest["settings"] = settings.to_dict()
    manifest["commands"] = []
    manifest.setdefault("artifacts", {})
    manifest.setdefault("signatures", {})
    manifest.setdefault("logs", {})
    trace: list[dict[str, Any]] = []

    source_dir = output_dir / "source"
    collision_dir = output_dir / "collision"
    mjcf_dir = output_dir / "mjcf"
    logs_dir = output_dir / "logs"
    source_obj = source_dir / "scene.obj"
    source_mtl = source_dir / "scene.mtl"
    transform_path = source_dir / "transform.json"
    validation_path = output_dir / "validation.json"

    try:
        versions = _package_versions(dry_run=dry_run)
        manifest["versions"] = versions
        input_reference = (
            _planned_stage2_reference(settings.input_mesh_run_dir)
            if dry_run
            else _validate_stage2_input(settings.input_mesh_run_dir)
        )
        manifest["input_stage2"] = input_reference
        signatures = _stage_signatures(settings, input_reference, versions)

        if stage in {"prepare", "all"}:
            if dry_run:
                transform = _planned_transform(settings, input_reference)
                _append_completed_event(trace, "prepare", reused=False)
            else:
                reused = _can_reuse_prepare(
                    manifest,
                    signatures["prepare"],
                    source_obj,
                    source_mtl,
                    transform_path,
                )
                if reused:
                    transform = json.loads(transform_path.read_text(encoding="utf-8"))
                    _append_completed_event(trace, "prepare", reused=True)
                else:
                    transform, _ = _timed_step(
                        trace,
                        "prepare",
                        lambda: prepare_visual_mesh(
                            Path(input_reference["mesh"]),
                            source_obj,
                            source_mtl,
                            transform_path,
                            settings.geometry,
                        ),
                    )
                    _invalidate_after(manifest, "prepare")
                _write_json(logs_dir / "10_prepare.json", transform)
                manifest["logs"]["prepare"] = str((logs_dir / "10_prepare.json").resolve())
            manifest["transform"] = transform
            manifest["artifacts"].update(
                {
                    "source_obj": str(source_obj.resolve()),
                    "source_mtl": str(source_mtl.resolve()),
                    "transform": str(transform_path.resolve()),
                }
            )
            manifest["signatures"]["prepare"] = signatures["prepare"]
            manifest["stage"] = "commands_planned" if dry_run else "prepared"

        if stage in {"decompose", "all"}:
            if not dry_run:
                _require_signature(manifest, "prepare", signatures["prepare"], source_obj)
            command = build_coacd_command(sys.executable, source_obj, collision_dir, settings.coacd)
            reused = False
            if dry_run:
                result = run_command(command, logs_dir / "20_coacd.log", dry_run=True)
                collision_stats: dict[str, Any] = {
                    "count": None,
                    "total_vertex_count": None,
                    "total_face_count": None,
                    "parts": [],
                    "dry_run": True,
                }
                _append_completed_event(trace, "decompose", reused=False)
            else:
                reused = _can_reuse_collision(
                    manifest,
                    signatures["decompose"],
                    collision_dir,
                    settings.coacd.max_convex_hulls,
                )
                if reused:
                    collision_stats = collect_collision_stats(collision_dir)
                    result = None
                    _append_completed_event(trace, "decompose", reused=True)
                else:
                    result, _ = _timed_step(
                        trace,
                        "decompose",
                        lambda: run_command(command, logs_dir / "20_coacd.log"),
                    )
                    collision_stats = collect_collision_stats(collision_dir)
                    if collision_stats["count"] > settings.coacd.max_convex_hulls:
                        raise ValueError("CoACD exceeded coacd.max_convex_hulls")
                    _invalidate_after(manifest, "decompose")
            _record_command(
                manifest,
                "decompose",
                command,
                logs_dir / "20_coacd.log",
                result,
                reused,
            )
            manifest["collision"] = collision_stats
            manifest["artifacts"]["collision_dir"] = str(collision_dir.resolve())
            manifest["signatures"]["decompose"] = signatures["decompose"]
            manifest["stage"] = "commands_planned" if dry_run else "collision_decomposed"

        if stage in {"mjcf", "all"}:
            if not dry_run:
                _require_signature(manifest, "prepare", signatures["prepare"], source_obj)
                _require_signature(manifest, "decompose", signatures["decompose"], collision_dir)
            executable = resolve_executable("obj2mjcf")
            command = build_obj2mjcf_command(executable, source_dir)
            reused = False
            if dry_run:
                result = run_command(command, logs_dir / "30_obj2mjcf.log", dry_run=True)
                mjcf_stats: dict[str, Any] = {
                    "scene_xml": str((mjcf_dir / "scene.xml").resolve()),
                    "dry_run": True,
                }
                _append_completed_event(trace, "mjcf", reused=False)
            else:
                reused = _can_reuse_mjcf(manifest, signatures["mjcf"], mjcf_dir / "scene.xml")
                if reused:
                    result = None
                    mjcf_stats = validate_mjcf_structure(mjcf_dir / "scene.xml")
                    mjcf_stats["scene_xml"] = str((mjcf_dir / "scene.xml").resolve())
                    _append_completed_event(trace, "mjcf", reused=True)
                else:
                    result, _ = _timed_step(
                        trace,
                        "mjcf_obj2mjcf",
                        lambda: run_command(command, logs_dir / "30_obj2mjcf.log"),
                    )
                    mjcf_stats, _ = _timed_step(
                        trace,
                        "mjcf",
                        lambda: assemble_mjcf(
                            source_dir / "scene",
                            source_mtl,
                            collision_dir,
                            mjcf_dir,
                            gravity=settings.mujoco.gravity,
                        ),
                    )
                    _invalidate_after(manifest, "mjcf")
            _record_command(manifest, "mjcf", command, logs_dir / "30_obj2mjcf.log", result, reused)
            manifest["mjcf"] = mjcf_stats
            manifest["artifacts"]["mjcf_dir"] = str(mjcf_dir.resolve())
            manifest["artifacts"]["scene_xml"] = str((mjcf_dir / "scene.xml").resolve())
            manifest["signatures"]["mjcf"] = signatures["mjcf"]
            manifest["stage"] = "commands_planned" if dry_run else "mjcf_exported"

        if stage in {"validate", "all"}:
            if not dry_run:
                _require_signature(manifest, "mjcf", signatures["mjcf"], mjcf_dir / "scene.xml")
            if dry_run:
                validation: dict[str, Any] = {
                    "schema": "real2sim.physics_validation.v1",
                    "success": None,
                    "steps": 10,
                    "dry_run": True,
                }
                _append_completed_event(trace, "validate", reused=False)
            else:
                reused = _can_reuse_validation(
                    manifest, signatures["validate"], validation_path
                )
                if reused:
                    validation = json.loads(validation_path.read_text(encoding="utf-8"))
                    _append_completed_event(trace, "validate", reused=True)
                else:
                    validation, _ = _timed_step(
                        trace,
                        "validate",
                        lambda: validate_mujoco_scene(
                            mjcf_dir / "scene.xml", validation_path, steps=10
                        ),
                    )
                _write_json(logs_dir / "40_validate.json", validation)
                manifest["logs"]["validate"] = str((logs_dir / "40_validate.json").resolve())
            manifest["validation"] = validation
            manifest["artifacts"]["validation"] = str(validation_path.resolve())
            manifest["signatures"]["validate"] = signatures["validate"]
            manifest["stage"] = "commands_planned" if dry_run else "validated"
    except Exception as exc:
        manifest["stage"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["dry_run"] = dry_run
        manifest["updated_at"] = _now()
        _write_json(output_dir / "manifest.json", manifest)
        _write_json(output_dir / "trace.json", {"schema": "real2sim.trace.v1", "events": trace})
        _write_json(
            output_dir / "provenance.json",
            {
                "schema": "real2sim.physics_provenance.v1",
                "input_stage2": manifest.get("input_stage2", {}),
                "settings": settings.to_dict(),
                "versions": manifest.get("versions", {}),
                "signatures": manifest.get("signatures", {}),
                "transform": manifest.get("transform", {}),
                "artifacts": manifest.get("artifacts", {}),
                "commands": manifest.get("commands", []),
            },
        )
    return manifest


def _validate_stage2_input(input_dir: Path) -> dict[str, Any]:
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing Stage 2 manifest: {manifest_path}")
    payload: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "real2sim.mesh.v1":
        raise ValueError(f"Unsupported Stage 2 manifest schema: {payload.get('schema')}")
    if payload.get("stage") != "mesh_exported":
        raise ValueError("Stage 2 manifest must have stage='mesh_exported'")
    mesh_path = input_dir / "mesh" / "post.ply"
    stats = validate_mesh_ply(mesh_path)
    return {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "schema": payload["schema"],
        "run_id": payload.get("run_id"),
        "stage": payload["stage"],
        "mesh": str(mesh_path.resolve()),
        "mesh_sha256": _sha256(mesh_path),
        "mesh_stats": stats.to_dict(),
        "upstream": payload.get("upstream", {}),
    }


def _planned_stage2_reference(input_dir: Path) -> dict[str, Any]:
    return {
        "manifest": str((input_dir / "manifest.json").resolve()),
        "manifest_sha256": None,
        "schema": "real2sim.mesh.v1",
        "stage": "mesh_exported",
        "mesh": str((input_dir / "mesh" / "post.ply").resolve()),
        "mesh_sha256": None,
        "mesh_stats": None,
        "dry_run": True,
    }


def _package_versions(*, dry_run: bool) -> dict[str, str]:
    if sys.version_info[:2] != (3, 11) and not dry_run:
        raise RuntimeError(f"Stage 3 requires Python 3.11, got {sys.version.split()[0]}")
    versions = {name: importlib.metadata.version(name) for name in EXPECTED_PACKAGE_VERSIONS}
    for name, expected in EXPECTED_PACKAGE_VERSIONS.items():
        if versions[name] != expected:
            raise RuntimeError(f"{name} version mismatch: {versions[name]} expected={expected}")
    return {"python": sys.version.split()[0], **versions}


def _stage_signatures(
    settings: Stage3Settings,
    input_reference: dict[str, Any],
    versions: dict[str, str],
) -> dict[str, str]:
    prepare = _fingerprint(
        {
            "input_manifest": input_reference.get("manifest_sha256"),
            "input_mesh": input_reference.get("mesh_sha256"),
            "geometry": asdict(settings.geometry),
            "trimesh": versions["trimesh"],
            "fast-simplification": versions["fast-simplification"],
        }
    )
    decompose = _fingerprint(
        {"prepare": prepare, "coacd": asdict(settings.coacd), "version": versions["coacd"]}
    )
    mjcf = _fingerprint(
        {
            "prepare": prepare,
            "decompose": decompose,
            "obj2mjcf": asdict(settings.obj2mjcf),
            "obj2mjcf_version": versions["obj2mjcf"],
            "mujoco": asdict(settings.mujoco),
            "mujoco_version": versions["mujoco"],
        }
    )
    validate = _fingerprint({"mjcf": mjcf, "mujoco_version": versions["mujoco"], "steps": 10})
    return {"prepare": prepare, "decompose": decompose, "mjcf": mjcf, "validate": validate}


def _planned_transform(
    settings: Stage3Settings, input_reference: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema": "real2sim.physics_transform.v1",
        "input_mesh": input_reference["mesh"],
        "scale_mode": settings.geometry.scale_mode,
        "target_extent_m": settings.geometry.target_extent_m,
        "dry_run": True,
    }


def _load_or_create_manifest(settings: Stage3Settings) -> dict[str, Any]:
    path = settings.output_dir / "manifest.json"
    if path.is_file():
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "real2sim.physics.v1":
            raise ValueError(f"Unsupported Stage 3 manifest schema: {payload.get('schema')}")
        return payload
    return {
        "schema": "real2sim.physics.v1",
        "run_id": settings.output_dir.name,
        "created_at": _now(),
        "stage": "initialized",
        "settings": settings.to_dict(),
        "input_stage2": {},
        "versions": {},
        "signatures": {},
        "artifacts": {},
        "logs": {},
        "commands": [],
    }


def _can_reuse_prepare(
    manifest: dict[str, Any], signature: str, obj: Path, mtl: Path, transform: Path
) -> bool:
    if manifest.get("signatures", {}).get("prepare") != signature:
        return False
    if not obj.is_file() or not mtl.is_file() or not transform.is_file():
        return False
    read_obj_stats(obj)
    json.loads(transform.read_text(encoding="utf-8"))
    return True


def _can_reuse_collision(
    manifest: dict[str, Any], signature: str, collision_dir: Path, maximum: int
) -> bool:
    if manifest.get("signatures", {}).get("decompose") != signature:
        return False
    try:
        stats = collect_collision_stats(collision_dir)
    except (OSError, ValueError):
        return False
    count = int(stats["count"])
    return 0 < count <= maximum


def _can_reuse_mjcf(manifest: dict[str, Any], signature: str, xml_path: Path) -> bool:
    if manifest.get("signatures", {}).get("mjcf") != signature or not xml_path.is_file():
        return False
    try:
        validate_mjcf_structure(xml_path)
    except (OSError, ValueError, ET.ParseError):
        return False
    return True


def _can_reuse_validation(
    manifest: dict[str, Any], signature: str, report_path: Path
) -> bool:
    if manifest.get("signatures", {}).get("validate") != signature or not report_path.is_file():
        return False
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return payload.get("success") is True and payload.get("steps", 0) >= 10


def _require_signature(
    manifest: dict[str, Any], stage: str, expected: str, artifact: Path
) -> None:
    if manifest.get("signatures", {}).get(stage) != expected or not artifact.exists():
        raise ValueError(f"Stage 3 {stage} output is missing or stale; run stage={stage} first")


def _invalidate_after(manifest: dict[str, Any], stage: str) -> None:
    order = ["prepare", "decompose", "mjcf", "validate"]
    index = order.index(stage)
    for later in order[index + 1 :]:
        manifest.get("signatures", {}).pop(later, None)
    if stage in {"prepare", "decompose"}:
        manifest.pop("mjcf", None)
        manifest.pop("validation", None)
    elif stage == "mjcf":
        manifest.pop("validation", None)


def _record_command(
    manifest: dict[str, Any],
    stage: str,
    command: list[str],
    log_path: Path,
    result: CommandResult | None,
    reused: bool,
) -> None:
    manifest["commands"].append(
        {
            "stage": stage,
            "command": command,
            "log": str(log_path.resolve()),
            "dry_run": bool(result.dry_run) if result is not None else False,
            "reused": reused,
        }
    )


def _timed_step(
    trace: list[dict[str, Any]], name: str, action: Callable[[], Any]
) -> tuple[Any, dict[str, Any]]:
    started = time.perf_counter()
    event: dict[str, Any] = {"stage": name, "status": "running", "started_at": _now()}
    trace.append(event)
    try:
        result = action()
    except Exception as exc:
        event.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "duration_s": round(time.perf_counter() - started, 3),
                "ended_at": _now(),
            }
        )
        raise
    event.update(
        {
            "status": "completed",
            "duration_s": round(time.perf_counter() - started, 3),
            "ended_at": _now(),
            "reused": False,
        }
    )
    return result, event


def _append_completed_event(
    trace: list[dict[str, Any]], name: str, *, reused: bool
) -> None:
    trace.append(
        {
            "stage": name,
            "status": "completed",
            "duration_s": 0.0,
            "started_at": _now(),
            "ended_at": _now(),
            "reused": reused,
        }
    )


def _fingerprint(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _now() -> str:
    return datetime.now(UTC).isoformat()
