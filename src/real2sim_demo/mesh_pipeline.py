from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from .mesh import (
    MeshArtifacts,
    create_diagnostics,
    create_preview,
    export_mesh,
    source_versions,
    train_2dgs,
)
from .mesh_config import Stage2Settings
from .mesh_dataset import prepare_2dgs_dataset, validate_prepared_dataset


def run_stage2(
    settings: Stage2Settings,
    *,
    stage: str = "all",
    dry_run: bool = False,
) -> dict[str, Any]:
    if stage not in {"prepare", "train", "mesh", "all"}:
        raise ValueError(f"Unsupported Stage 2 stage: {stage}")
    settings.validate()
    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_or_create_manifest(settings, dry_run=dry_run)
    manifest.pop("error", None)
    manifest["settings"] = settings.to_dict()
    manifest["upstream"] = source_versions(
        settings.twodgs.root, settings.twodgs.source_revision, dry_run=True
    )
    manifest.setdefault("commands", [])
    manifest.setdefault("artifacts", {})
    trace: list[dict[str, Any]] = []
    dataset_dir = output_dir / "dataset"
    model_dir = output_dir / "2dgs"
    mesh_dir = output_dir / "mesh"
    logs_dir = output_dir / "logs"

    try:
        if stage in {"prepare", "all"}:
            if dry_run:
                provenance: dict[str, Any] = {
                    "schema": "real2sim.2dgs_dataset.v1",
                    "dataset": str(dataset_dir.resolve()),
                    "registered_image_count": None,
                    "dry_run": True,
                }
            else:
                provenance, _ = _timed_step(
                    trace,
                    "prepare",
                    lambda: prepare_2dgs_dataset(settings.input_run_dir, dataset_dir),
                )
            if dry_run:
                _append_completed_event(trace, "prepare")
            manifest["input_stage1"] = _stage1_reference(settings.input_run_dir, dry_run=dry_run)
            manifest["artifacts"].update(
                {
                    "dataset": str(dataset_dir.resolve()),
                    "registered_image_count": provenance["registered_image_count"],
                }
            )
            manifest["stage"] = "commands_planned" if dry_run else "prepared"

        if stage in {"train", "all"}:
            if not dry_run:
                validate_prepared_dataset(dataset_dir)
            result, _ = _timed_step(
                trace,
                "train",
                lambda: train_2dgs(
                    settings.twodgs,
                    dataset_dir,
                    model_dir,
                    logs_dir,
                    dry_run=dry_run,
                ),
            )
            ply, command, reused = result
            _record_command(manifest, command, "train", reused=reused)
            if ply is not None:
                manifest["artifacts"]["surfel_ply"] = str(ply.resolve())
            manifest["training"] = {
                "backend": "2dgs",
                "iterations": settings.twodgs.iterations,
                "reused": reused,
            }
            manifest["stage"] = "commands_planned" if dry_run else "2dgs_trained"

        if stage in {"mesh", "all"}:
            surfel = (
                model_dir
                / "point_cloud"
                / f"iteration_{settings.twodgs.iterations}"
                / "point_cloud.ply"
            )
            if not dry_run and not surfel.is_file():
                raise FileNotFoundError(f"Missing {surfel}; run stage=train first")
            result, _ = _timed_step(
                trace,
                "mesh",
                lambda: export_mesh(
                    settings.twodgs,
                    dataset_dir,
                    model_dir,
                    mesh_dir,
                    logs_dir,
                    mesh_res=settings.mesh.mesh_res,
                    num_clusters=settings.mesh.num_clusters,
                    depth_trunc=settings.mesh.depth_trunc,
                    voxel_size=settings.mesh.voxel_size,
                    sdf_trunc=settings.mesh.sdf_trunc,
                    dry_run=dry_run,
                ),
            )
            artifacts, command, reused = result
            _record_command(manifest, command, "mesh", reused=reused)
            if artifacts is not None:
                _record_mesh_artifacts(manifest, artifacts)
            preview_script = (
                Path(__file__).resolve().parents[2] / "scripts" / "render_mesh_preview.py"
            )
            preview_result, _ = _timed_step(
                trace,
                "preview",
                lambda: create_preview(
                    settings.twodgs.python,
                    preview_script,
                    mesh_dir / "post.ply",
                    mesh_dir / "preview.png",
                    logs_dir,
                    allow_reuse=reused,
                    dry_run=dry_run,
                ),
            )
            preview_command, preview_reused = preview_result
            _record_command(manifest, preview_command, "preview", reused=preview_reused)
            manifest["artifacts"]["preview"] = str((mesh_dir / "preview.png").resolve())
            diagnostics_dir = output_dir / "diagnostics"
            diagnostics_result, _ = _timed_step(
                trace,
                "diagnostics",
                lambda: create_diagnostics(
                    settings.twodgs,
                    dataset_dir,
                    model_dir,
                    diagnostics_dir,
                    logs_dir,
                    composer_python=sys.executable,
                    dry_run=dry_run,
                ),
            )
            report, render_command, triptych_command, diagnostics_reused = diagnostics_result
            _record_command(
                manifest,
                render_command,
                "diagnostics_render",
                reused=diagnostics_reused,
            )
            _record_command(
                manifest,
                triptych_command,
                "diagnostics_triptych",
                reused=diagnostics_reused,
            )
            diagnostics_artifacts: dict[str, Any] = {
                "report": str((diagnostics_dir / "report.json").resolve()),
                "contact_sheet": str((diagnostics_dir / "contact_sheet.jpg").resolve()),
                "triptych_dir": str((diagnostics_dir / "triptychs").resolve()),
                "frame_count": None,
            }
            if report is not None:
                diagnostics_artifacts.update(
                    {
                        "frame_count": report["frame_count"],
                        "depth_percentile_02": report["depth_percentile_02"],
                        "depth_percentile_98": report["depth_percentile_98"],
                        "metrics": report["metrics"],
                    }
                )
            manifest["artifacts"]["diagnostics"] = diagnostics_artifacts
            manifest["stage"] = "commands_planned" if dry_run else "mesh_exported"
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
                "schema": "real2sim.mesh_provenance.v1",
                "input_stage1": manifest.get("input_stage1", {}),
                "settings": settings.to_dict(),
                "upstream": manifest["upstream"],
                "artifacts": manifest["artifacts"],
                "commands": manifest["commands"],
            },
        )
    return manifest


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
        }
    )
    return result, event


def _append_completed_event(trace: list[dict[str, Any]], name: str) -> None:
    trace.append(
        {
            "stage": name,
            "status": "completed",
            "duration_s": 0.0,
            "started_at": _now(),
            "ended_at": _now(),
        }
    )


def _record_command(manifest: dict[str, Any], command: Any, stage: str, *, reused: bool) -> None:
    record: dict[str, Any] = {"stage": stage, "reused": reused}
    if command is not None:
        record.update(
            {
                "command": command.command,
                "log": str(command.log_path.resolve()),
                "dry_run": command.dry_run,
            }
        )
    manifest["commands"].append(record)


def _record_mesh_artifacts(manifest: dict[str, Any], artifacts: MeshArtifacts) -> None:
    manifest["artifacts"].update(
        {
            "mesh_raw": str(artifacts.raw.resolve()),
            "mesh_post": str(artifacts.post.resolve()),
            "mesh_stats": {
                "raw": artifacts.raw_stats.to_dict(),
                "post": artifacts.post_stats.to_dict(),
            },
        }
    )


def _load_or_create_manifest(settings: Stage2Settings, *, dry_run: bool) -> dict[str, Any]:
    path = settings.output_dir / "manifest.json"
    if path.is_file():
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "real2sim.mesh.v1":
            raise ValueError(f"Unsupported Stage 2 manifest schema: {payload.get('schema')}")
        return payload
    return {
        "schema": "real2sim.mesh.v1",
        "run_id": settings.output_dir.name,
        "created_at": _now(),
        "stage": "initialized",
        "settings": settings.to_dict(),
        "input_stage1": _stage1_reference(settings.input_run_dir, dry_run=dry_run),
        "upstream": {},
        "artifacts": {},
        "commands": [],
    }


def _stage1_reference(input_run_dir: Path, *, dry_run: bool) -> dict[str, Any]:
    manifest_path = input_run_dir / "manifest.json"
    reference: dict[str, Any] = {"manifest": str(manifest_path.resolve())}
    if manifest_path.is_file() and not dry_run:
        payload: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        reference.update(
            {
                "sha256": _sha256(manifest_path),
                "schema": payload.get("schema"),
                "run_id": payload.get("run_id"),
                "stage": payload.get("stage"),
            }
        )
    return reference


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
