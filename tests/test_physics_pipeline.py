from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import NoReturn

import numpy as np
import pytest
import trimesh

import real2sim_demo.physics_pipeline as physics_pipeline
from real2sim_demo.physics_config import (
    CoacdSettings,
    GeometrySettings,
    Stage3Settings,
)
from real2sim_demo.physics_pipeline import run_stage3


def test_stage3_rejects_invalid_stage2_state_and_records_failure(tmp_path: Path) -> None:
    input_dir = tmp_path / "stage2"
    input_dir.mkdir()
    (input_dir / "manifest.json").write_text(
        json.dumps({"schema": "real2sim.mesh.v1", "stage": "failed"}),
        encoding="utf-8",
    )
    settings = Stage3Settings(input_mesh_run_dir=input_dir, output_dir=tmp_path / "stage3")

    with pytest.raises(ValueError, match="mesh_exported"):
        run_stage3(settings, stage="prepare")

    manifest = json.loads((settings.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "failed"
    assert "ValueError" in manifest["error"]


def test_stage3_rejects_invalid_stage2_schema(tmp_path: Path) -> None:
    input_dir = tmp_path / "stage2"
    input_dir.mkdir()
    (input_dir / "manifest.json").write_text(
        json.dumps({"schema": "real2sim.mesh.v0", "stage": "mesh_exported"}),
        encoding="utf-8",
    )
    settings = Stage3Settings(input_mesh_run_dir=input_dir, output_dir=tmp_path / "stage3")

    with pytest.raises(ValueError, match="manifest schema"):
        run_stage3(settings, stage="prepare")


def test_stage3_records_coacd_command_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Stage3Settings(
        input_mesh_run_dir=_stage2_cube(tmp_path),
        output_dir=tmp_path / "stage3",
        geometry=GeometrySettings(
            plane_threshold_m=0.01,
            plane_min_inlier_fraction=0.2,
            visual_face_count=100,
        ),
    )
    run_stage3(settings, stage="prepare")

    def fail_command(*args: object, **kwargs: object) -> NoReturn:
        raise subprocess.CalledProcessError(2, ["coacd"])

    monkeypatch.setattr(physics_pipeline, "run_command", fail_command)
    with pytest.raises(subprocess.CalledProcessError):
        run_stage3(settings, stage="decompose")

    manifest = json.loads((settings.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "failed"
    assert "CalledProcessError" in manifest["error"]


def test_stage3_dry_run_trace_has_required_order(tmp_path: Path) -> None:
    settings = Stage3Settings(
        input_mesh_run_dir=tmp_path / "missing-stage2",
        output_dir=tmp_path / "stage3",
    )

    manifest = run_stage3(settings, dry_run=True)

    trace = json.loads((settings.output_dir / "trace.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "commands_planned"
    assert [event["stage"] for event in trace["events"]] == [
        "prepare",
        "decompose",
        "mjcf",
        "validate",
    ]


def test_stage3_cpu_integration_runs_coacd_obj2mjcf_and_mujoco(tmp_path: Path) -> None:
    input_dir = _stage2_cube(tmp_path)
    settings = Stage3Settings(
        input_mesh_run_dir=input_dir,
        output_dir=tmp_path / "stage3",
        geometry=GeometrySettings(
            plane_threshold_m=0.01,
            plane_min_inlier_fraction=0.2,
            visual_face_count=100,
        ),
        coacd=CoacdSettings(
            threshold_m=0.05,
            max_convex_hulls=4,
            preprocess_resolution=20,
            resolution=1_000,
            mcts_nodes=10,
            mcts_iterations=60,
            mcts_max_depth=2,
            max_ch_vertex=64,
        ),
    )

    manifest = run_stage3(settings)

    assert manifest["stage"] == "validated"
    assert manifest["collision"]["count"] >= 1
    assert manifest["collision"]["count"] <= 4
    assert manifest["validation"]["success"] is True
    assert manifest["validation"]["model"]["njnt"] == 0
    assert (settings.output_dir / "mjcf" / "scene.xml").is_file()
    assert (settings.output_dir / "mjcf" / "scene.obj").is_file()
    assert (settings.output_dir / "mjcf" / "scene.mtl").is_file()

    reused = run_stage3(settings)
    trace = json.loads((settings.output_dir / "trace.json").read_text(encoding="utf-8"))
    assert reused["stage"] == "validated"
    assert all(event["reused"] for event in trace["events"])


def _stage2_cube(tmp_path: Path) -> Path:
    run_dir = tmp_path / "stage2"
    mesh_dir = run_dir / "mesh"
    mesh_dir.mkdir(parents=True)
    mesh = trimesh.creation.box(extents=[2.0, 1.0, 0.5])
    mesh.visual.vertex_colors = np.tile([100, 140, 180, 255], (len(mesh.vertices), 1))
    mesh.export(mesh_dir / "post.ply")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "real2sim.mesh.v1",
                "run_id": "fixture",
                "stage": "mesh_exported",
                "upstream": {"2dgs": "fixture"},
            }
        ),
        encoding="utf-8",
    )
    return run_dir
