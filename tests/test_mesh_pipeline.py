from __future__ import annotations

import json
from pathlib import Path

import pytest

from real2sim_demo.mesh_config import Stage2Settings, TwoDGSSettings
from real2sim_demo.mesh_pipeline import run_stage2


def test_stage2_dry_run_records_prepare_train_mesh_preview_order(tmp_path: Path) -> None:
    settings = Stage2Settings(
        input_run_dir=tmp_path / "missing-stage1",
        output_dir=tmp_path / "stage2",
        twodgs=TwoDGSSettings(
            python="python",
            root=tmp_path / "2dgs",
            iterations=30_000,
        ),
    )

    manifest = run_stage2(settings, dry_run=True)

    assert manifest["stage"] == "commands_planned"
    assert manifest["dry_run"] is True
    trace = json.loads((settings.output_dir / "trace.json").read_text(encoding="utf-8"))
    assert [event["stage"] for event in trace["events"]] == [
        "prepare",
        "train",
        "mesh",
        "preview",
        "diagnostics",
    ]
    assert all(event["status"] == "completed" for event in trace["events"])
    assert [command["stage"] for command in manifest["commands"]] == [
        "train",
        "mesh",
        "preview",
        "diagnostics_render",
        "diagnostics_triptych",
    ]
    train_command = manifest["commands"][0]["command"]
    mesh_command = manifest["commands"][1]["command"]
    assert "--lambda_dist" in train_command
    assert "100.0" in train_command
    assert "--eval" not in train_command
    assert "--skip_train" in mesh_command
    assert "--skip_test" in mesh_command
    assert "--unbounded" not in mesh_command
    assert (settings.output_dir / "logs" / "10_mesh_preview.log").is_file()
    assert (settings.output_dir / "logs" / "11_2dgs_diagnostics_render.log").is_file()
    assert (settings.output_dir / "logs" / "12_2dgs_diagnostics_triptych.log").is_file()
    diagnostics = manifest["artifacts"]["diagnostics"]
    assert diagnostics["frame_count"] is None
    report_path = Path(diagnostics["report"])
    assert report_path.name == "report.json"
    assert report_path.parent.name == "diagnostics"


def test_stage2_failure_is_written_to_manifest(tmp_path: Path) -> None:
    input_dir = tmp_path / "stage1"
    input_dir.mkdir()
    (input_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "real2sim.stage1.manifest.v1",
                "run_id": "bad",
                "stage": "failed",
            }
        ),
        encoding="utf-8",
    )
    settings = Stage2Settings(
        input_run_dir=input_dir,
        output_dir=tmp_path / "stage2",
        twodgs=TwoDGSSettings(python="python", root=tmp_path / "2dgs"),
    )

    with pytest.raises(ValueError, match="prepared or gaussian_trained"):
        run_stage2(settings, stage="prepare")

    manifest = json.loads((settings.output_dir / "manifest.json").read_text(encoding="utf-8"))
    trace = json.loads((settings.output_dir / "trace.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "failed"
    assert "ValueError" in manifest["error"]
    assert trace["events"][0]["status"] == "failed"
