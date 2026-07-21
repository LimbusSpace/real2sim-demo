from __future__ import annotations

import json
from pathlib import Path

from real2sim_demo.config import (
    ColmapSettings,
    GaussianSettings,
    Stage1Settings,
    VideoSettings,
)
from real2sim_demo.pipeline import run_stage1


def test_stage1_dry_run_records_planned_commands(tmp_path: Path) -> None:
    settings = Stage1Settings(
        video=VideoSettings(path=tmp_path / "missing.mp4"),
        colmap=ColmapSettings(),
        gaussian=GaussianSettings(
            python="python",
            launcher=str(tmp_path / "trainer.py"),
            torch_home=str(tmp_path / "torch"),
            max_steps=10,
            data_factor=2,
        ),
        run_dir=tmp_path / "run",
    )

    manifest = run_stage1(settings, dry_run=True)

    assert manifest["stage"] == "commands_planned"
    assert manifest["dry_run"] is True
    trace = json.loads((settings.run_dir / "trace.json").read_text(encoding="utf-8"))
    assert [event["stage"] for event in trace["events"]] == [
        "extract_frames",
        "colmap_reconstruction",
        "gaussian_training",
    ]
    assert all(event["status"] == "completed" for event in trace["events"])
    assert (settings.run_dir / "logs/07_hyworld_train.log").is_file()
