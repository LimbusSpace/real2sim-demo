from __future__ import annotations

from pathlib import Path

import pytest

from real2sim_demo.physics_collision import build_coacd_command, read_obj_stats
from real2sim_demo.physics_config import CoacdSettings


def test_coacd_command_contains_metric_and_deterministic_parameters(tmp_path: Path) -> None:
    command = build_coacd_command(
        "python",
        tmp_path / "scene.obj",
        tmp_path / "collision",
        CoacdSettings(),
    )

    assert "--real-metric" in command
    assert "--decimate" in command
    assert command[command.index("--threshold") + 1] == "0.005"
    assert command[command.index("--max-convex-hull") + 1] == "64"
    assert command[command.index("--seed") + 1] == "0"


def test_obj_stats_rejects_invalid_indices(tmp_path: Path) -> None:
    path = tmp_path / "bad.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 4\n", encoding="ascii")

    with pytest.raises(ValueError, match="out of range"):
        read_obj_stats(path)


def test_obj_stats_rejects_non_triangular_faces(tmp_path: Path) -> None:
    path = tmp_path / "quad.obj"
    path.write_text(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n", encoding="ascii"
    )

    with pytest.raises(ValueError, match="Non-triangular"):
        read_obj_stats(path)
