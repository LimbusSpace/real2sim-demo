from __future__ import annotations

from pathlib import Path

from real2sim_demo.mesh import build_mesh_command, build_train_command
from real2sim_demo.mesh_config import TwoDGSSettings


def test_mesh_command_adds_only_explicit_scale_overrides(tmp_path: Path) -> None:
    settings = TwoDGSSettings(python="python", root=tmp_path / "2dgs")

    automatic = build_mesh_command(
        settings,
        tmp_path / "dataset",
        tmp_path / "model",
        mesh_res=1024,
        num_clusters=50,
        depth_trunc=None,
        voxel_size=None,
        sdf_trunc=None,
    )
    explicit = build_mesh_command(
        settings,
        tmp_path / "dataset",
        tmp_path / "model",
        mesh_res=512,
        num_clusters=20,
        depth_trunc=3.0,
        voxel_size=0.004,
        sdf_trunc=0.02,
    )

    assert "--depth_trunc" not in automatic
    assert "--voxel_size" not in automatic
    assert "--sdf_trunc" not in automatic
    assert explicit[-6:] == [
        "--depth_trunc",
        "3.0",
        "--voxel_size",
        "0.004",
        "--sdf_trunc",
        "0.02",
    ]


def test_train_command_uses_all_views_without_eval_holdout(tmp_path: Path) -> None:
    settings = TwoDGSSettings(python="python", root=tmp_path / "2dgs")

    command = build_train_command(settings, tmp_path / "dataset", tmp_path / "model")

    assert "--eval" not in command
    assert command[command.index("--iterations") + 1] == "30000"
    assert command[command.index("--sh_degree") + 1] == "3"
