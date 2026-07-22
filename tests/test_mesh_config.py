from __future__ import annotations

from pathlib import Path

import pytest

from real2sim_demo.mesh_config import TWODGS_REVISION, Stage2Settings


def test_stage2_config_expands_environment_and_keeps_optional_mesh_values_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASSETS", str(tmp_path / "assets"))
    monkeypatch.setenv("TWODGS_ROOT", str(tmp_path / "2dgs"))
    monkeypatch.setenv("TWODGS_PYTHON", str(tmp_path / "env" / "python.exe"))
    config = tmp_path / "stage2.toml"
    config.write_text(
        """
input_run_dir = "${ASSETS}/runs/input"
output_dir = "${ASSETS}/runs/mesh"
[twodgs]
python = "${TWODGS_PYTHON}"
root = "${TWODGS_ROOT}"
[mesh]
mesh_res = 512
num_clusters = 20
""",
        encoding="utf-8",
    )

    settings = Stage2Settings.from_toml(config).resolved(tmp_path)

    assert settings.input_run_dir == tmp_path / "assets" / "runs" / "input"
    assert settings.output_dir == tmp_path / "assets" / "runs" / "mesh"
    assert settings.twodgs.source_revision == TWODGS_REVISION
    assert settings.twodgs.iterations == 30_000
    assert settings.mesh.mesh_res == 512
    assert settings.mesh.depth_trunc is None
    assert settings.mesh.voxel_size is None
    assert settings.mesh.sdf_trunc is None


def test_stage2_config_accepts_explicit_mesh_scale_overrides(tmp_path: Path) -> None:
    config = tmp_path / "stage2.toml"
    config.write_text(
        """
input_run_dir = "input"
output_dir = "output"
[mesh]
depth_trunc = 3.0
voxel_size = 0.004
sdf_trunc = 0.02
""",
        encoding="utf-8",
    )

    settings = Stage2Settings.from_toml(config)

    assert settings.mesh.depth_trunc == 3.0
    assert settings.mesh.voxel_size == 0.004
    assert settings.mesh.sdf_trunc == 0.02


def test_stage2_config_rejects_unbounded_mode(tmp_path: Path) -> None:
    config = tmp_path / "stage2.toml"
    config.write_text(
        'input_run_dir = "input"\noutput_dir = "output"\n[mesh]\nmode = "unbounded"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="bounded"):
        Stage2Settings.from_toml(config)
