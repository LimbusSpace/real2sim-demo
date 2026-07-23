from __future__ import annotations

from pathlib import Path

import pytest

from real2sim_demo.physics_config import Stage3Settings


def test_stage3_config_expands_environment_and_resolves_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = tmp_path / "assets"
    monkeypatch.setenv("REAL2SIM_ASSETS", str(assets))
    config = tmp_path / "stage3.toml"
    config.write_text(
        '\n'.join(
            [
                'input_mesh_run_dir = "${REAL2SIM_ASSETS}/runs/mesh"',
                'output_dir = "${REAL2SIM_ASSETS}/runs/physics"',
                "[coacd]",
                'version = "1.0.11"',
                "[obj2mjcf]",
                'version = "0.0.25"',
                "[mujoco]",
                'version = "3.10.0"',
            ]
        ),
        encoding="utf-8",
    )

    settings = Stage3Settings.from_toml(config).resolved(tmp_path)

    assert settings.input_mesh_run_dir == assets / "runs" / "mesh"
    assert settings.output_dir == assets / "runs" / "physics"
    assert settings.geometry.target_extent_m == 1.0
    assert settings.mujoco.gravity == (0.0, 0.0, -9.81)


def test_stage3_config_rejects_unpinned_versions(tmp_path: Path) -> None:
    config = tmp_path / "stage3.toml"
    config.write_text(
        '\n'.join(
            [
                'input_mesh_run_dir = "mesh"',
                'output_dir = "physics"',
                "[coacd]",
                'version = "latest"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="coacd.version"):
        Stage3Settings.from_toml(config)
