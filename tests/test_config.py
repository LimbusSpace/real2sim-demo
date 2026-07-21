from __future__ import annotations

from pathlib import Path

import pytest

from real2sim_demo.config import Stage1Settings


def test_example_config_resolves_environment_and_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets = tmp_path / "real2sim-assets"
    video = tmp_path / "captures" / "tabletop.mp4"
    colmap = tmp_path / "tools" / "colmap"
    gaussian_python = tmp_path / "envs" / "hyworld" / "python"
    monkeypatch.setenv("REAL2SIM_ASSETS", str(assets))
    monkeypatch.setenv("REAL2SIM_VIDEO", str(video))
    monkeypatch.setenv("REAL2SIM_COLMAP", str(colmap))
    monkeypatch.setenv("REAL2SIM_GAUSSIAN_PYTHON", str(gaussian_python))
    config = Path("configs/stage1.windows.example.toml")
    settings = Stage1Settings.from_toml(config).resolved(Path.cwd())

    assert settings.video.fps == 2.0
    assert settings.video.path == video
    assert settings.colmap.matcher == "sequential"
    assert Path(settings.gaussian.torch_home) == assets / "weights" / "torch"
    assert Path(settings.gaussian.launcher).is_absolute()
    assert settings.gaussian.source_revision == "7f668e67c74338d50684e57be46a438459b6bbe1"
    assert settings.gaussian.data_factor == 1
    assert settings.run_dir.is_absolute()


def test_config_reports_unset_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("REAL2SIM_MISSING", raising=False)
    config = tmp_path / "config.toml"
    config.write_text('[video]\npath = "${REAL2SIM_MISSING}/video.mp4"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="REAL2SIM_MISSING"):
        Stage1Settings.from_toml(config)
