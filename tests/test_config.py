from __future__ import annotations

from pathlib import Path

import pytest

from real2sim_demo.config import Stage1Settings


def test_example_config_resolves_environment_and_relative_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REAL2SIM_ASSETS", "E:/real2sim-assets")
    monkeypatch.setenv("REAL2SIM_VIDEO", "E:/captures/tabletop.mp4")
    monkeypatch.setenv("REAL2SIM_COLMAP", "E:/tools/COLMAP.bat")
    monkeypatch.setenv("REAL2SIM_GAUSSIAN_PYTHON", "E:/envs/hyworld/python.exe")
    config = Path("configs/stage1.windows.example.toml")
    settings = Stage1Settings.from_toml(config).resolved(Path.cwd())

    assert settings.video.fps == 2.0
    assert settings.video.path == Path("E:/captures/tabletop.mp4")
    assert settings.colmap.matcher == "sequential"
    assert Path(settings.gaussian.torch_home) == Path("E:/real2sim-assets/weights/torch")
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
