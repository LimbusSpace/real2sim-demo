from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ENVIRONMENT_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True, slots=True)
class VideoSettings:
    path: Path
    fps: float = 2.0
    ffmpeg: str = "ffmpeg"


@dataclass(frozen=True, slots=True)
class ColmapSettings:
    executable: str = "colmap"
    camera_model: str = "SIMPLE_RADIAL"
    matcher: str = "sequential"
    use_gpu: bool = True
    sequential_overlap: int = 10


@dataclass(frozen=True, slots=True)
class GaussianSettings:
    python: str = ""
    launcher: str = ""
    torch_home: str = ""
    source_revision: str = ""
    max_steps: int = 5_000
    data_factor: int = 1
    sh_degree: int = 1
    test_every: int = 8
    disable_video: bool = True


@dataclass(frozen=True, slots=True)
class Stage1Settings:
    video: VideoSettings
    colmap: ColmapSettings = field(default_factory=ColmapSettings)
    gaussian: GaussianSettings = field(default_factory=GaussianSettings)
    run_dir: Path = Path("artifacts/runs/stage1")

    @classmethod
    def from_toml(cls, path: Path) -> "Stage1Settings":
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        video = payload.get("video", {})
        colmap = payload.get("colmap", {})
        gaussian = payload.get("gaussian", {})
        return cls(
            video=VideoSettings(
                path=Path(_expand_environment(str(video.get("path", "data/input/video.mp4")))),
                fps=float(video.get("fps", 2.0)),
                ffmpeg=_expand_environment(str(video.get("ffmpeg", "ffmpeg"))),
            ),
            colmap=ColmapSettings(
                executable=_expand_environment(str(colmap.get("executable", "colmap"))),
                camera_model=str(colmap.get("camera_model", "SIMPLE_RADIAL")),
                matcher=str(colmap.get("matcher", "sequential")),
                use_gpu=bool(colmap.get("use_gpu", True)),
                sequential_overlap=int(colmap.get("sequential_overlap", 10)),
            ),
            gaussian=GaussianSettings(
                python=_expand_environment(str(gaussian.get("python", ""))),
                launcher=_expand_environment(str(gaussian.get("launcher", ""))),
                torch_home=_expand_environment(str(gaussian.get("torch_home", ""))),
                source_revision=str(gaussian.get("source_revision", "")),
                max_steps=int(gaussian.get("max_steps", 5_000)),
                data_factor=int(gaussian.get("data_factor", 1)),
                sh_degree=int(gaussian.get("sh_degree", 1)),
                test_every=int(gaussian.get("test_every", 8)),
                disable_video=bool(gaussian.get("disable_video", True)),
            ),
            run_dir=Path(
                _expand_environment(str(payload.get("run_dir", "artifacts/runs/stage1")))
            ),
        )

    def resolved(
        self,
        repo_root: Path,
        *,
        video: Path | None = None,
        run_dir: Path | None = None,
    ) -> "Stage1Settings":
        selected_video = video if video is not None else self.video.path
        selected_run = run_dir if run_dir is not None else self.run_dir
        return Stage1Settings(
            video=VideoSettings(
                path=_resolve_path(repo_root, selected_video),
                fps=self.video.fps,
                ffmpeg=self.video.ffmpeg,
            ),
            colmap=self.colmap,
            gaussian=GaussianSettings(
                python=_resolve_optional_path(repo_root, self.gaussian.python),
                launcher=_resolve_optional_path(repo_root, self.gaussian.launcher),
                torch_home=_resolve_optional_path(repo_root, self.gaussian.torch_home),
                source_revision=self.gaussian.source_revision,
                max_steps=self.gaussian.max_steps,
                data_factor=self.gaussian.data_factor,
                sh_degree=self.gaussian.sh_degree,
                test_every=self.gaussian.test_every,
                disable_video=self.gaussian.disable_video,
            ),
            run_dir=_resolve_path(repo_root, selected_run),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": {
                "path": str(self.video.path),
                "fps": self.video.fps,
                "ffmpeg": self.video.ffmpeg,
            },
            "colmap": {
                "executable": self.colmap.executable,
                "camera_model": self.colmap.camera_model,
                "matcher": self.colmap.matcher,
                "use_gpu": self.colmap.use_gpu,
                "sequential_overlap": self.colmap.sequential_overlap,
            },
            "gaussian": {
                "python": self.gaussian.python,
                "launcher": self.gaussian.launcher,
                "torch_home": self.gaussian.torch_home,
                "source_revision": self.gaussian.source_revision,
                "max_steps": self.gaussian.max_steps,
                "data_factor": self.gaussian.data_factor,
                "sh_degree": self.gaussian.sh_degree,
                "test_every": self.gaussian.test_every,
                "disable_video": self.gaussian.disable_video,
            },
            "run_dir": str(self.run_dir),
        }


def _resolve_path(repo_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else (repo_root / value).resolve()


def _resolve_optional_path(repo_root: Path, value: str) -> str:
    if not value:
        return value
    path = _resolve_path(repo_root, Path(value))
    return str(path)


def _expand_environment(value: str) -> str:
    names = set(_ENVIRONMENT_VARIABLE.findall(value))
    missing = sorted(name for name in names if name not in os.environ)
    if missing:
        raise ValueError("Unset environment variable(s): " + ", ".join(missing))
    return _ENVIRONMENT_VARIABLE.sub(lambda match: os.environ[match.group(1)], value)
