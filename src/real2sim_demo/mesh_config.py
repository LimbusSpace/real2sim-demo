from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import _expand_environment, _resolve_optional_path, _resolve_path

TWODGS_REVISION = "335ad612f2e783a4e57b9cbc4d1e167bd599fc98"
DIFF_SURFEL_REVISION = "e0ed0207b3e0669960cfad70852200a4a5847f61"
SIMPLE_KNN_REVISION = "f155ec04131cb579f53443a06879d37115f4612f"


@dataclass(frozen=True, slots=True)
class TwoDGSSettings:
    python: str = "python"
    root: Path = Path("third_party/2d-gaussian-splatting")
    source_revision: str = TWODGS_REVISION
    iterations: int = 30_000
    resolution: int = 1
    sh_degree: int = 3
    depth_ratio: float = 0.0
    lambda_normal: float = 0.05
    lambda_dist: float = 100.0


@dataclass(frozen=True, slots=True)
class MeshSettings:
    mode: str = "bounded"
    mesh_res: int = 1024
    num_clusters: int = 50
    depth_trunc: float | None = None
    voxel_size: float | None = None
    sdf_trunc: float | None = None


@dataclass(frozen=True, slots=True)
class Stage2Settings:
    input_run_dir: Path
    output_dir: Path
    twodgs: TwoDGSSettings = field(default_factory=TwoDGSSettings)
    mesh: MeshSettings = field(default_factory=MeshSettings)

    @classmethod
    def from_toml(cls, path: Path) -> "Stage2Settings":
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        twodgs = payload.get("twodgs", {})
        mesh = payload.get("mesh", {})
        settings = cls(
            input_run_dir=Path(
                _expand_environment(str(payload.get("input_run_dir", "artifacts/runs/stage1")))
            ),
            output_dir=Path(
                _expand_environment(str(payload.get("output_dir", "artifacts/runs/stage2")))
            ),
            twodgs=TwoDGSSettings(
                python=_expand_environment(str(twodgs.get("python", "python"))),
                root=Path(
                    _expand_environment(
                        str(twodgs.get("root", "third_party/2d-gaussian-splatting"))
                    )
                ),
                source_revision=str(twodgs.get("source_revision", TWODGS_REVISION)),
                iterations=int(twodgs.get("iterations", 30_000)),
                resolution=int(twodgs.get("resolution", 1)),
                sh_degree=int(twodgs.get("sh_degree", 3)),
                depth_ratio=float(twodgs.get("depth_ratio", 0.0)),
                lambda_normal=float(twodgs.get("lambda_normal", 0.05)),
                lambda_dist=float(twodgs.get("lambda_dist", 100.0)),
            ),
            mesh=MeshSettings(
                mode=str(mesh.get("mode", "bounded")),
                mesh_res=int(mesh.get("mesh_res", 1024)),
                num_clusters=int(mesh.get("num_clusters", 50)),
                depth_trunc=_optional_float(mesh, "depth_trunc"),
                voxel_size=_optional_float(mesh, "voxel_size"),
                sdf_trunc=_optional_float(mesh, "sdf_trunc"),
            ),
        )
        settings.validate()
        return settings

    def resolved(
        self,
        repo_root: Path,
        *,
        input_run_dir: Path | None = None,
        output_dir: Path | None = None,
    ) -> "Stage2Settings":
        return Stage2Settings(
            input_run_dir=_resolve_path(
                repo_root, input_run_dir if input_run_dir is not None else self.input_run_dir
            ),
            output_dir=_resolve_path(
                repo_root, output_dir if output_dir is not None else self.output_dir
            ),
            twodgs=TwoDGSSettings(
                python=_resolve_optional_path(repo_root, self.twodgs.python),
                root=_resolve_path(repo_root, self.twodgs.root),
                source_revision=self.twodgs.source_revision,
                iterations=self.twodgs.iterations,
                resolution=self.twodgs.resolution,
                sh_degree=self.twodgs.sh_degree,
                depth_ratio=self.twodgs.depth_ratio,
                lambda_normal=self.twodgs.lambda_normal,
                lambda_dist=self.twodgs.lambda_dist,
            ),
            mesh=self.mesh,
        )

    def validate(self) -> None:
        if self.twodgs.source_revision != TWODGS_REVISION:
            raise ValueError(
                f"twodgs.source_revision must be the pinned revision {TWODGS_REVISION}"
            )
        if self.twodgs.iterations <= 0:
            raise ValueError("twodgs.iterations must be positive")
        if self.twodgs.resolution <= 0:
            raise ValueError("twodgs.resolution must be positive")
        if self.twodgs.sh_degree < 0:
            raise ValueError("twodgs.sh_degree must be non-negative")
        if self.mesh.mode != "bounded":
            raise ValueError("Only mesh.mode='bounded' is supported")
        if self.mesh.mesh_res <= 0:
            raise ValueError("mesh.mesh_res must be positive")
        if self.mesh.num_clusters <= 0:
            raise ValueError("mesh.num_clusters must be positive")
        for name in ("depth_trunc", "voxel_size", "sdf_trunc"):
            value = getattr(self.mesh, name)
            if value is not None and value <= 0:
                raise ValueError(f"mesh.{name} must be positive when provided")

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_run_dir": str(self.input_run_dir),
            "output_dir": str(self.output_dir),
            "twodgs": {
                "python": self.twodgs.python,
                "root": str(self.twodgs.root),
                "source_revision": self.twodgs.source_revision,
                "iterations": self.twodgs.iterations,
                "resolution": self.twodgs.resolution,
                "sh_degree": self.twodgs.sh_degree,
                "depth_ratio": self.twodgs.depth_ratio,
                "lambda_normal": self.twodgs.lambda_normal,
                "lambda_dist": self.twodgs.lambda_dist,
            },
            "mesh": {
                "mode": self.mesh.mode,
                "mesh_res": self.mesh.mesh_res,
                "num_clusters": self.mesh.num_clusters,
                "depth_trunc": self.mesh.depth_trunc,
                "voxel_size": self.mesh.voxel_size,
                "sdf_trunc": self.mesh.sdf_trunc,
            },
        }


def _optional_float(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    return None if value is None else float(value)
