from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import _expand_environment, _resolve_path

COACD_VERSION = "1.0.11"
OBJ2MJCF_VERSION = "0.0.25"
MUJOCO_VERSION = "3.10.0"


@dataclass(frozen=True, slots=True)
class GeometrySettings:
    scale_mode: str = "normalize_max_extent"
    target_extent_m: float = 1.0
    align_support_plane: bool = True
    plane_threshold_m: float = 0.005
    plane_min_inlier_fraction: float = 0.05
    visual_face_count: int = 250_000


@dataclass(frozen=True, slots=True)
class CoacdSettings:
    version: str = COACD_VERSION
    threshold_m: float = 0.005
    real_metric: bool = True
    max_convex_hulls: int = 64
    preprocess_mode: str = "auto"
    preprocess_resolution: int = 50
    resolution: int = 2_000
    mcts_nodes: int = 20
    mcts_iterations: int = 150
    mcts_max_depth: int = 3
    decimate: bool = True
    max_ch_vertex: int = 64
    seed: int = 0


@dataclass(frozen=True, slots=True)
class Obj2MjcfSettings:
    version: str = OBJ2MJCF_VERSION


@dataclass(frozen=True, slots=True)
class MujocoSettings:
    version: str = MUJOCO_VERSION
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)


@dataclass(frozen=True, slots=True)
class Stage3Settings:
    input_mesh_run_dir: Path
    output_dir: Path
    geometry: GeometrySettings = field(default_factory=GeometrySettings)
    coacd: CoacdSettings = field(default_factory=CoacdSettings)
    obj2mjcf: Obj2MjcfSettings = field(default_factory=Obj2MjcfSettings)
    mujoco: MujocoSettings = field(default_factory=MujocoSettings)

    @classmethod
    def from_toml(cls, path: Path) -> "Stage3Settings":
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        geometry = payload.get("geometry", {})
        coacd = payload.get("coacd", {})
        obj2mjcf = payload.get("obj2mjcf", {})
        mujoco = payload.get("mujoco", {})
        gravity = mujoco.get("gravity", [0.0, 0.0, -9.81])
        settings = cls(
            input_mesh_run_dir=Path(
                _expand_environment(
                    str(payload.get("input_mesh_run_dir", "artifacts/runs/stage2"))
                )
            ),
            output_dir=Path(
                _expand_environment(str(payload.get("output_dir", "artifacts/runs/stage3")))
            ),
            geometry=GeometrySettings(
                scale_mode=str(geometry.get("scale_mode", "normalize_max_extent")),
                target_extent_m=float(geometry.get("target_extent_m", 1.0)),
                align_support_plane=bool(geometry.get("align_support_plane", True)),
                plane_threshold_m=float(geometry.get("plane_threshold_m", 0.005)),
                plane_min_inlier_fraction=float(
                    geometry.get("plane_min_inlier_fraction", 0.05)
                ),
                visual_face_count=int(geometry.get("visual_face_count", 250_000)),
            ),
            coacd=CoacdSettings(
                version=str(coacd.get("version", COACD_VERSION)),
                threshold_m=float(coacd.get("threshold_m", 0.005)),
                real_metric=bool(coacd.get("real_metric", True)),
                max_convex_hulls=int(coacd.get("max_convex_hulls", 64)),
                preprocess_mode=str(coacd.get("preprocess_mode", "auto")),
                preprocess_resolution=int(coacd.get("preprocess_resolution", 50)),
                resolution=int(coacd.get("resolution", 2_000)),
                mcts_nodes=int(coacd.get("mcts_nodes", 20)),
                mcts_iterations=int(coacd.get("mcts_iterations", 150)),
                mcts_max_depth=int(coacd.get("mcts_max_depth", 3)),
                decimate=bool(coacd.get("decimate", True)),
                max_ch_vertex=int(coacd.get("max_ch_vertex", 64)),
                seed=int(coacd.get("seed", 0)),
            ),
            obj2mjcf=Obj2MjcfSettings(
                version=str(obj2mjcf.get("version", OBJ2MJCF_VERSION))
            ),
            mujoco=MujocoSettings(
                version=str(mujoco.get("version", MUJOCO_VERSION)),
                gravity=_gravity_tuple(gravity),
            ),
        )
        settings.validate()
        return settings

    def resolved(
        self,
        repo_root: Path,
        *,
        input_mesh_run_dir: Path | None = None,
        output_dir: Path | None = None,
    ) -> "Stage3Settings":
        return Stage3Settings(
            input_mesh_run_dir=_resolve_path(
                repo_root,
                input_mesh_run_dir
                if input_mesh_run_dir is not None
                else self.input_mesh_run_dir,
            ),
            output_dir=_resolve_path(
                repo_root, output_dir if output_dir is not None else self.output_dir
            ),
            geometry=self.geometry,
            coacd=self.coacd,
            obj2mjcf=self.obj2mjcf,
            mujoco=self.mujoco,
        )

    def validate(self) -> None:
        if self.geometry.scale_mode != "normalize_max_extent":
            raise ValueError("Only geometry.scale_mode='normalize_max_extent' is supported")
        if not self.geometry.align_support_plane:
            raise ValueError("geometry.align_support_plane must be true for Stage 3")
        if self.geometry.target_extent_m <= 0:
            raise ValueError("geometry.target_extent_m must be positive")
        if self.geometry.plane_threshold_m <= 0:
            raise ValueError("geometry.plane_threshold_m must be positive")
        if not 0 < self.geometry.plane_min_inlier_fraction <= 1:
            raise ValueError("geometry.plane_min_inlier_fraction must be in (0, 1]")
        if self.geometry.visual_face_count <= 0:
            raise ValueError("geometry.visual_face_count must be positive")
        if self.coacd.version != COACD_VERSION:
            raise ValueError(f"coacd.version must be {COACD_VERSION}")
        if self.coacd.threshold_m <= 0:
            raise ValueError("coacd.threshold_m must be positive")
        if not self.coacd.real_metric:
            raise ValueError("coacd.real_metric must be true")
        if self.coacd.max_convex_hulls <= 0:
            raise ValueError("coacd.max_convex_hulls must be positive")
        if self.coacd.preprocess_mode not in {"auto", "on", "off"}:
            raise ValueError("coacd.preprocess_mode must be auto, on, or off")
        for name in (
            "preprocess_resolution",
            "resolution",
            "mcts_nodes",
            "mcts_iterations",
            "mcts_max_depth",
            "max_ch_vertex",
        ):
            if getattr(self.coacd, name) <= 0:
                raise ValueError(f"coacd.{name} must be positive")
        if self.obj2mjcf.version != OBJ2MJCF_VERSION:
            raise ValueError(f"obj2mjcf.version must be {OBJ2MJCF_VERSION}")
        if self.mujoco.version != MUJOCO_VERSION:
            raise ValueError(f"mujoco.version must be {MUJOCO_VERSION}")
        if len(self.mujoco.gravity) != 3:
            raise ValueError("mujoco.gravity must contain three values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_mesh_run_dir": str(self.input_mesh_run_dir),
            "output_dir": str(self.output_dir),
            "geometry": asdict(self.geometry),
            "coacd": asdict(self.coacd),
            "obj2mjcf": asdict(self.obj2mjcf),
            "mujoco": {
                "version": self.mujoco.version,
                "gravity": list(self.mujoco.gravity),
            },
        }


def _gravity_tuple(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError("mujoco.gravity must contain three values")
    return (float(value[0]), float(value[1]), float(value[2]))
