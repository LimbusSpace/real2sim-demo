from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .process import CommandResult, resolve_executable, run_command


@dataclass(frozen=True, slots=True)
class Mast3rConfig:
    python: str
    repository: Path
    weights: Path
    runner: Path
    device: str = "cuda"
    image_size: int = 512
    scene_graph: str = "logwin"
    window_size: int = 5
    cyclic: bool = True
    shared_intrinsics: bool = True
    coarse_iterations: int = 300
    fine_iterations: int = 300
    matching_confidence: float = 5.0
    max_points: int = 200_000


@dataclass(frozen=True, slots=True)
class Mast3rArtifacts:
    source_images_dir: Path
    text_model_dir: Path
    stats_path: Path
    cache_dir: Path


def run_mast3r(
    frames_dir: Path,
    output_dir: Path,
    config: Mast3rConfig,
    *,
    dry_run: bool = False,
) -> tuple[Mast3rArtifacts, CommandResult]:
    python = config.python if dry_run else resolve_executable(config.python)
    text_model_dir = output_dir / "model_txt"
    stats_path = output_dir / "stats.json"
    cache_dir = output_dir / "cache"
    for path in (output_dir, text_model_dir, cache_dir):
        path.mkdir(parents=True, exist_ok=True)

    if not dry_run:
        _require_file(config.runner, "MASt3R runner")
        _require_file(config.weights, "MASt3R checkpoint")
        if not (config.repository / "mast3r" / "model.py").is_file():
            raise FileNotFoundError(f"Invalid MASt3R repository: {config.repository}")

    command: list[str | Path] = [
        python,
        config.runner,
        "--repository",
        config.repository,
        "--weights",
        config.weights,
        "--images",
        frames_dir,
        "--output",
        output_dir,
        "--device",
        config.device,
        "--image-size",
        str(config.image_size),
        "--scene-graph",
        config.scene_graph,
        "--window-size",
        str(config.window_size),
        "--coarse-iterations",
        str(config.coarse_iterations),
        "--fine-iterations",
        str(config.fine_iterations),
        "--matching-confidence",
        str(config.matching_confidence),
        "--max-points",
        str(config.max_points),
    ]
    if config.cyclic:
        command.append("--cyclic")
    if config.shared_intrinsics:
        command.append("--shared-intrinsics")

    result = run_command(
        command,
        output_dir.parent / "logs" / "02_mast3r_sfm.log",
        dry_run=dry_run,
    )
    artifacts = Mast3rArtifacts(
        source_images_dir=frames_dir,
        text_model_dir=text_model_dir,
        stats_path=stats_path,
        cache_dir=cache_dir,
    )
    if not dry_run:
        _validate_outputs(artifacts)
    return artifacts, result


def read_mast3r_stats(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _validate_outputs(artifacts: Mast3rArtifacts) -> None:
    required = [
        artifacts.text_model_dir / "cameras.txt",
        artifacts.text_model_dir / "images.txt",
        artifacts.text_model_dir / "points3D.txt",
        artifacts.stats_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("MASt3R-SfM did not produce a usable model: " + ", ".join(missing))
    stats = read_mast3r_stats(artifacts.stats_path)
    if int(stats.get("registered_image_count", 0)) < 2:
        raise RuntimeError("MASt3R-SfM registered fewer than two images")
