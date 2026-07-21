from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from .process import CommandResult, resolve_executable, run_command


@dataclass(frozen=True, slots=True)
class ColmapConfig:
    executable: str = "colmap"
    camera_model: str = "SIMPLE_RADIAL"
    matcher: str = "sequential"
    use_gpu: bool = True
    sequential_overlap: int = 10


@dataclass(frozen=True, slots=True)
class ColmapArtifacts:
    source_images_dir: Path
    sparse_model_dir: Path
    text_model_dir: Path
    database_path: Path


def run_colmap(
    frames_dir: Path,
    output_dir: Path,
    config: ColmapConfig,
    *,
    dry_run: bool = False,
) -> tuple[ColmapArtifacts, list[CommandResult]]:
    executable = config.executable if dry_run else resolve_executable(config.executable)
    database_path = output_dir / "database.db"
    sparse_root = output_dir / "sparse"
    undistorted_root = output_dir / "undistorted"
    text_model_dir = output_dir / "model_txt"
    for path in (output_dir, sparse_root, undistorted_root, text_model_dir):
        path.mkdir(parents=True, exist_ok=True)

    gpu = "1" if config.use_gpu else "0"
    commands: list[tuple[str, list[str | Path]]] = [
        (
            "02_colmap_features",
            [
                executable,
                "feature_extractor",
                "--database_path",
                database_path,
                "--image_path",
                frames_dir,
                "--ImageReader.single_camera",
                "1",
                "--ImageReader.camera_model",
                config.camera_model,
                "--FeatureExtraction.use_gpu",
                gpu,
            ],
        )
    ]
    if config.matcher == "sequential":
        commands.append(
            (
                "03_colmap_matching",
                [
                    executable,
                    "sequential_matcher",
                    "--database_path",
                    database_path,
                    "--SequentialMatching.overlap",
                    str(config.sequential_overlap),
                    "--FeatureMatching.use_gpu",
                    gpu,
                ],
            )
        )
    elif config.matcher == "exhaustive":
        commands.append(
            (
                "03_colmap_matching",
                [
                    executable,
                    "exhaustive_matcher",
                    "--database_path",
                    database_path,
                    "--FeatureMatching.use_gpu",
                    gpu,
                ],
            )
        )
    else:
        raise ValueError(f"Unsupported COLMAP matcher: {config.matcher}")

    commands.append(
        (
            "04_colmap_mapper",
            [
                executable,
                "mapper",
                "--database_path",
                database_path,
                "--image_path",
                frames_dir,
                "--output_path",
                sparse_root,
            ],
        )
    )

    results = []
    for name, command in commands:
        results.append(
            run_command(
                command,
                output_dir.parent / "logs" / f"{name}.log",
                dry_run=dry_run,
            )
        )

    selected_model = sparse_root / "0" if dry_run else _select_largest_model(sparse_root)
    final_commands: list[tuple[str, list[str | Path]]] = [
        (
            "05_colmap_undistort",
            [
                executable,
                "image_undistorter",
                "--image_path",
                frames_dir,
                "--input_path",
                selected_model,
                "--output_path",
                undistorted_root,
                "--output_type",
                "COLMAP",
            ],
        ),
        (
            "06_colmap_model_to_text",
            [
                executable,
                "model_converter",
                "--input_path",
                undistorted_root / "sparse",
                "--output_path",
                text_model_dir,
                "--output_type",
                "TXT",
            ],
        ),
    ]
    for name, command in final_commands:
        results.append(
            run_command(
                command,
                output_dir.parent / "logs" / f"{name}.log",
                dry_run=dry_run,
            )
        )
    artifacts = ColmapArtifacts(
        source_images_dir=undistorted_root / "images",
        sparse_model_dir=undistorted_root / "sparse",
        text_model_dir=text_model_dir,
        database_path=database_path,
    )
    if not dry_run:
        _validate_outputs(artifacts)
    return artifacts, results


def _select_largest_model(sparse_root: Path) -> Path:
    candidates: list[tuple[int, int, Path]] = []
    for model_dir in sparse_root.iterdir():
        images_path = model_dir / "images.bin"
        if not model_dir.is_dir() or not images_path.is_file():
            continue
        with images_path.open("rb") as handle:
            header = handle.read(8)
        if len(header) != 8:
            continue
        image_count = struct.unpack("<Q", header)[0]
        points_path = model_dir / "points3D.bin"
        points_bytes = points_path.stat().st_size if points_path.is_file() else 0
        candidates.append((image_count, points_bytes, model_dir))
    if not candidates:
        raise RuntimeError(f"COLMAP mapper produced no sparse model under {sparse_root}.")
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _validate_outputs(artifacts: ColmapArtifacts) -> None:
    required = [
        artifacts.text_model_dir / "cameras.txt",
        artifacts.text_model_dir / "images.txt",
        artifacts.text_model_dir / "points3D.txt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("COLMAP did not produce a usable model: " + ", ".join(missing))
    images = [path for path in artifacts.source_images_dir.rglob("*") if path.is_file()]
    if len(images) < 2:
        raise RuntimeError(
            f"COLMAP undistortion produced {len(images)} images; at least 2 are required."
        )
