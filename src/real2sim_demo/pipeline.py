from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .colmap import ColmapConfig, run_colmap
from .colmap_text import convert_colmap_text_to_hyworld
from .config import Stage1Settings
from .gaussian import GaussianRunConfig, parse_hyworld_evaluation, train_hyworld
from .video import VideoExtractionConfig, extract_frames


def run_stage1(
    settings: Stage1Settings,
    *,
    stage: str = "all",
    dry_run: bool = False,
) -> dict[str, Any]:
    if stage not in {"prepare", "train", "all"}:
        raise ValueError(f"Unsupported stage: {stage}")
    run_dir = settings.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_or_create_manifest(settings, dry_run=dry_run)
    manifest.pop("error", None)
    manifest["settings"] = settings.to_dict()
    trace: list[dict[str, Any]] = []

    try:
        if stage in {"prepare", "all"}:
            frames_dir = run_dir / "frames"
            frames, _ = _timed_step(
                trace,
                "extract_frames",
                lambda: extract_frames(
                    VideoExtractionConfig(
                        video_path=settings.video.path,
                        output_dir=frames_dir,
                        ffmpeg_executable=settings.video.ffmpeg,
                        fps=settings.video.fps,
                    ),
                    dry_run=dry_run,
                ),
            )
            manifest["artifacts"]["frames_dir"] = str(frames_dir.resolve())
            manifest["artifacts"]["frame_count"] = len(frames)

            colmap_artifacts, _ = _timed_step(
                trace,
                "colmap_reconstruction",
                lambda: run_colmap(
                    frames_dir,
                    run_dir / "colmap",
                    ColmapConfig(
                        executable=settings.colmap.executable,
                        camera_model=settings.colmap.camera_model,
                        matcher=settings.colmap.matcher,
                        use_gpu=settings.colmap.use_gpu,
                        sequential_overlap=settings.colmap.sequential_overlap,
                    ),
                    dry_run=dry_run,
                ),
            )
            manifest["artifacts"]["colmap_text_model"] = str(
                colmap_artifacts.text_model_dir.resolve()
            )
            if not dry_run:
                provenance = convert_colmap_text_to_hyworld(
                    colmap_artifacts.text_model_dir,
                    colmap_artifacts.source_images_dir,
                    run_dir / "hyworld_dataset",
                )
                manifest["artifacts"]["hyworld_dataset"] = str(
                    (run_dir / "hyworld_dataset").resolve()
                )
                manifest["reconstruction"] = provenance
            manifest["stage"] = "commands_planned" if dry_run else "prepared"

        if stage in {"train", "all"}:
            dataset_dir = run_dir / "hyworld_dataset"
            if not dry_run and not (dataset_dir / "cameras.json").is_file():
                raise FileNotFoundError(
                    f"Missing {dataset_dir / 'cameras.json'}; run stage=prepare first."
                )
            ply, training_result = _timed_step(
                trace,
                "gaussian_training",
                lambda: train_hyworld(
                    dataset_dir,
                    run_dir / "gaussian",
                    GaussianRunConfig(
                        python=settings.gaussian.python,
                        launcher=settings.gaussian.launcher,
                        torch_home=settings.gaussian.torch_home,
                        source_revision=settings.gaussian.source_revision,
                        max_steps=settings.gaussian.max_steps,
                        data_factor=settings.gaussian.data_factor,
                        sh_degree=settings.gaussian.sh_degree,
                        test_every=settings.gaussian.test_every,
                        disable_video=settings.gaussian.disable_video,
                    ),
                    dry_run=dry_run,
                ),
            )
            if ply is not None:
                manifest["artifacts"]["gaussian_ply"] = str(ply.resolve())
                manifest["training"] = {
                    "backend": "hyworld",
                    "steps": settings.gaussian.max_steps,
                    "test_every": settings.gaussian.test_every,
                    "evaluation": parse_hyworld_evaluation(training_result.log_path),
                }
            manifest["stage"] = "commands_planned" if dry_run else "gaussian_trained"
    except Exception as exc:
        manifest["stage"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["updated_at"] = _now()
        manifest["dry_run"] = dry_run
        _write_json(run_dir / "manifest.json", manifest)
        _write_json(run_dir / "trace.json", {"schema": "real2sim.trace.v1", "events": trace})
    return manifest


def _timed_step(trace: list[dict[str, Any]], name: str, action: Any) -> Any:
    started = time.perf_counter()
    event: dict[str, Any] = {
        "stage": name,
        "status": "running",
        "started_at": _now(),
    }
    trace.append(event)
    try:
        result = action()
    except Exception as exc:
        event.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "duration_s": round(time.perf_counter() - started, 3),
                "ended_at": _now(),
            }
        )
        raise
    event.update(
        {
            "status": "completed",
            "duration_s": round(time.perf_counter() - started, 3),
            "ended_at": _now(),
        }
    )
    return result


def _load_or_create_manifest(settings: Stage1Settings, *, dry_run: bool) -> dict[str, Any]:
    path = settings.run_dir / "manifest.json"
    if path.is_file():
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return payload
    video_info: dict[str, Any] = {"path": str(settings.video.path.resolve())}
    if settings.video.path.is_file() and not dry_run:
        video_info["sha256"] = _sha256(settings.video.path)
        video_info["bytes"] = settings.video.path.stat().st_size
    return {
        "schema": "real2sim.stage1.manifest.v1",
        "run_id": settings.run_dir.name,
        "created_at": _now(),
        "stage": "initialized",
        "input": video_info,
        "settings": settings.to_dict(),
        "artifacts": {},
        "reconstruction": {},
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
