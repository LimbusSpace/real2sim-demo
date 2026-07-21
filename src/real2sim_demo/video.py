from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .process import CommandResult, resolve_executable, run_command


@dataclass(frozen=True, slots=True)
class VideoExtractionConfig:
    video_path: Path
    output_dir: Path
    ffmpeg_executable: str = "ffmpeg"
    fps: float = 2.0


def extract_frames(
    config: VideoExtractionConfig,
    *,
    dry_run: bool = False,
) -> tuple[list[Path], CommandResult]:
    if not dry_run and not config.video_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {config.video_path}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = (
        resolve_executable(config.ffmpeg_executable)
        if not dry_run
        else config.ffmpeg_executable
    )
    pattern = config.output_dir / "frame_%06d.png"
    command: list[str | Path] = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-i",
        config.video_path,
        "-vf",
        f"fps={config.fps:g}",
        "-vsync",
        "vfr",
        "-q:v",
        "2",
        pattern,
    ]
    result = run_command(
        command,
        config.output_dir.parent / "logs" / "01_extract_frames.log",
        dry_run=dry_run,
    )
    frames = sorted(config.output_dir.glob("frame_*.png"))
    if not dry_run and len(frames) < 2:
        raise RuntimeError(
            f"Frame extraction produced {len(frames)} frames; at least 2 are required."
        )
    manifest = {
        "schema": "real2sim.frames.v1",
        "source_video": str(config.video_path.resolve()),
        "fps": config.fps,
        "frame_count": len(frames),
        "frames": [str(path.resolve()) for path in frames],
    }
    (config.output_dir.parent / "frames_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return frames, result
