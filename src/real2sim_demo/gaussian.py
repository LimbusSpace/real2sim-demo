from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .process import CommandResult, resolve_executable, run_command

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_EVALUATION = re.compile(
    r"PSNR:\s*(?P<psnr>[0-9]+(?:\.[0-9]+)?),\s*"
    r"SSIM:\s*(?P<ssim>[0-9]+(?:\.[0-9]+)?),\s*"
    r"LPIPS:\s*(?P<lpips>[0-9]+(?:\.[0-9]+)?).*?"
    r"Number of GS:\s*(?P<gaussian_count>[0-9]+)",
    flags=re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class GaussianRunConfig:
    python: str
    launcher: str
    torch_home: str = ""
    source_revision: str = ""
    max_steps: int = 5_000
    data_factor: int = 1
    sh_degree: int = 1
    test_every: int = 8
    disable_video: bool = True


def train_hyworld(
    dataset_dir: Path,
    output_dir: Path,
    config: GaussianRunConfig,
    *,
    dry_run: bool = False,
) -> tuple[Path | None, CommandResult]:
    if not config.python:
        raise ValueError("gaussian.python is required for the HY-World backend.")
    if not config.launcher:
        raise ValueError("gaussian.launcher is required for the HY-World backend.")
    python = config.python if dry_run else resolve_executable(config.python)
    launcher = config.launcher
    if not dry_run and not Path(launcher).is_file():
        raise FileNotFoundError(f"HY-World launcher does not exist: {launcher}")
    if not dry_run:
        prepare_scaled_images(dataset_dir, config.data_factor)
    output_dir.mkdir(parents=True, exist_ok=True)
    command: list[str | Path] = [
        python,
        launcher,
        "default",
        "--disable-viewer",
        "--data-dir",
        dataset_dir,
        "--result-dir",
        output_dir,
        "--test-every",
        str(config.test_every),
        "--max-steps",
        str(config.max_steps),
        "--data-factor",
        str(config.data_factor),
        "--eval-steps",
        str(config.max_steps),
        "--save-steps",
        str(config.max_steps),
        "--ply-steps",
        str(config.max_steps),
        "--save-ply",
        "--init-type",
        "sfm",
        "--sh-degree",
        str(config.sh_degree),
        "--lpips-lambda1",
        "0",
        "--lpips-lambda2",
        "0",
        "--no-normalize",
    ]
    if config.disable_video:
        command.append("--disable-video")
    environment = {"PYTHONUTF8": "1"}
    if config.torch_home:
        environment["TORCH_HOME"] = config.torch_home
    result = run_command(
        command,
        output_dir.parent / "logs" / "07_hyworld_train.log",
        cwd=Path(launcher).parent if not dry_run else None,
        env=environment,
        dry_run=dry_run,
    )
    if dry_run:
        return None, result
    ply_candidates = sorted(
        (output_dir / "ply").glob("point_cloud_*.ply"),
        key=_ply_step,
    )
    if not ply_candidates:
        raise FileNotFoundError(
            f"HY-World finished without a PLY under {output_dir / 'ply'}."
        )
    selected = ply_candidates[-1]
    evaluation = parse_hyworld_evaluation(result.log_path)
    provenance = {
        "schema": "real2sim.gaussian_output.v1",
        "backend": "hyworld",
        "trainer_launcher": str(Path(launcher).resolve()),
        "python": str(Path(python).resolve()),
        "torch_home": config.torch_home,
        "source_revision": config.source_revision,
        "max_steps": config.max_steps,
        "data_factor": config.data_factor,
        "sh_degree": config.sh_degree,
        "ply": str(selected.resolve()),
    }
    if evaluation:
        provenance["evaluation"] = evaluation
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    return selected, result


def parse_hyworld_evaluation(log_path: Path) -> dict[str, float | int]:
    """Return the final evaluation reported by the HY-World trainer."""
    if not log_path.is_file():
        return {}
    text = _ANSI_ESCAPE.sub("", log_path.read_text(encoding="utf-8", errors="replace"))
    matches = list(_EVALUATION.finditer(text))
    if not matches:
        return {}
    values = matches[-1].groupdict()
    return {
        "psnr": float(values["psnr"]),
        "ssim": float(values["ssim"]),
        "lpips": float(values["lpips"]),
        "gaussian_count": int(values["gaussian_count"]),
    }


def _ply_step(path: Path) -> int:
    match = re.fullmatch(r"point_cloud_(\d+)", path.stem)
    return int(match.group(1)) if match is not None else -1


def prepare_scaled_images(dataset_dir: Path, factor: int) -> Path:
    source_dir = dataset_dir / "images"
    if factor <= 1:
        return source_dir
    source_images = sorted(source_dir.glob("*.png"))
    if not source_images:
        raise FileNotFoundError(f"No PNG images found under {source_dir}.")
    output_dir = dataset_dir / f"images_{factor}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in source_images:
        destination = output_dir / source.name
        if destination.is_file():
            continue
        with Image.open(source) as image:
            width = max(1, int(round(image.width / factor)))
            height = max(1, int(round(image.height / factor)))
            image.resize((width, height), Image.Resampling.LANCZOS).save(destination)
    return output_dir
