from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .mesh_config import Stage2Settings
from .mesh_pipeline import run_stage2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Stage 2: COLMAP scene -> 2D Gaussian Splatting -> bounded mesh."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage2.tabletop_v1.toml"),
        help="TOML configuration file.",
    )
    parser.add_argument(
        "--stage",
        choices=("prepare", "train", "mesh", "all"),
        default="all",
        help="Run dataset preparation, 2DGS training, mesh export, or the complete pipeline.",
    )
    parser.add_argument("--input-run-dir", type=Path, help="Override input_run_dir.")
    parser.add_argument("--output-dir", type=Path, help="Override output_dir.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write trace and planned external commands without executing them.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.config.is_file():
        print(f"Config does not exist: {args.config}", file=sys.stderr)
        return 2
    repo_root = Path(__file__).resolve().parents[2]
    try:
        settings = Stage2Settings.from_toml(args.config).resolved(
            repo_root,
            input_run_dir=args.input_run_dir,
            output_dir=args.output_dir,
        )
        manifest = run_stage2(settings, stage=args.stage, dry_run=args.dry_run)
    except Exception as exc:
        print(f"Stage 2 failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
