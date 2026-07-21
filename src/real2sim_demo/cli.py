from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Stage1Settings
from .pipeline import run_stage1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the stage-1 real video -> COLMAP -> HY-World 3DGS pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage1.windows.example.toml"),
        help="TOML configuration file.",
    )
    parser.add_argument(
        "--stage",
        choices=("prepare", "train", "all"),
        default="all",
        help="Run only preparation, only Gaussian training, or both.",
    )
    parser.add_argument("--video", type=Path, help="Override [video].path.")
    parser.add_argument("--run-dir", type=Path, help="Override run_dir.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print external commands and write a trace without executing them.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.config.is_file():
        print(f"Config does not exist: {args.config}", file=sys.stderr)
        return 2
    repo_root = Path(__file__).resolve().parents[2]
    settings = Stage1Settings.from_toml(args.config).resolved(
        repo_root,
        video=args.video,
        run_dir=args.run_dir,
    )
    try:
        manifest = run_stage1(settings, stage=args.stage, dry_run=args.dry_run)
    except Exception as exc:
        print(f"Stage 1 failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
