from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .physics_config import Stage3Settings
from .physics_pipeline import run_stage3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Stage 3: colored mesh -> CoACD -> static MuJoCo scene."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage3.tabletop_v1.toml"),
        help="TOML configuration file.",
    )
    parser.add_argument(
        "--stage",
        choices=("prepare", "decompose", "mjcf", "validate", "all"),
        default="all",
    )
    parser.add_argument("--input-mesh-run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.config.is_file():
        print(f"Config does not exist: {args.config}", file=sys.stderr)
        return 2
    repo_root = Path(__file__).resolve().parents[2]
    try:
        settings = Stage3Settings.from_toml(args.config).resolved(
            repo_root,
            input_mesh_run_dir=args.input_mesh_run_dir,
            output_dir=args.output_dir,
        )
        manifest = run_stage3(settings, stage=args.stage, dry_run=args.dry_run)
    except Exception as exc:
        print(f"Stage 3 failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
