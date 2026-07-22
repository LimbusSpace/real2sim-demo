from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def verify_snapshot(manifest_path: Path, root: Path) -> list[str]:
    payload: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for item in payload.get("files", []):
        relative_path = Path(str(item["path"]))
        expected = str(item["sha256"]).lower()
        candidate = root / relative_path
        if not candidate.is_file():
            failures.append(f"missing: {relative_path.as_posix()}")
            continue
        actual = _sha256(candidate)
        if actual != expected:
            failures.append(
                f"hash mismatch: {relative_path.as_posix()} expected={expected} actual={actual}"
            )
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify a downloaded external source snapshot.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.manifest.is_file():
        print(f"Snapshot manifest does not exist: {args.manifest}", file=sys.stderr)
        return 2
    if not args.root.is_dir():
        print(f"Source root does not exist: {args.root}", file=sys.stderr)
        return 2
    failures = verify_snapshot(args.manifest, args.root)
    if failures:
        for failure in failures:
            print(f"Source snapshot verification failed: {failure}", file=sys.stderr)
        return 1
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    print(f"{payload['source']['name']} snapshot OK: {payload['source']['commit']}")
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
