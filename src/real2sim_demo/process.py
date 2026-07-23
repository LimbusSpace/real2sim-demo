from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class CommandResult:
    command: list[str]
    returncode: int
    log_path: Path
    dry_run: bool = False


def resolve_executable(value: str | Path) -> str:
    candidate = str(value)
    path = Path(candidate)
    if path.exists():
        return str(path.resolve())
    resolved = which(candidate)
    if resolved is None:
        raise FileNotFoundError(
            f"Executable not found: {candidate}. Pass an absolute path or add it to PATH."
        )
    return resolved


def run_command(
    command: Sequence[str | Path],
    log_path: Path,
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    dry_run: bool = False,
) -> CommandResult:
    normalized = [str(item) for item in command]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("$ " + _display_command(normalized) + "\n", encoding="utf-8")
    if dry_run:
        print("[dry-run] " + _display_command(normalized))
        return CommandResult(normalized, 0, log_path, dry_run=True)

    merged_env = os.environ.copy()
    if env is not None:
        merged_env.update(env)
    print("[run] " + _display_command(normalized), flush=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            normalized,
            cwd=str(cwd) if cwd is not None else None,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            output_encoding = sys.stdout.encoding or "utf-8"
            console_line = line.encode(output_encoding, errors="replace").decode(output_encoding)
            print(console_line, end="", flush=True)
            log_file.write(line)
        returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, normalized)
    return CommandResult(normalized, returncode, log_path)


def _display_command(command: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(command))
    return " ".join(_quote_unix(item) for item in command)


def _quote_unix(value: str) -> str:
    if not value or any(char.isspace() for char in value):
        return "'" + value.replace("'", "'\\''") + "'"
    return value
