from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PlyStats:
    vertex_count: int
    face_count: int
    has_rgb: bool
    format: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_ply_header(path: Path) -> PlyStats:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        raise ValueError(f"Empty PLY file: {path}")

    vertex_count: int | None = None
    face_count = 0
    ply_format: str | None = None
    current_element: str | None = None
    vertex_properties: set[str] = set()
    header_ended = False
    with path.open("rb") as handle:
        for line_number in range(1, 4097):
            raw = handle.readline(64 * 1024)
            if not raw:
                break
            try:
                line = raw.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise ValueError(f"PLY header is not ASCII: {path}") from exc
            if line_number == 1 and line != "ply":
                raise ValueError(f"Invalid PLY magic: {path}")
            fields = line.split()
            if fields[:1] == ["format"]:
                if len(fields) != 3 or fields[1] not in {
                    "ascii",
                    "binary_little_endian",
                    "binary_big_endian",
                }:
                    raise ValueError(f"Unsupported PLY format row: {line}")
                ply_format = fields[1]
            elif fields[:1] == ["element"]:
                if len(fields) != 3:
                    raise ValueError(f"Malformed PLY element row: {line}")
                current_element = fields[1]
                try:
                    count = int(fields[2])
                except ValueError as exc:
                    raise ValueError(f"Malformed PLY element count: {line}") from exc
                if count < 0:
                    raise ValueError(f"Negative PLY element count: {line}")
                if current_element == "vertex":
                    vertex_count = count
                elif current_element == "face":
                    face_count = count
            elif fields[:1] == ["property"] and current_element == "vertex":
                if len(fields) < 3:
                    raise ValueError(f"Malformed PLY property row: {line}")
                vertex_properties.add(fields[-1])
            elif line == "end_header":
                header_ended = True
                if handle.tell() >= path.stat().st_size:
                    raise ValueError(f"PLY has a header but no body: {path}")
                break

    if not header_ended:
        raise ValueError(f"PLY end_header was not found: {path}")
    if ply_format is None:
        raise ValueError(f"PLY format was not declared: {path}")
    if vertex_count is None:
        raise ValueError(f"PLY vertex element was not declared: {path}")
    return PlyStats(
        vertex_count=vertex_count,
        face_count=face_count,
        has_rgb={"red", "green", "blue"}.issubset(vertex_properties),
        format=ply_format,
    )


def validate_surfel_ply(path: Path) -> PlyStats:
    stats = read_ply_header(path)
    if stats.vertex_count <= 0:
        raise ValueError(f"Surfel PLY contains no vertices: {path}")
    return stats


def validate_mesh_ply(path: Path) -> PlyStats:
    stats = read_ply_header(path)
    failures: list[str] = []
    if stats.vertex_count <= 0:
        failures.append("no vertices")
    if stats.face_count <= 0:
        failures.append("no triangle faces")
    if not stats.has_rgb:
        failures.append("no RGB vertex colors")
    if failures:
        raise ValueError(f"Invalid mesh PLY {path}: " + ", ".join(failures))
    return stats


def validate_mesh_pair(raw_path: Path, post_path: Path) -> tuple[PlyStats, PlyStats]:
    raw = validate_mesh_ply(raw_path)
    post = validate_mesh_ply(post_path)
    if post.vertex_count > raw.vertex_count or post.face_count > raw.face_count:
        raise ValueError("Post-processed mesh is larger than the raw mesh")
    return raw, post
