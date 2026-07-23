from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .physics_config import CoacdSettings


@dataclass(frozen=True, slots=True)
class ObjStats:
    vertex_count: int
    face_count: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def build_coacd_command(
    python: str,
    source_obj: Path,
    output_dir: Path,
    settings: CoacdSettings,
) -> list[str]:
    command = [
        python,
        "-m",
        "real2sim_demo.physics_coacd",
        "--input",
        str(source_obj),
        "--output-dir",
        str(output_dir),
        "--threshold",
        str(settings.threshold_m),
        "--max-convex-hull",
        str(settings.max_convex_hulls),
        "--preprocess-mode",
        settings.preprocess_mode,
        "--preprocess-resolution",
        str(settings.preprocess_resolution),
        "--resolution",
        str(settings.resolution),
        "--mcts-nodes",
        str(settings.mcts_nodes),
        "--mcts-iterations",
        str(settings.mcts_iterations),
        "--mcts-max-depth",
        str(settings.mcts_max_depth),
        "--max-ch-vertex",
        str(settings.max_ch_vertex),
        "--seed",
        str(settings.seed),
    ]
    if settings.real_metric:
        command.append("--real-metric")
    if settings.decimate:
        command.append("--decimate")
    return command


def read_obj_stats(path: Path) -> ObjStats:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty OBJ file: {path}")
    vertex_count = 0
    face_rows: list[tuple[int, list[str]]] = []
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if line.startswith("v "):
                fields = line.split()
                if len(fields) < 4:
                    raise ValueError(f"Malformed OBJ vertex at {path}:{line_number}")
                vertex_count += 1
            elif line.startswith("f "):
                fields = line.split()[1:]
                if len(fields) != 3:
                    raise ValueError(f"Non-triangular OBJ face at {path}:{line_number}")
                face_rows.append((line_number, fields))
    if vertex_count <= 0 or not face_rows:
        raise ValueError(f"OBJ contains no vertices or triangle faces: {path}")
    for line_number, fields in face_rows:
        for field in fields:
            token = field.split("/", maxsplit=1)[0]
            try:
                index = int(token)
            except ValueError as exc:
                raise ValueError(f"Malformed OBJ index at {path}:{line_number}") from exc
            if index == 0 or index > vertex_count or index < -vertex_count:
                raise ValueError(f"OBJ index out of range at {path}:{line_number}: {index}")
    return ObjStats(vertex_count=vertex_count, face_count=len(face_rows))


def collect_collision_stats(collision_dir: Path) -> dict[str, Any]:
    paths = sorted(collision_dir.glob("scene_collision_*.obj"))
    if not paths:
        raise ValueError(f"CoACD produced no collision OBJ files in {collision_dir}")
    parts = []
    total_vertices = 0
    total_faces = 0
    for path in paths:
        stats = read_obj_stats(path)
        total_vertices += stats.vertex_count
        total_faces += stats.face_count
        parts.append({"path": str(path.resolve()), **stats.to_dict()})
    return {
        "count": len(paths),
        "total_vertex_count": total_vertices,
        "total_face_count": total_faces,
        "parts": parts,
    }
