from __future__ import annotations

from pathlib import Path

import pytest

from real2sim_demo.mesh_ply import validate_mesh_pair, validate_mesh_ply


def test_validate_colored_triangle_mesh(tmp_path: Path) -> None:
    raw = _write_mesh(tmp_path / "raw.ply", vertices=3, faces=1, color=True)
    post = _write_mesh(tmp_path / "post.ply", vertices=3, faces=1, color=True)

    raw_stats, post_stats = validate_mesh_pair(raw, post)

    assert raw_stats.vertex_count == 3
    assert raw_stats.face_count == 1
    assert raw_stats.has_rgb is True
    assert post_stats.vertex_count == 3


@pytest.mark.parametrize(
    ("vertices", "faces", "color", "message"),
    [
        (3, 0, True, "no triangle faces"),
        (3, 1, False, "no RGB vertex colors"),
        (0, 1, True, "no vertices"),
    ],
)
def test_validate_mesh_rejects_missing_required_data(
    tmp_path: Path, vertices: int, faces: int, color: bool, message: str
) -> None:
    path = _write_mesh(tmp_path / "mesh.ply", vertices=vertices, faces=faces, color=color)

    with pytest.raises(ValueError, match=message):
        validate_mesh_ply(path)


def test_validate_mesh_rejects_corrupt_header(tmp_path: Path) -> None:
    path = tmp_path / "mesh.ply"
    path.write_bytes(b"not-ply\n")

    with pytest.raises(ValueError, match="magic|end_header"):
        validate_mesh_ply(path)


def test_validate_mesh_rejects_post_larger_than_raw(tmp_path: Path) -> None:
    raw = _write_mesh(tmp_path / "raw.ply", vertices=3, faces=1, color=True)
    post = _write_mesh(tmp_path / "post.ply", vertices=4, faces=2, color=True)

    with pytest.raises(ValueError, match="larger"):
        validate_mesh_pair(raw, post)


def _write_mesh(path: Path, *, vertices: int, faces: int, color: bool) -> Path:
    properties = ["property float x", "property float y", "property float z"]
    if color:
        properties.extend(["property uchar red", "property uchar green", "property uchar blue"])
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {vertices}",
        *properties,
        f"element face {faces}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    vertex_row = "0 0 0 10 20 30" if color else "0 0 0"
    body = [vertex_row for _ in range(vertices)] + ["3 0 1 2" for _ in range(faces)]
    path.write_text("\n".join(header + body) + "\n", encoding="ascii")
    return path
