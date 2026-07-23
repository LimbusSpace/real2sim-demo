from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from real2sim_demo.physics_config import GeometrySettings
from real2sim_demo.physics_geometry import (
    apply_transform,
    fit_support_plane,
    normalize_scale,
    prepare_visual_mesh,
    rotation_to_positive_z,
)


def test_normalize_scale_uses_maximum_extent() -> None:
    assert normalize_scale(15.1, 1.0) == pytest.approx(0.06622516556)


def test_support_plane_fit_orients_object_side_positive() -> None:
    grid = np.array(
        [[x, y, 0.0] for x in np.linspace(-1, 1, 12) for y in np.linspace(-1, 1, 12)]
    )
    above = np.array([[0.2, 0.1, value] for value in np.linspace(0.1, 1.0, 30)])
    vertices = np.vstack((grid, above))

    plane = fit_support_plane(
        vertices,
        threshold=1e-5,
        min_inlier_fraction=0.5,
        seed=0,
        iterations=100,
    )
    normal = np.asarray(plane.normal)
    rotation = rotation_to_positive_z(normal)

    assert plane.inlier_fraction >= 0.8
    assert np.dot(np.array([0.0, 0.0, 0.8]) - np.asarray(plane.point), normal) > 0
    assert rotation @ normal == pytest.approx([0.0, 0.0, 1.0], abs=1e-8)


def test_support_plane_fit_rejects_low_inlier_fraction() -> None:
    vertices = np.random.default_rng(7).normal(size=(200, 3))

    with pytest.raises(ValueError, match="inlier fraction"):
        fit_support_plane(
            vertices,
            threshold=1e-12,
            min_inlier_fraction=0.05,
            seed=0,
            iterations=100,
        )


def test_apply_transform_uses_homogeneous_matrix() -> None:
    vertices = np.array([[1.0, 2.0, 3.0]])
    matrix = np.eye(4)
    matrix[:3, 3] = [2.0, -1.0, 0.5]

    assert np.allclose(apply_transform(vertices, matrix), [[3.0, 1.0, 3.5]])


def test_prepare_visual_mesh_writes_obj_mtl_transform_and_face_limit(tmp_path: Path) -> None:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    mesh.visual.vertex_colors = np.tile([20, 80, 140, 255], (len(mesh.vertices), 1))
    input_ply = tmp_path / "post.ply"
    mesh.export(input_ply)
    output_obj = tmp_path / "source" / "scene.obj"
    output_mtl = tmp_path / "source" / "scene.mtl"
    transform_path = tmp_path / "source" / "transform.json"

    result = prepare_visual_mesh(
        input_ply,
        output_obj,
        output_mtl,
        transform_path,
        GeometrySettings(
            plane_threshold_m=0.02,
            plane_min_inlier_fraction=0.03,
            visual_face_count=100,
        ),
    )

    assert output_obj.is_file()
    assert output_mtl.is_file()
    assert "mtllib scene.mtl" in output_obj.read_text(encoding="ascii")
    assert "Kd 0.07843137 0.31372549 0.54901961" in output_mtl.read_text(encoding="ascii")
    assert result["output_stats"]["face_count"] <= 100
    assert max(result["output_bbox"]["extents"]) == pytest.approx(1.0)
    stored = json.loads(transform_path.read_text(encoding="utf-8"))
    assert len(stored["transform_matrix"]) == 4
    assert stored["mean_rgb"] == [20, 80, 140]
