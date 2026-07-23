from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .physics_config import GeometrySettings

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class PlaneFit:
    point: tuple[float, float, float]
    normal: tuple[float, float, float]
    threshold_source_units: float
    inlier_count: int
    sample_count: int
    inlier_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_scale(max_extent: float, target_extent_m: float) -> float:
    if max_extent <= 0:
        raise ValueError("Mesh maximum extent must be positive")
    if target_extent_m <= 0:
        raise ValueError("Target extent must be positive")
    return target_extent_m / max_extent


def fit_support_plane(
    vertices: FloatArray,
    *,
    threshold: float,
    min_inlier_fraction: float,
    seed: int = 0,
    max_samples: int = 50_000,
    iterations: int = 512,
) -> PlaneFit:
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
        raise ValueError("At least three 3D vertices are required to fit a plane")
    if threshold <= 0:
        raise ValueError("Plane threshold must be positive")
    rng = np.random.default_rng(seed)
    if points.shape[0] > max_samples:
        sample = points[rng.choice(points.shape[0], size=max_samples, replace=False)]
    else:
        sample = points

    best_normal: FloatArray | None = None
    best_point: FloatArray | None = None
    best_mask: npt.NDArray[np.bool_] | None = None
    best_count = 0
    for _ in range(iterations):
        selected = sample[rng.choice(sample.shape[0], size=3, replace=False)]
        normal = np.cross(selected[1] - selected[0], selected[2] - selected[0])
        length = float(np.linalg.norm(normal))
        if length <= np.finfo(np.float64).eps:
            continue
        normal /= length
        distances = np.abs((sample - selected[0]) @ normal)
        mask = distances <= threshold
        count = int(np.count_nonzero(mask))
        if count > best_count:
            best_count = count
            best_normal = normal
            best_point = selected[0]
            best_mask = mask

    if best_normal is None or best_point is None or best_mask is None:
        raise ValueError("Support plane fitting failed: no non-degenerate plane was found")
    fraction = best_count / sample.shape[0]
    if fraction < min_inlier_fraction:
        raise ValueError(
            "Support plane inlier fraction "
            f"{fraction:.6f} is below minimum {min_inlier_fraction:.6f}; "
            "provide a manual support plane"
        )

    inliers = sample[best_mask]
    point = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - point, full_matrices=False)
    normal = np.asarray(vh[-1], dtype=np.float64)
    normal /= np.linalg.norm(normal)

    signed = (sample - point) @ normal
    positive = int(np.count_nonzero(signed > threshold))
    negative = int(np.count_nonzero(signed < -threshold))
    if negative > positive:
        normal = -normal

    refined_distances = np.abs((sample - point) @ normal)
    refined_count = int(np.count_nonzero(refined_distances <= threshold))
    refined_fraction = refined_count / sample.shape[0]
    if refined_fraction < min_inlier_fraction:
        raise ValueError(
            "Refined support plane inlier fraction "
            f"{refined_fraction:.6f} is below minimum {min_inlier_fraction:.6f}; "
            "provide a manual support plane"
        )
    return PlaneFit(
        point=(float(point[0]), float(point[1]), float(point[2])),
        normal=(float(normal[0]), float(normal[1]), float(normal[2])),
        threshold_source_units=threshold,
        inlier_count=refined_count,
        sample_count=sample.shape[0],
        inlier_fraction=refined_fraction,
    )


def rotation_to_positive_z(normal: FloatArray) -> FloatArray:
    source = np.asarray(normal, dtype=np.float64)
    source /= np.linalg.norm(source)
    target = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if np.isclose(dot, 1.0):
        return np.eye(3, dtype=np.float64)
    if np.isclose(dot, -1.0):
        return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    cross = np.cross(source, target)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3) + skew + (skew @ skew) * (1.0 / (1.0 + dot))


def apply_transform(vertices: FloatArray, matrix: FloatArray) -> FloatArray:
    points = np.asarray(vertices, dtype=np.float64)
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("Transform matrix must have shape (4, 4)")
    homogeneous = np.column_stack((points, np.ones(points.shape[0], dtype=np.float64)))
    return (homogeneous @ transform.T)[:, :3]


def prepare_visual_mesh(
    input_ply: Path,
    output_obj: Path,
    output_mtl: Path,
    transform_json: Path,
    settings: GeometrySettings,
) -> dict[str, Any]:
    import fast_simplification  # type: ignore[import-untyped]
    import trimesh
    from trimesh.visual.material import SimpleMaterial
    from trimesh.visual.texture import TextureVisuals

    loaded = trimesh.load_mesh(input_ply, process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Expected a single triangular mesh in {input_ply}")
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    if vertices.shape[0] == 0 or faces.shape[0] == 0 or faces.shape[1] != 3:
        raise ValueError(f"Input mesh is empty or non-triangular: {input_ply}")
    input_bbox = _bbox(vertices)
    input_max_extent = max(input_bbox["extents"])
    preliminary_scale = normalize_scale(input_max_extent, settings.target_extent_m)
    threshold_source = settings.plane_threshold_m / preliminary_scale
    plane = fit_support_plane(
        vertices,
        threshold=threshold_source,
        min_inlier_fraction=settings.plane_min_inlier_fraction,
        seed=0,
    )
    plane_point = np.asarray(plane.point, dtype=np.float64)
    rotation = rotation_to_positive_z(np.asarray(plane.normal, dtype=np.float64))
    rotated = (vertices - plane_point) @ rotation.T
    rotated_bbox = _bbox(rotated)
    scale = normalize_scale(max(rotated_bbox["extents"]), settings.target_extent_m)
    transformed = rotated * scale

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation * scale
    matrix[:3, 3] = -(rotation @ plane_point) * scale
    output_bbox = _bbox(transformed)
    if not np.isclose(max(output_bbox["extents"]), settings.target_extent_m, atol=1e-8):
        raise ValueError("Normalized mesh extent does not match geometry.target_extent_m")

    simplified_vertices = transformed
    simplified_faces = faces
    if faces.shape[0] > settings.visual_face_count:
        simplified_vertices, simplified_faces = fast_simplification.simplify(
            transformed,
            faces,
            target_count=settings.visual_face_count,
            agg=10.0,
        )
    if simplified_faces.shape[0] > settings.visual_face_count:
        raise ValueError(
            f"Visual simplification produced {simplified_faces.shape[0]} faces, "
            f"above limit {settings.visual_face_count}"
        )

    colors = np.asarray(getattr(loaded.visual, "vertex_colors", None))
    if colors.ndim != 2 or colors.shape[0] != vertices.shape[0] or colors.shape[1] < 3:
        raise ValueError(f"Input mesh does not expose RGB vertex colors: {input_ply}")
    mean_rgb = np.rint(colors[:, :3].astype(np.float64).mean(axis=0)).astype(np.uint8)
    visual_mesh = trimesh.Trimesh(
        vertices=simplified_vertices,
        faces=simplified_faces,
        process=False,
    )
    visual_mesh.visual = TextureVisuals(  # type: ignore[no-untyped-call]
        material=SimpleMaterial(  # type: ignore[no-untyped-call]
            diffuse=[*mean_rgb.tolist(), 255]
        )
    )
    obj_text, resources = trimesh.exchange.obj.export_obj(  # type: ignore[no-untyped-call]
        visual_mesh,
        include_normals=False,
        include_color=False,
        include_texture=True,
        return_texture=True,
        mtl_name=output_mtl.name,
        header=None,
    )
    mtl_bytes = resources.get(output_mtl.name)
    if not isinstance(mtl_bytes, bytes):
        raise ValueError("OBJ export did not produce the expected MTL resource")
    output_obj.parent.mkdir(parents=True, exist_ok=True)
    output_obj.write_text(obj_text, encoding="ascii")
    output_mtl.write_bytes(mtl_bytes)

    result = {
        "schema": "real2sim.physics_transform.v1",
        "input_mesh": str(input_ply.resolve()),
        "scale_mode": settings.scale_mode,
        "target_extent_m": settings.target_extent_m,
        "preliminary_scale_m_per_unit": preliminary_scale,
        "scale_m_per_unit": scale,
        "rotation_matrix": rotation.tolist(),
        "translation_source_units": (-plane_point).tolist(),
        "transform_matrix": matrix.tolist(),
        "support_plane": plane.to_dict(),
        "input_bbox": input_bbox,
        "rotated_bbox_before_scale": rotated_bbox,
        "output_bbox": output_bbox,
        "input_stats": {
            "vertex_count": int(vertices.shape[0]),
            "face_count": int(faces.shape[0]),
        },
        "output_stats": {
            "vertex_count": int(simplified_vertices.shape[0]),
            "face_count": int(simplified_faces.shape[0]),
        },
        "mean_rgb": mean_rgb.tolist(),
    }
    transform_json.parent.mkdir(parents=True, exist_ok=True)
    transform_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _bbox(vertices: FloatArray) -> dict[str, list[float]]:
    minimum = np.min(vertices, axis=0)
    maximum = np.max(vertices, axis=0)
    return {
        "min": [float(value) for value in minimum],
        "max": [float(value) for value in maximum],
        "extents": [float(value) for value in maximum - minimum],
    }
