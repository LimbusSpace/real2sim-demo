from __future__ import annotations

import importlib.metadata
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from .physics_mjcf import validate_mjcf_structure


def validate_mujoco_scene(xml_path: Path, report_path: Path, *, steps: int = 10) -> dict[str, Any]:
    if steps < 10:
        raise ValueError("MuJoCo validation must execute at least 10 steps")
    structure = validate_mjcf_structure(xml_path)
    import mujoco  # type: ignore[import-untyped]

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if model.njnt != 0:
        raise ValueError(f"Expected a static MJCF with njnt=0, got {model.njnt}")
    if model.nbody <= 1:
        raise ValueError("Compiled MJCF contains no static body")
    if not np.all(model.body_jntnum[1:] == 0):
        raise ValueError("Compiled MJCF contains a non-static body")
    mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    visual_mask = (model.geom_contype == 0) & (model.geom_conaffinity == 0)
    collision_mask = (model.geom_contype == 1) & (model.geom_conaffinity == 1)
    if not np.any(visual_mask):
        raise ValueError("Compiled MJCF contains no visual geom")
    if not np.any(collision_mask):
        raise ValueError("Compiled MJCF contains no collision geom")
    if not np.all(model.geom_type[visual_mask | collision_mask] == mesh_type):
        raise ValueError("Stage 3 visual and collision geoms must use mesh assets")

    data = mujoco.MjData(model)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    tree = ET.parse(xml_path)
    option = tree.getroot().find("option")
    if option is None:
        raise ValueError("MJCF is missing the option element")
    gravity = [float(value) for value in option.get("gravity", "").split()]
    report = {
        "schema": "real2sim.physics_validation.v1",
        "success": True,
        "mujoco_version": importlib.metadata.version("mujoco"),
        "scene_xml": str(xml_path.resolve()),
        "steps": steps,
        "final_time": float(data.time),
        "gravity": gravity,
        "model": {
            "nbody": int(model.nbody),
            "njnt": int(model.njnt),
            "ngeom": int(model.ngeom),
            "nmesh": int(model.nmesh),
            "visual_geom_count": int(np.count_nonzero(visual_mask)),
            "collision_geom_count": int(np.count_nonzero(collision_mask)),
        },
        "structure": structure,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
