from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from real2sim_demo.physics_mjcf import assemble_mjcf, validate_mjcf_structure
from real2sim_demo.physics_validate import validate_mujoco_scene


def test_mjcf_assembly_injects_static_collision_meshes(tmp_path: Path) -> None:
    obj2_dir, source_mtl, collision_dir = _base_assets(tmp_path)

    result = assemble_mjcf(
        obj2_dir,
        source_mtl,
        collision_dir,
        tmp_path / "mjcf",
        gravity=(0.0, 0.0, -9.81),
    )
    structure = validate_mjcf_structure(Path(result["scene_xml"]))
    report = validate_mujoco_scene(
        Path(result["scene_xml"]), tmp_path / "validation.json", steps=10
    )

    assert structure["static_body_count"] == 1
    assert structure["visual_geom_count"] == 1
    assert structure["collision_geom_count"] == 1
    assert report["model"]["njnt"] == 0
    assert report["steps"] == 10


def test_mujoco_compile_failure_is_not_swallowed(tmp_path: Path) -> None:
    obj2_dir, source_mtl, collision_dir = _base_assets(tmp_path)
    result = assemble_mjcf(
        obj2_dir,
        source_mtl,
        collision_dir,
        tmp_path / "mjcf",
        gravity=(0.0, 0.0, -9.81),
    )
    xml_path = Path(result["scene_xml"])
    tree = ET.parse(xml_path)
    option = tree.getroot().find("option")
    assert option is not None
    option.set("integrator", "not-an-integrator")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    with pytest.raises(ValueError):
        validate_mujoco_scene(xml_path, tmp_path / "failed.json", steps=10)


def _base_assets(tmp_path: Path) -> tuple[Path, Path, Path]:
    obj2_dir = tmp_path / "source" / "scene"
    obj2_dir.mkdir(parents=True)
    tetra = (
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "v 0 0 1\n"
        "f 1 3 2\n"
        "f 1 2 4\n"
        "f 2 3 4\n"
        "f 3 1 4\n"
    )
    (obj2_dir / "scene.obj").write_text(tetra, encoding="ascii")
    (obj2_dir / "scene.xml").write_text(
        """<mujoco model="scene">
  <default>
    <default class="visual"><geom type="mesh" contype="0" conaffinity="0"/></default>
    <default class="collision"><geom type="mesh"/></default>
  </default>
  <asset>
    <material name="material_0" rgba="0.2 0.4 0.6 1"/>
    <mesh file="scene.obj"/>
  </asset>
  <worldbody>
    <body name="scene">
      <geom class="visual" mesh="scene" material="material_0"/>
      <geom class="collision" mesh="scene"/>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    source_mtl = tmp_path / "source" / "scene.mtl"
    source_mtl.write_text(
        "newmtl material_0\nKd 0.2 0.4 0.6\n", encoding="ascii"
    )
    collision_dir = tmp_path / "collision"
    collision_dir.mkdir()
    (collision_dir / "scene_collision_000.obj").write_text(tetra, encoding="ascii")
    return obj2_dir, source_mtl, collision_dir
