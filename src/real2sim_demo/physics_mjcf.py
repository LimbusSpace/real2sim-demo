from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .physics_collision import read_obj_stats


def build_obj2mjcf_command(executable: str, source_dir: Path) -> list[str]:
    return [
        executable,
        "--obj-dir",
        str(source_dir),
        "--obj-filter",
        r"^scene\.obj$",
        "--save-mjcf",
        "--no-compile-model",
        "--no-decompose",
        "--overwrite",
        "--no-add-free-joint",
    ]


def assemble_mjcf(
    obj2mjcf_dir: Path,
    source_mtl: Path,
    collision_dir: Path,
    output_dir: Path,
    *,
    gravity: tuple[float, float, float],
) -> dict[str, Any]:
    base_xml = obj2mjcf_dir / "scene.xml"
    processed_obj = obj2mjcf_dir / "scene.obj"
    if not base_xml.is_file() or not processed_obj.is_file():
        raise FileNotFoundError(
            f"obj2mjcf did not produce scene.xml and scene.obj in {obj2mjcf_dir}"
        )
    read_obj_stats(processed_obj)
    collisions = sorted(collision_dir.glob("scene_collision_*.obj"))
    if not collisions:
        raise ValueError(f"No collision meshes found in {collision_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_obj = output_dir / "scene.obj"
    output_mtl = output_dir / "scene.mtl"
    shutil.copy2(processed_obj, output_obj)
    shutil.copy2(source_mtl, output_mtl)
    _normalize_material_reference(output_obj, output_mtl.name)
    for stale in output_dir.glob("scene_collision_*.obj"):
        stale.unlink()
    for collision in collisions:
        shutil.copy2(collision, output_dir / collision.name)

    tree = ET.parse(base_xml)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler", {"angle": "radian", "meshdir": "."})
        root.insert(0, compiler)
    else:
        compiler.set("angle", "radian")
        compiler.set("meshdir", ".")
    option = root.find("option")
    gravity_text = " ".join(_number(value) for value in gravity)
    if option is None:
        option = ET.Element("option", {"gravity": gravity_text})
        root.insert(1, option)
    else:
        option.set("gravity", gravity_text)

    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError("obj2mjcf base XML is missing asset or worldbody")
    bodies = worldbody.findall("body")
    if not bodies:
        raise ValueError("obj2mjcf base XML contains no body")
    body = bodies[0]
    if body.find("joint") is not None or body.find("freejoint") is not None:
        raise ValueError("obj2mjcf unexpectedly generated a joint for the static environment")

    visual_geoms = []
    for geom in list(body.findall("geom")):
        if geom.get("class") == "collision":
            body.remove(geom)
            continue
        if geom.get("class") == "visual":
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            geom.set("group", "2")
            visual_geoms.append(geom)
    if not visual_geoms:
        raise ValueError("obj2mjcf base XML contains no visual geom")

    defaults = root.find("default")
    if defaults is None:
        defaults = ET.Element("default")
        root.insert(2, defaults)
    collision_default = None
    for default in defaults.findall("default"):
        if default.get("class") == "collision":
            collision_default = default
            break
    if collision_default is None:
        collision_default = ET.SubElement(defaults, "default", {"class": "collision"})
    default_geom = collision_default.find("geom")
    if default_geom is None:
        default_geom = ET.SubElement(collision_default, "geom")
    default_geom.attrib.update(
        {
            "type": "mesh",
            "group": "3",
            "contype": "1",
            "conaffinity": "1",
            "condim": "4",
            "friction": "1 0.005 0.0001",
        }
    )

    for collision in collisions:
        name = collision.stem
        ET.SubElement(asset, "mesh", {"name": name, "file": collision.name})
        ET.SubElement(
            body,
            "geom",
            {
                "name": name,
                "class": "collision",
                "mesh": name,
                "contype": "1",
                "conaffinity": "1",
                "condim": "4",
                "friction": "1 0.005 0.0001",
            },
        )

    output_xml = output_dir / "scene.xml"
    ET.indent(tree, space="  ")
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    structure = validate_mjcf_structure(output_xml)
    return {
        "scene_xml": str(output_xml.resolve()),
        "visual_obj": str(output_obj.resolve()),
        "visual_mtl": str(output_mtl.resolve()),
        "collision_meshes": [str((output_dir / path.name).resolve()) for path in collisions],
        **structure,
    }


def validate_mjcf_structure(xml_path: Path) -> dict[str, Any]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError("MJCF must contain asset and worldbody elements")
    mesh_names: set[str] = set()
    mesh_files: list[str] = []
    for mesh in asset.findall("mesh"):
        filename = mesh.get("file")
        if not filename:
            raise ValueError("MJCF mesh asset is missing a file attribute")
        relative = Path(filename)
        if relative.is_absolute():
            raise ValueError(f"MJCF mesh path must be relative: {filename}")
        resolved = (xml_path.parent / relative).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"MJCF mesh asset does not exist: {resolved}")
        name = mesh.get("name") or relative.stem
        if name in mesh_names:
            raise ValueError(f"Duplicate MJCF mesh asset name: {name}")
        mesh_names.add(name)
        mesh_files.append(filename)

    bodies = worldbody.findall(".//body")
    if not bodies:
        raise ValueError("MJCF contains no static environment body")
    for body in bodies:
        if body.find("joint") is not None or body.find("freejoint") is not None:
            raise ValueError("Stage 3 MJCF must not contain joints")
    visual_count = 0
    collision_count = 0
    for geom in worldbody.findall(".//geom"):
        mesh_name = geom.get("mesh")
        if mesh_name and mesh_name not in mesh_names:
            raise ValueError(f"MJCF geom references unknown mesh asset: {mesh_name}")
        if geom.get("class") == "visual":
            if geom.get("contype") != "0" or geom.get("conaffinity") != "0":
                raise ValueError("Visual geom must disable contact")
            visual_count += 1
        elif geom.get("class") == "collision":
            if geom.get("contype") != "1" or geom.get("conaffinity") != "1":
                raise ValueError("Collision geom must enable contact")
            if geom.get("condim") != "4" or geom.get("friction") != "1 0.005 0.0001":
                raise ValueError("Collision geom has unexpected deterministic contact defaults")
            collision_count += 1
    if visual_count <= 0 or collision_count <= 0:
        raise ValueError("MJCF must contain visual and collision geoms")
    return {
        "static_body_count": len(bodies),
        "visual_geom_count": visual_count,
        "collision_geom_count": collision_count,
        "mesh_asset_count": len(mesh_files),
        "mesh_files": mesh_files,
    }


def _normalize_material_reference(obj_path: Path, mtl_name: str) -> None:
    lines = obj_path.read_text(encoding="utf-8").splitlines()
    normalized = [line for line in lines if not line.startswith("mtllib ")]
    normalized.insert(0, f"mtllib {mtl_name}")
    obj_path.write_text("\n".join(normalized) + "\n", encoding="utf-8")


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)
