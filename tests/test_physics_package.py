from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import pytest


def test_checked_in_physics_package_is_complete_and_runnable() -> None:
    package_dir = Path(__file__).resolve().parents[1] / "examples" / "tabletop_v1_physics"
    manifest = json.loads((package_dir / "package_manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema"] == "real2sim.physics_package.v1"
    for record in manifest["files"]:
        path = package_dir / record["path"]
        assert path.is_file()
        assert path.stat().st_size == record["size"]
        assert _sha256(path) == record["sha256"]

    model = mujoco.MjModel.from_xml_path(str(package_dir / "scene.xml"))
    data = mujoco.MjData(model)
    for _ in range(10):
        mujoco.mj_step(model, data)

    assert model.nbody == 2
    assert model.njnt == 0
    assert model.ngeom == 65
    assert model.nmesh == 65
    assert data.time == pytest.approx(0.02)
    video_header = (package_dir / "video" / "scene_preview.mp4").read_bytes()[:16]
    assert b"ftyp" in video_header


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
