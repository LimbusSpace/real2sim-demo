from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    root_value = os.environ.get("REAL2SIM_HYWORLD_ROOT")
    if not root_value:
        raise RuntimeError("REAL2SIM_HYWORLD_ROOT is required.")
    root = Path(root_value).resolve()
    worldgen_dir = root / "hyworld2" / "worldgen"
    trainer = Path(
        os.environ.get("REAL2SIM_HYWORLD_TRAINER", str(worldgen_dir / "world_gs_trainer.py"))
    ).resolve()
    gsplat_dir = Path(
        os.environ.get(
            "REAL2SIM_HYWORLD_GSPLAT",
            str(worldgen_dir / "third_party" / "gsplat_maskgaussian"),
        )
    ).resolve()
    for name, path in (("trainer", trainer), ("gsplat", gsplat_dir)):
        if not path.exists():
            raise FileNotFoundError(f"HY-World {name} path does not exist: {path}")

    # Load CUDA torch from the selected environment before any optional dependency overlay.
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"HY-World requires CUDA; loaded torch={torch.__version__}")
    sys.path.insert(0, str(gsplat_dir))
    sys.path.insert(0, str(worldgen_dir))
    dependency_overlay = os.environ.get("REAL2SIM_HYWORLD_DEPS")
    if dependency_overlay:
        overlay_path = Path(dependency_overlay).resolve()
        if not overlay_path.is_dir():
            raise FileNotFoundError(f"HY-World dependency overlay does not exist: {overlay_path}")
        sys.path.append(str(overlay_path))
    runpy.run_path(str(trainer), run_name="__main__")


if __name__ == "__main__":
    main()
