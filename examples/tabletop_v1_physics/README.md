# Tabletop Physics Demo

This directory is a self-contained Stage 3 asset package. `scene.xml` loads the visual mesh and
64 CoACD convex collision meshes using only paths relative to this directory. The final scene is
a static MuJoCo environment: it has no free joint and no object-level dynamics.

Rebuild this package from a validated local run:

```powershell
uv run python scripts/package_physics_example.py `
  --run-dir "$env:REAL2SIM_ASSETS\runs\tabletop_v1_physics" `
  --output-dir examples/tabletop_v1_physics
```

Validate the checked-in scene:

```powershell
uv run python -c "import mujoco; m=mujoco.MjModel.from_xml_path('examples/tabletop_v1_physics/scene.xml'); d=mujoco.MjData(m); [mujoco.mj_step(m,d) for _ in range(10)]; print(m.nbody, m.njnt, m.ngeom, d.time)"
```

Expected values are `nbody=2`, `njnt=0`, `ngeom=65`, and `time=0.02`.

The preview video is `video/scene_preview.mp4`. It is rendered from this scene with temporary
red probe spheres to demonstrate collision contacts; those probe bodies are not part of
`scene.xml`. Regenerate it with:

```powershell
uv run python scripts/render_physics_video.py `
  --scene examples/tabletop_v1_physics/scene.xml `
  --output examples/tabletop_v1_physics/video/scene_preview.mp4
```
