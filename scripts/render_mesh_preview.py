from __future__ import annotations

import argparse
from pathlib import Path


def render_preview(
    mesh_path: Path, output_path: Path, *, width: int = 512, height: int = 512
) -> None:
    import numpy as np
    from PIL import Image
    from plyfile import PlyData

    data = PlyData.read(str(mesh_path))
    vertices = data["vertex"].data
    points = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(
        np.float32, copy=False
    )
    if len(points) == 0:
        raise ValueError(f"Mesh is empty: {mesh_path}")
    color_names = {name.lower() for name in vertices.dtype.names or ()}
    if not {"red", "green", "blue"}.issubset(color_names):
        raise ValueError(f"Mesh has no RGB vertex colors: {mesh_path}")
    colors = np.column_stack(
        (vertices["red"], vertices["green"], vertices["blue"])
    ).astype(np.uint8, copy=False)

    # A bounded TSDF can contain millions of vertices. A deterministic point sample is
    # sufficient for an inspection preview and avoids requiring a desktop OpenGL context.
    stride = max(1, int(np.ceil(len(points) / 180_000)))
    points = points[::stride]
    colors = colors[::stride]
    center = (points.min(axis=0) + points.max(axis=0)) * 0.5
    extent = max(float(np.ptp(points, axis=0).max()), 1e-6)
    views = [
        ("front", np.array([0.0, -1.0, 0.15]), np.array([0.0, 0.0, 1.0])),
        ("isometric", np.array([1.0, -1.0, 0.75]), np.array([0.0, 0.0, 1.0])),
        ("top", np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])),
    ]
    panels = [
        _project_panel(points, colors, center, extent, direction, up, label, width, height)
        for label, direction, up in views
    ]
    divider = Image.new("RGB", (4, height), (50, 55, 60))
    canvas = Image.new("RGB", (width * 3 + 8, height), (235, 238, 240))
    x = 0
    for index, panel in enumerate(panels):
        canvas.paste(panel, (x, 0))
        x += width
        if index < len(panels) - 1:
            canvas.paste(divider, (x, 0))
            x += divider.width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")


def _project_panel(
    points: object,
    colors: object,
    center: object,
    extent: float,
    direction: object,
    up: object,
    label: str,
    width: int,
    height: int,
) -> object:
    import numpy as np
    from PIL import Image, ImageDraw

    xyz = np.asarray(points, dtype=np.float32) - np.asarray(center, dtype=np.float32)
    view = np.asarray(direction, dtype=np.float32)
    view /= np.linalg.norm(view)
    world_up = np.asarray(up, dtype=np.float32)
    right = np.cross(view, world_up)
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, view)
    horizontal = xyz @ right
    vertical = xyz @ camera_up
    scale = max(extent * 0.55, 1e-6)
    px = np.clip(((horizontal / scale) * 0.5 + 0.5) * (width - 1), 0, width - 1)
    py = np.clip((0.5 - (vertical / scale) * 0.5) * (height - 1), 0, height - 1)
    depth = xyz @ view
    order = np.argsort(depth)
    rgb = np.asarray(colors, dtype=np.uint8)
    x = px[order].astype(np.int32)
    y = py[order].astype(np.int32)
    ordered_rgb = rgb[order]
    pixels = np.full((height, width, 3), (235, 238, 240), dtype=np.uint8)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        shifted_x = x + dx
        shifted_y = y + dy
        mask = (shifted_x < width) & (shifted_y < height)
        pixels[shifted_y[mask], shifted_x[mask]] = ordered_rgb[mask]
    panel = Image.fromarray(pixels, mode="RGB")
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, 92, 24), fill=(35, 40, 45))
    draw.text((8, 6), label, fill=(255, 255, 255))
    return panel


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render fixed front, isometric, and top mesh views."
    )
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    render_preview(args.mesh, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
