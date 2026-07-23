from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pinned CoACD convex decomposition.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--max-convex-hull", type=int, required=True)
    parser.add_argument("--preprocess-mode", required=True)
    parser.add_argument("--preprocess-resolution", type=int, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--mcts-nodes", type=int, required=True)
    parser.add_argument("--mcts-iterations", type=int, required=True)
    parser.add_argument("--mcts-max-depth", type=int, required=True)
    parser.add_argument("--max-ch-vertex", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--real-metric", action="store_true")
    parser.add_argument("--decimate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import coacd  # type: ignore[import-untyped]
    import trimesh

    loaded = trimesh.load_mesh(args.input, process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Expected a single mesh in {args.input}")
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int32)
    if vertices.shape[0] == 0 or faces.shape[0] == 0 or faces.shape[1] != 3:
        raise ValueError(f"Input mesh is empty or non-triangular: {args.input}")

    mesh = coacd.Mesh(vertices, faces)
    parts = coacd.run_coacd(
        mesh=mesh,
        threshold=args.threshold,
        max_convex_hull=args.max_convex_hull,
        preprocess_mode=args.preprocess_mode,
        preprocess_resolution=args.preprocess_resolution,
        resolution=args.resolution,
        mcts_nodes=args.mcts_nodes,
        mcts_iterations=args.mcts_iterations,
        mcts_max_depth=args.mcts_max_depth,
        decimate=args.decimate,
        max_ch_vertex=args.max_ch_vertex,
        seed=args.seed,
        real_metric=args.real_metric,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for old_path in args.output_dir.glob("scene_collision_*.obj"):
        old_path.unlink()
    for index, (part_vertices, part_faces) in enumerate(parts):
        part = trimesh.Trimesh(part_vertices, part_faces, process=False)
        part.export(args.output_dir / f"scene_collision_{index:03d}.obj")
    if not parts:
        raise RuntimeError("CoACD returned no convex parts")
    print(f"collision_parts={len(parts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
