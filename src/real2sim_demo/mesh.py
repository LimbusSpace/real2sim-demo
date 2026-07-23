from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2, rmtree
from typing import Any

from PIL import Image, ImageStat

from .mesh_config import DIFF_SURFEL_REVISION, SIMPLE_KNN_REVISION, TwoDGSSettings
from .mesh_ply import PlyStats, validate_mesh_pair, validate_surfel_ply
from .process import CommandResult, resolve_executable, run_command
from .snapshot import verify_snapshot


@dataclass(frozen=True, slots=True)
class MeshArtifacts:
    raw: Path
    post: Path
    raw_stats: PlyStats
    post_stats: PlyStats


def build_train_command(
    settings: TwoDGSSettings, dataset_dir: Path, model_dir: Path
) -> list[str | Path]:
    return [
        settings.python,
        settings.root / "train.py",
        "--source_path",
        dataset_dir,
        "--model_path",
        model_dir,
        "--iterations",
        str(settings.iterations),
        "--resolution",
        str(settings.resolution),
        "--sh_degree",
        str(settings.sh_degree),
        "--depth_ratio",
        str(settings.depth_ratio),
        "--lambda_normal",
        str(settings.lambda_normal),
        "--lambda_dist",
        str(settings.lambda_dist),
        "--save_iterations",
        str(settings.iterations),
        "--quiet",
    ]


def build_mesh_command(
    settings: TwoDGSSettings,
    dataset_dir: Path,
    model_dir: Path,
    *,
    mesh_res: int,
    num_clusters: int,
    depth_trunc: float | None,
    voxel_size: float | None,
    sdf_trunc: float | None,
) -> list[str | Path]:
    command: list[str | Path] = [
        settings.python,
        settings.root / "render.py",
        "--source_path",
        dataset_dir,
        "--model_path",
        model_dir,
        "--iteration",
        str(settings.iterations),
        "--skip_train",
        "--skip_test",
        "--depth_ratio",
        str(settings.depth_ratio),
        "--mesh_res",
        str(mesh_res),
        "--num_cluster",
        str(num_clusters),
        "--quiet",
    ]
    for name, value in (
        ("depth_trunc", depth_trunc),
        ("voxel_size", voxel_size),
        ("sdf_trunc", sdf_trunc),
    ):
        if value is not None:
            command.extend([f"--{name}", str(value)])
    return command


def build_preview_command(
    python: str, script: Path, mesh_path: Path, output_path: Path
) -> list[str | Path]:
    return [python, script, "--mesh", mesh_path, "--output", output_path]


def build_diagnostics_render_command(
    settings: TwoDGSSettings, dataset_dir: Path, model_dir: Path
) -> list[str | Path]:
    return [
        settings.python,
        settings.root / "render.py",
        "--source_path",
        dataset_dir,
        "--model_path",
        model_dir,
        "--iteration",
        str(settings.iterations),
        "--skip_test",
        "--skip_mesh",
        "--depth_ratio",
        str(settings.depth_ratio),
        "--quiet",
    ]


def build_diagnostics_triptych_command(
    python: str,
    gt_dir: Path,
    render_dir: Path,
    depth_dir: Path,
    output_dir: Path,
    camera_map: Path,
) -> list[str | Path]:
    return [
        python,
        "-m",
        "real2sim_demo.mesh_diagnostics",
        "--gt-dir",
        gt_dir,
        "--render-dir",
        render_dir,
        "--depth-dir",
        depth_dir,
        "--output-dir",
        output_dir,
        "--camera-map",
        camera_map,
    ]


def train_2dgs(
    settings: TwoDGSSettings,
    dataset_dir: Path,
    model_dir: Path,
    logs_dir: Path,
    *,
    dry_run: bool = False,
) -> tuple[Path | None, CommandResult | None, bool]:
    expected = _train_provenance(settings, dataset_dir, model_dir)
    output_ply = model_dir / "point_cloud" / f"iteration_{settings.iterations}" / "point_cloud.ply"
    provenance_path = model_dir / "provenance.json"
    if not dry_run and _provenance_matches(provenance_path, expected) and output_ply.is_file():
        validate_surfel_ply(output_ply)
        return output_ply, None, True

    if not dry_run:
        verify_2dgs_source(settings.root, settings.source_revision)
    python = settings.python if dry_run else resolve_executable(settings.python)
    configured = TwoDGSSettings(
        python=python,
        root=settings.root,
        source_revision=settings.source_revision,
        iterations=settings.iterations,
        resolution=settings.resolution,
        sh_degree=settings.sh_degree,
        depth_ratio=settings.depth_ratio,
        lambda_normal=settings.lambda_normal,
        lambda_dist=settings.lambda_dist,
    )
    result = run_command(
        build_train_command(configured, dataset_dir, model_dir),
        logs_dir / "08_2dgs_train.log",
        cwd=settings.root if not dry_run else None,
        env={"PYTHONUTF8": "1"},
        dry_run=dry_run,
    )
    if dry_run:
        return None, result, False
    validate_surfel_ply(output_ply)
    _write_json(provenance_path, expected | {"surfel_ply": str(output_ply.resolve())})
    return output_ply, result, False


def export_mesh(
    settings: TwoDGSSettings,
    dataset_dir: Path,
    model_dir: Path,
    mesh_dir: Path,
    logs_dir: Path,
    *,
    mesh_res: int,
    num_clusters: int,
    depth_trunc: float | None,
    voxel_size: float | None,
    sdf_trunc: float | None,
    dry_run: bool = False,
) -> tuple[MeshArtifacts | None, CommandResult | None, bool]:
    expected = _mesh_provenance(
        settings,
        dataset_dir,
        model_dir,
        mesh_res=mesh_res,
        num_clusters=num_clusters,
        depth_trunc=depth_trunc,
        voxel_size=voxel_size,
        sdf_trunc=sdf_trunc,
    )
    raw = mesh_dir / "raw.ply"
    post = mesh_dir / "post.ply"
    provenance_path = mesh_dir / "provenance.json"
    if (
        not dry_run
        and _provenance_matches(provenance_path, expected)
        and raw.is_file()
        and post.is_file()
    ):
        raw_stats, post_stats = validate_mesh_pair(raw, post)
        return MeshArtifacts(raw, post, raw_stats, post_stats), None, True

    if not dry_run:
        verify_2dgs_source(settings.root, settings.source_revision)
    python = settings.python if dry_run else resolve_executable(settings.python)
    configured = TwoDGSSettings(
        python=python,
        root=settings.root,
        source_revision=settings.source_revision,
        iterations=settings.iterations,
        resolution=settings.resolution,
        sh_degree=settings.sh_degree,
        depth_ratio=settings.depth_ratio,
        lambda_normal=settings.lambda_normal,
        lambda_dist=settings.lambda_dist,
    )
    result = run_command(
        build_mesh_command(
            configured,
            dataset_dir,
            model_dir,
            mesh_res=mesh_res,
            num_clusters=num_clusters,
            depth_trunc=depth_trunc,
            voxel_size=voxel_size,
            sdf_trunc=sdf_trunc,
        ),
        logs_dir / "09_2dgs_mesh.log",
        cwd=settings.root if not dry_run else None,
        env={"PYTHONUTF8": "1"},
        dry_run=dry_run,
    )
    if dry_run:
        return None, result, False

    upstream_dir = model_dir / "train" / f"ours_{settings.iterations}"
    upstream_raw = upstream_dir / "fuse.ply"
    upstream_post = upstream_dir / "fuse_post.ply"
    if not upstream_raw.is_file() or not upstream_post.is_file():
        raise FileNotFoundError(
            f"2DGS finished without bounded mesh outputs: {upstream_raw}, {upstream_post}"
        )
    mesh_dir.mkdir(parents=True, exist_ok=True)
    copy2(upstream_raw, raw)
    copy2(upstream_post, post)
    raw_stats, post_stats = validate_mesh_pair(raw, post)
    _write_json(
        provenance_path,
        expected
        | {
            "raw": str(raw.resolve()),
            "post": str(post.resolve()),
            "raw_stats": raw_stats.to_dict(),
            "post_stats": post_stats.to_dict(),
        },
    )
    return MeshArtifacts(raw, post, raw_stats, post_stats), result, False


def create_preview(
    python: str,
    preview_script: Path,
    mesh_path: Path,
    output_path: Path,
    logs_dir: Path,
    *,
    allow_reuse: bool = True,
    dry_run: bool = False,
) -> tuple[CommandResult, bool]:
    preview_provenance = output_path.with_name("preview.provenance.json")
    expected = {
        "schema": "real2sim.mesh_preview.v1",
        "mesh": str(mesh_path.resolve()),
        "mesh_bytes": mesh_path.stat().st_size if mesh_path.is_file() else None,
        "mesh_mtime_ns": mesh_path.stat().st_mtime_ns if mesh_path.is_file() else None,
        "script": str(preview_script.resolve()),
        "script_sha256": _sha256(preview_script) if preview_script.is_file() else None,
    }
    if (
        not dry_run
        and allow_reuse
        and output_path.is_file()
        and _provenance_matches(preview_provenance, expected)
    ):
        validate_preview(output_path)
        return CommandResult([], 0, logs_dir / "10_mesh_preview.log"), True
    executable = python if dry_run else resolve_executable(python)
    result = run_command(
        build_preview_command(executable, preview_script, mesh_path, output_path),
        logs_dir / "10_mesh_preview.log",
        cwd=preview_script.parent if not dry_run else None,
        env={"PYTHONUTF8": "1"},
        dry_run=dry_run,
    )
    if not dry_run:
        validate_preview(output_path)
        _write_json(preview_provenance, expected | {"preview": str(output_path.resolve())})
    return result, False


def create_diagnostics(
    settings: TwoDGSSettings,
    dataset_dir: Path,
    model_dir: Path,
    diagnostics_dir: Path,
    logs_dir: Path,
    *,
    composer_python: str,
    dry_run: bool = False,
) -> tuple[
    dict[str, Any] | None,
    CommandResult | None,
    CommandResult | None,
    bool,
]:
    upstream_dir = model_dir / "train" / f"ours_{settings.iterations}"
    gt_dir = upstream_dir / "gt"
    render_dir = upstream_dir / "renders"
    depth_dir = upstream_dir / "vis"
    report_path = diagnostics_dir / "report.json"
    provenance_path = diagnostics_dir / "provenance.json"
    expected = _diagnostics_provenance(settings, dataset_dir, model_dir)
    if not dry_run and _provenance_matches(provenance_path, expected) and report_path.is_file():
        return validate_diagnostics(report_path), None, None, True

    if not dry_run:
        verify_2dgs_source(settings.root, settings.source_revision)
        checkpoint = (
            model_dir / "point_cloud" / f"iteration_{settings.iterations}" / "point_cloud.ply"
        )
        validate_surfel_ply(checkpoint)
        for path in (gt_dir, render_dir, depth_dir):
            if path.is_dir():
                rmtree(path)

    twodgs_python = settings.python if dry_run else resolve_executable(settings.python)
    configured = TwoDGSSettings(
        python=twodgs_python,
        root=settings.root,
        source_revision=settings.source_revision,
        iterations=settings.iterations,
        resolution=settings.resolution,
        sh_degree=settings.sh_degree,
        depth_ratio=settings.depth_ratio,
        lambda_normal=settings.lambda_normal,
        lambda_dist=settings.lambda_dist,
    )
    render_result = run_command(
        build_diagnostics_render_command(configured, dataset_dir, model_dir),
        logs_dir / "11_2dgs_diagnostics_render.log",
        cwd=settings.root if not dry_run else None,
        env={"PYTHONUTF8": "1"},
        dry_run=dry_run,
    )
    executable = composer_python if dry_run else resolve_executable(composer_python)
    triptych_result = run_command(
        build_diagnostics_triptych_command(
            executable,
            gt_dir,
            render_dir,
            depth_dir,
            diagnostics_dir,
            model_dir / "cameras.json",
        ),
        logs_dir / "12_2dgs_diagnostics_triptych.log",
        cwd=Path(__file__).resolve().parents[2] if not dry_run else None,
        env={"PYTHONUTF8": "1"},
        dry_run=dry_run,
    )
    if dry_run:
        return None, render_result, triptych_result, False

    report = validate_diagnostics(report_path)
    _write_json(
        provenance_path,
        expected
        | {
            "report": str(report_path.resolve()),
            "contact_sheet": report["contact_sheet"],
            "triptych_dir": report["triptych_dir"],
            "frame_count": report["frame_count"],
        },
    )
    return report, render_result, triptych_result, False


def validate_preview(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Mesh preview was not created: {path}")
    with Image.open(path) as image:
        if image.width < 3 or image.height < 1:
            raise ValueError(f"Mesh preview has invalid dimensions: {path}")
        extrema = ImageStat.Stat(image.convert("RGB")).extrema
    if all(low == high for low, high in extrema):
        raise ValueError(f"Mesh preview is blank: {path}")


def validate_diagnostics(report_path: Path) -> dict[str, Any]:
    if not report_path.is_file():
        raise FileNotFoundError(f"2DGS diagnostics report was not created: {report_path}")
    payload: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "real2sim.2dgs_diagnostics.v1":
        raise ValueError(f"Unsupported 2DGS diagnostics report: {report_path}")
    frame_count = int(payload.get("frame_count", 0))
    triptych_dir = Path(str(payload.get("triptych_dir", "")))
    contact_sheet = Path(str(payload.get("contact_sheet", "")))
    triptychs = sorted(triptych_dir.glob("*.jpg")) if triptych_dir.is_dir() else []
    if frame_count <= 0 or len(triptychs) != frame_count:
        raise ValueError(
            f"2DGS diagnostics frame count mismatch: report={frame_count}, files={len(triptychs)}"
        )
    validate_preview(contact_sheet)
    validate_preview(triptychs[0])
    validate_preview(triptychs[-1])
    return payload


def verify_2dgs_source(root: Path, expected_revision: str) -> dict[str, str]:
    required = [root / "train.py", root / "render.py", root / "submodules" / "simple-knn"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("2DGS source is incomplete: " + ", ".join(missing))
    revisions = {
        "2dgs": _git_revision(root),
        "diff_surfel_rasterization": _git_revision(
            root / "submodules" / "diff-surfel-rasterization"
        ),
        "simple_knn": _git_revision(root / "submodules" / "simple-knn"),
    }
    expected = {
        "2dgs": expected_revision,
        "diff_surfel_rasterization": DIFF_SURFEL_REVISION,
        "simple_knn": SIMPLE_KNN_REVISION,
    }
    mismatch = [
        f"{name}={revisions[name]} expected={value}"
        for name, value in expected.items()
        if revisions[name] != value
    ]
    if mismatch:
        raise RuntimeError("2DGS source revision mismatch: " + "; ".join(mismatch))
    snapshot_path = Path(__file__).resolve().parents[2] / "reproducibility" / "2dgs.snapshot.json"
    if snapshot_path.is_file():
        failures = verify_snapshot(snapshot_path, root)
        if failures:
            raise RuntimeError("2DGS source snapshot verification failed: " + "; ".join(failures))
    return revisions


def source_versions(root: Path, expected_revision: str, *, dry_run: bool) -> dict[str, str]:
    if dry_run:
        return {
            "2dgs": expected_revision,
            "diff_surfel_rasterization": DIFF_SURFEL_REVISION,
            "simple_knn": SIMPLE_KNN_REVISION,
        }
    return verify_2dgs_source(root, expected_revision)


def _git_revision(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Unable to read Git revision for {path}: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _train_provenance(
    settings: TwoDGSSettings, dataset_dir: Path, model_dir: Path
) -> dict[str, Any]:
    return {
        "schema": "real2sim.2dgs_training.v1",
        "source_root": str(settings.root.resolve()),
        "source_revision": settings.source_revision,
        "dataset": str(dataset_dir.resolve()),
        "model_dir": str(model_dir.resolve()),
        "iterations": settings.iterations,
        "resolution": settings.resolution,
        "sh_degree": settings.sh_degree,
        "depth_ratio": settings.depth_ratio,
        "lambda_normal": settings.lambda_normal,
        "lambda_dist": settings.lambda_dist,
    }


def _mesh_provenance(
    settings: TwoDGSSettings,
    dataset_dir: Path,
    model_dir: Path,
    **mesh: Any,
) -> dict[str, Any]:
    surfel_ply = model_dir / "point_cloud" / f"iteration_{settings.iterations}" / "point_cloud.ply"
    return {
        "schema": "real2sim.2dgs_mesh.v1",
        "source_root": str(settings.root.resolve()),
        "source_revision": settings.source_revision,
        "dataset": str(dataset_dir.resolve()),
        "model_dir": str(model_dir.resolve()),
        "iterations": settings.iterations,
        "surfel_ply": str(surfel_ply.resolve()),
        "surfel_sha256": _sha256(surfel_ply) if surfel_ply.is_file() else None,
        "depth_ratio": settings.depth_ratio,
        "mode": "bounded",
        **mesh,
    }


def _diagnostics_provenance(
    settings: TwoDGSSettings, dataset_dir: Path, model_dir: Path
) -> dict[str, Any]:
    checkpoint = model_dir / "point_cloud" / f"iteration_{settings.iterations}" / "point_cloud.ply"
    composer = Path(__file__).with_name("mesh_diagnostics.py")
    render_script = settings.root / "render.py"
    return {
        "schema": "real2sim.2dgs_diagnostics_provenance.v1",
        "source_root": str(settings.root.resolve()),
        "source_revision": settings.source_revision,
        "dataset": str(dataset_dir.resolve()),
        "dataset_images_sha256": _directory_sha256(dataset_dir / "images"),
        "model_dir": str(model_dir.resolve()),
        "iterations": settings.iterations,
        "checkpoint_sha256": _sha256(checkpoint) if checkpoint.is_file() else None,
        "depth_ratio": settings.depth_ratio,
        "render_script_sha256": _sha256(render_script) if render_script.is_file() else None,
        "composer_sha256": _sha256(composer) if composer.is_file() else None,
        "camera_map_sha256": _sha256(model_dir / "cameras.json")
        if (model_dir / "cameras.json").is_file()
        else None,
    }


def _provenance_matches(path: Path, expected: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(payload.get(key) == value for key, value in expected.items())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_sha256(path: Path) -> str | None:
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()
