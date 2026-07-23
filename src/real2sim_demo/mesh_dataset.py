from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path, PurePosixPath
from typing import Any

from .colmap_text import read_registered_images


def prepare_2dgs_dataset(input_run_dir: Path, dataset_dir: Path) -> dict[str, Any]:
    manifest_path = input_run_dir / "manifest.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "real2sim.stage1.manifest.v1":
        raise ValueError(f"Unsupported Stage 1 manifest schema: {manifest.get('schema')}")
    if manifest.get("stage") not in {"prepared", "gaussian_trained"}:
        raise ValueError(
            "Stage 1 must be prepared or gaussian_trained before Stage 2; "
            f"got {manifest.get('stage')!r}"
        )

    backend = str(manifest.get("reconstruction", {}).get("backend", "colmap")).lower()
    if backend == "mast3r":
        source_images = input_run_dir / "frames"
        source_model = input_run_dir / "mast3r" / "model_txt"
        model_format = "text"
        model_files = _TEXT_MODEL_FILES
        registered_names = read_colmap_text_image_names(source_model / "images.txt")
    elif backend == "colmap":
        source_root = input_run_dir / "colmap" / "undistorted"
        source_images = source_root / "images"
        source_model = source_root / "sparse"
        model_format = "binary"
        model_files = _BINARY_MODEL_FILES
        registered_names = read_colmap_image_names(source_model / "images.bin")
    else:
        raise ValueError(f"Unsupported Stage 1 SfM backend: {backend!r}")

    required_model = [source_model / name for name in model_files]
    missing = [str(path) for path in required_model if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Stage 1 SfM model files: " + ", ".join(missing))

    if len(registered_names) < 2:
        raise ValueError(f"SfM model has only {len(registered_names)} registered images")

    destination_images = dataset_dir / "images"
    destination_model = dataset_dir / "sparse" / "0"
    destination_images.mkdir(parents=True, exist_ok=True)
    destination_model.mkdir(parents=True, exist_ok=True)
    copied_images: list[str] = []
    for image_name in registered_names:
        relative = _safe_image_path(image_name)
        source = source_images.joinpath(*relative.parts)
        if not source.is_file():
            source = source_images / relative.name
        if not source.is_file():
            raise FileNotFoundError(f"Registered COLMAP image is missing: {image_name}")
        destination = destination_images.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied_images.append(str(destination.resolve()))

    for name in model_files:
        shutil.copy2(source_model / name, destination_model / name)

    actual_images = [path for path in destination_images.rglob("*") if path.is_file()]
    if len(actual_images) != len(registered_names):
        raise ValueError(
            f"Adapted image count mismatch: registered={len(registered_names)}, "
            f"dataset={len(actual_images)}. Remove stale files from {destination_images}."
        )

    provenance: dict[str, Any] = {
        "schema": "real2sim.2dgs_dataset.v1",
        "source_stage1_manifest": str(manifest_path.resolve()),
        "source_stage1_stage": manifest["stage"],
        "source_sfm_backend": backend,
        "source_images": str(source_images.resolve()),
        "source_model": str(source_model.resolve()),
        "model_format": model_format,
        "model_files": list(model_files),
        "dataset": str(dataset_dir.resolve()),
        "registered_image_count": len(registered_names),
        "registered_images": registered_names,
        "images": copied_images,
    }
    (dataset_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    return provenance


def validate_prepared_dataset(dataset_dir: Path) -> dict[str, Any]:
    provenance_path = dataset_dir / "provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(f"Missing {provenance_path}; run stage=prepare first")
    provenance: dict[str, Any] = json.loads(provenance_path.read_text(encoding="utf-8"))
    model_format = provenance.get("model_format", "binary")
    if model_format == "binary":
        model_files = _BINARY_MODEL_FILES
        registered_names = read_colmap_image_names(dataset_dir / "sparse" / "0" / "images.bin")
    elif model_format == "text":
        model_files = _TEXT_MODEL_FILES
        registered_names = read_colmap_text_image_names(
            dataset_dir / "sparse" / "0" / "images.txt"
        )
    else:
        raise ValueError(f"Unsupported prepared SfM model format: {model_format!r}")
    for name in model_files:
        if not (dataset_dir / "sparse" / "0" / name).is_file():
            raise FileNotFoundError(dataset_dir / "sparse" / "0" / name)
    missing_images = [
        name
        for name in registered_names
        if not dataset_dir.joinpath("images", *_safe_image_path(name).parts).is_file()
    ]
    if missing_images:
        raise FileNotFoundError("Adapted dataset is missing images: " + ", ".join(missing_images))
    if provenance.get("registered_image_count") != len(registered_names):
        raise ValueError("Adapted dataset provenance image count does not match COLMAP")
    return provenance


def read_colmap_text_image_names(path: Path) -> list[str]:
    images = read_registered_images(path)
    return [image.name for image in images.values()]


def read_colmap_image_names(path: Path) -> list[str]:
    try:
        with path.open("rb") as handle:
            count = _read_struct(handle, "<Q", path)[0]
            names: list[str] = []
            for _ in range(count):
                _read_struct(handle, "<I7dI", path)
                name_bytes = bytearray()
                while True:
                    value = handle.read(1)
                    if not value:
                        raise ValueError(f"Truncated COLMAP image name in {path}")
                    if value == b"\0":
                        break
                    name_bytes.extend(value)
                    if len(name_bytes) > 32 * 1024:
                        raise ValueError(f"Unreasonably long COLMAP image name in {path}")
                try:
                    name = name_bytes.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError(f"Invalid UTF-8 COLMAP image name in {path}") from exc
                point_count = _read_struct(handle, "<Q", path)[0]
                point_bytes = point_count * 24
                point_payload = handle.read(point_bytes)
                if len(point_payload) != point_bytes:
                    raise ValueError(f"Truncated COLMAP points2D data in {path}")
                names.append(name)
    except struct.error as exc:
        raise ValueError(f"Malformed COLMAP images.bin: {path}") from exc
    if len(names) != count:
        raise ValueError(f"COLMAP image count mismatch in {path}")
    return names


def _read_struct(handle: Any, data_format: str, path: Path) -> tuple[Any, ...]:
    size = struct.calcsize(data_format)
    payload = handle.read(size)
    if len(payload) != size:
        raise ValueError(f"Truncated COLMAP binary model: {path}")
    return struct.unpack(data_format, payload)


def _safe_image_path(name: str) -> PurePosixPath:
    normalized = PurePosixPath(name.replace("\\", "/"))
    if normalized.is_absolute() or not normalized.parts or ".." in normalized.parts:
        raise ValueError(f"Unsafe COLMAP image path: {name}")
    return normalized


_BINARY_MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")
_TEXT_MODEL_FILES = ("cameras.txt", "images.txt", "points3D.txt")
