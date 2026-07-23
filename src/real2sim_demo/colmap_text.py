from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from typing import Any

from PIL import Image


@dataclass(frozen=True, slots=True)
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RegisteredImage:
    image_id: int
    qvec_wxyz: tuple[float, float, float, float]
    tvec_xyz: tuple[float, float, float]
    camera_id: int
    name: str


@dataclass(frozen=True, slots=True)
class Point3D:
    xyz: tuple[float, float, float]
    rgb: tuple[int, int, int]


def convert_colmap_text_to_hyworld(
    model_dir: Path,
    source_images_dir: Path,
    output_dir: Path,
    *,
    max_images: int | None = None,
) -> dict[str, Any]:
    cameras = read_cameras(model_dir / "cameras.txt")
    images = read_registered_images(model_dir / "images.txt")
    points = read_points3d(model_dir / "points3D.txt")
    if len(images) < 2:
        raise RuntimeError(f"COLMAP registered only {len(images)} images; at least 2 are required.")
    if not points:
        raise RuntimeError("COLMAP reconstruction contains no sparse 3D points.")

    if max_images is not None and len(images) > max_images:
        images = select_angular_uniform(images, max_images)

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    camera_payload: dict[str, dict[str, list[list[float]]]] = {}
    exported_images: list[str] = []
    used_names: set[str] = set()

    for record in sorted(images.values(), key=lambda item: item.name):
        source = source_images_dir / Path(record.name)
        if not source.is_file():
            source = source_images_dir / Path(record.name).name
        if not source.is_file():
            raise FileNotFoundError(f"Registered COLMAP image is missing: {record.name}")
        camera = cameras[record.camera_id]
        key = Path(record.name).stem
        if key in used_names:
            key = f"{key}_{record.image_id:06d}"
        used_names.add(key)
        destination = images_dir / f"{key}.png"
        _save_png(source, destination)
        camera_payload[key] = {
            "extrinsic": world_to_camera_matrix(record),
            "intrinsic": intrinsic_matrix(camera),
        }
        exported_images.append(str(destination.resolve()))

    (output_dir / "cameras.json").write_text(
        json.dumps(camera_payload, indent=2), encoding="utf-8"
    )
    write_ascii_ply(output_dir / "points.ply", points)
    provenance: dict[str, Any] = {
        "schema": "real2sim.hyworld_dataset.v1",
        "source_model": str(model_dir.resolve()),
        "source_images": str(source_images_dir.resolve()),
        "camera_convention": "COLMAP world-to-camera, OpenCV axes",
        "image_count": len(exported_images),
        "point_count": len(points),
        "images": exported_images,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    return provenance


def camera_azimuth_deg(image: RegisteredImage) -> float:
    """从 COLMAP world-to-camera 变换中提取相机的水平方位角（度）。

    COLMAP 存储的是 world-to-camera 变换 (R_wc, t_wc)。
    相机在世界坐标系中的位置为 C = -R_wc^T @ t_wc。
    方位角取 XZ 平面上的 atan2(cx, cz)，假设 Y 轴大致竖直向上。
    """
    qw, qx, qy, qz = image.qvec_wxyz
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0.0:
        return 0.0
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    # R_wc 的转置（即 R_cw），行优先列出 R_wc 各行再按列读
    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r10 = 2.0 * (qx * qy + qz * qw)
    r20 = 2.0 * (qx * qz - qy * qw)
    r02 = 2.0 * (qx * qz + qy * qw)
    r12 = 2.0 * (qy * qz - qx * qw)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
    # C = -R_wc^T @ t  （R_wc^T 的第 j 列 = R_wc 的第 j 行）
    tx, ty, tz = image.tvec_xyz
    cx = -(r00 * tx + r10 * ty + r20 * tz)
    cz = -(r02 * tx + r12 * ty + r22 * tz)
    return math.degrees(math.atan2(cx, cz))


def select_angular_uniform(
    images: dict[int, RegisteredImage],
    target_count: int,
) -> dict[int, RegisteredImage]:
    """从已注册图像中按方位角均匀选出 target_count 张。

    算法：将 [-180°, 180°) 等分为 target_count 个角度 bin，
    每个 bin 里保留方位角最靠近 bin 中心的那张图像。
    若某个 bin 为空，则从相邻 bin 的候选中补选最近的一张。
    """
    if target_count <= 0:
        raise ValueError("target_count 必须为正整数")
    if len(images) <= target_count:
        return images

    with_az: list[tuple[float, RegisteredImage]] = sorted(
        ((camera_azimuth_deg(img), img) for img in images.values()),
        key=lambda x: x[0],
    )

    bin_width = 360.0 / target_count
    # bins[i] = (最小距离, RegisteredImage)
    bins: dict[int, tuple[float, RegisteredImage]] = {}
    for azimuth, img in with_az:
        # 将 azimuth 映射到 [0, 360) 再除以 bin_width 取整
        bin_idx = int((azimuth + 180.0) % 360.0 / bin_width) % target_count
        bin_center = -180.0 + (bin_idx + 0.5) * bin_width
        dist = abs(azimuth - bin_center)
        if bin_idx not in bins or dist < bins[bin_idx][0]:
            bins[bin_idx] = (dist, img)

    # 若部分 bin 为空，从全部候选里按方位角距离补满
    if len(bins) < target_count:
        assigned_ids = {v[1].image_id for v in bins.values()}
        remaining = [(az, img) for az, img in with_az if img.image_id not in assigned_ids]
        for bin_idx in range(target_count):
            if bin_idx in bins:
                continue
            if not remaining:
                break
            bin_center = -180.0 + (bin_idx + 0.5) * bin_width
            best = min(remaining, key=lambda x: abs(x[0] - bin_center))
            bins[bin_idx] = (0.0, best[1])
            remaining.remove(best)

    return {img.image_id: img for _, img in bins.values()}


def read_cameras(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    for line in _data_lines(path):
        fields = line.split()
        if len(fields) < 5:
            raise ValueError(f"Malformed cameras.txt row: {line}")
        camera_id = int(fields[0])
        cameras[camera_id] = Camera(
            camera_id=camera_id,
            model=fields[1],
            width=int(fields[2]),
            height=int(fields[3]),
            params=tuple(float(value) for value in fields[4:]),
        )
    if not cameras:
        raise ValueError(f"No cameras found in {path}")
    return cameras


def read_registered_images(path: Path) -> dict[int, RegisteredImage]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = path.read_text(encoding="utf-8").splitlines()
    images: dict[int, RegisteredImage] = {}
    index = 0
    while index < len(rows):
        row = rows[index].strip()
        if not row or row.startswith("#"):
            index += 1
            continue
        fields = row.split()
        if len(fields) < 10:
            raise ValueError(f"Malformed images.txt row: {row}")
        image_id = int(fields[0])
        qvec = tuple(float(value) for value in fields[1:5])
        tvec = tuple(float(value) for value in fields[5:8])
        images[image_id] = RegisteredImage(
            image_id=image_id,
            qvec_wxyz=(qvec[0], qvec[1], qvec[2], qvec[3]),
            tvec_xyz=(tvec[0], tvec[1], tvec[2]),
            camera_id=int(fields[8]),
            name=" ".join(fields[9:]),
        )
        index += 2
    if not images:
        raise ValueError(f"No registered images found in {path}")
    return images


def read_points3d(path: Path) -> list[Point3D]:
    points: list[Point3D] = []
    for line in _data_lines(path):
        fields = line.split()
        if len(fields) < 8:
            raise ValueError(f"Malformed points3D.txt row: {line}")
        points.append(
            Point3D(
                xyz=(float(fields[1]), float(fields[2]), float(fields[3])),
                rgb=(int(fields[4]), int(fields[5]), int(fields[6])),
            )
        )
    return points


def intrinsic_matrix(camera: Camera) -> list[list[float]]:
    if camera.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        focal, cx, cy = camera.params[:3]
        fx = fy = focal
    elif camera.model in {
        "PINHOLE",
        "OPENCV",
        "OPENCV_FISHEYE",
        "FULL_OPENCV",
        "FOV",
        "THIN_PRISM_FISHEYE",
    }:
        fx, fy, cx, cy = camera.params[:4]
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {camera.model}")
    return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]


def world_to_camera_matrix(image: RegisteredImage) -> list[list[float]]:
    qw, qx, qy, qz = image.qvec_wxyz
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0.0:
        raise ValueError(f"Image {image.image_id} has a zero quaternion")
    qw, qx, qy, qz = (value / norm for value in (qw, qx, qy, qz))
    rotation = [
        [
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ],
        [
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ],
        [
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
    ]
    tx, ty, tz = image.tvec_xyz
    return [
        [*rotation[0], tx],
        [*rotation[1], ty],
        [*rotation[2], tz],
        [0.0, 0.0, 0.0, 1.0],
    ]


def write_ascii_ply(path: Path, points: list[Point3D]) -> None:
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    rows = [
        f"{point.xyz[0]:.9g} {point.xyz[1]:.9g} {point.xyz[2]:.9g} "
        f"{point.rgb[0]} {point.rgb[1]} {point.rgb[2]}"
        for point in points
    ]
    path.write_text("\n".join(header + rows) + "\n", encoding="ascii")


def _data_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _save_png(source: Path, destination: Path) -> None:
    if source.suffix.lower() == ".png":
        copy2(source, destination)
        return
    with Image.open(source) as image:
        image.convert("RGB").save(destination)
