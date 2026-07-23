# real2sim-demo

An end-to-end real2sim pipeline, built one inspectable stage at a time.

The current milestones are:

```text
casual video -> extracted frames -> COLMAP cameras/sparse points -> HY-World 3DGS -> PLY
                                      \-> official 2DGS -> bounded TSDF -> colored mesh
                                                                          \-> CoACD -> static MuJoCo MJCF
```

Object-level instance dynamics, measured metric calibration, mass identification, and joint
inference remain outside these milestones.

## Stage 1

```mermaid
flowchart LR
    A[Phone video] --> B[FFmpeg frames]
    B --> C[COLMAP feature matching]
    C --> D[COLMAP sparse reconstruction]
    D --> E[Undistorted images and text model]
    E --> F[HY-World dataset adapter]
    F --> G[HY-World 3DGS trainer]
    G --> H[Gaussian PLY]
```

Every run writes a manifest, trace, per-command logs, intermediate camera data, held-out
renders, and the final Gaussian PLY path. COLMAP and HY-World remain external tools; this
repository provides orchestration and the required data adapter rather than copying their
implementations. The configured HY-World source revision is also stored in each run manifest
and Gaussian provenance file.

## Stage 2

Stage 2 is an independent `real2sim-mesh` flow. It reuses the undistorted images and COLMAP
camera model from a completed Stage 1 run, trains the official 2D Gaussian Splatting model
with every registered view, and exports a colored mesh with bounded TSDF fusion. The HY-World
3DGS PLY remains a parallel Stage 1 artifact and is not used as 2DGS input.

The official `hbb1/2d-gaussian-splatting` code is licensed only for non-commercial research
and evaluation. Review its `LICENSE.md` before using Stage 2. The pinned source revisions are:

| Component | Revision |
| --- | --- |
| `2d-gaussian-splatting` | `335ad612f2e783a4e57b9cbc4d1e167bd599fc98` |
| `diff-surfel-rasterization` | `e0ed0207b3e0669960cfad70852200a4a5847f61` |
| `simple-knn` | `f155ec04131cb579f53443a06879d37115f4612f` |

### Install 2DGS

Use a separate Conda environment so its compiled extensions cannot alter `hyworld2`. The
following Windows setup uses Python 3.11, PyTorch 2.7.1 with CUDA 12.8 wheels, local CUDA
Toolkit 12.6, and the Visual Studio 2022 x64 build tools:

```powershell
git clone --recursive https://github.com/hbb1/2d-gaussian-splatting.git D:\tools\2d-gaussian-splatting
git -C D:\tools\2d-gaussian-splatting checkout 335ad612f2e783a4e57b9cbc4d1e167bd599fc98
git -C D:\tools\2d-gaussian-splatting submodule update --init --recursive

conda create -n surfel_splatting python=3.11 -y
$python = "$env:USERPROFILE\.conda\envs\surfel_splatting\python.exe"
& $python -m pip install torch==2.7.1 torchvision==0.22.1 `
  --index-url https://download.pytorch.org/whl/cu128
& $python -m pip install open3d==0.18.0 mediapy==1.1.2 lpips==0.1.4 `
  scikit-image==0.21.0 tqdm==4.66.2 trimesh==4.3.2 matplotlib==3.9.4 `
  plyfile opencv-python ninja

$root = "D:\tools\2d-gaussian-splatting"
$env:DISTUTILS_USE_SDK = "1"
$env:MSSdk = "1"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$env:CUDA_HOME = $env:CUDA_PATH
$env:TORCH_CUDA_ARCH_LIST = "8.9" # RTX 40 series; adjust for the target GPU
& $python -m pip install --no-build-isolation "$root\submodules\diff-surfel-rasterization"
& $python -m pip install --no-build-isolation "$root\submodules\simple-knn"
```

Run the two extension installation commands from a Visual Studio 2022 x64 developer shell
with `CUDA_PATH` pointing to CUDA 12.6. Then configure and verify the source and environment:

```powershell
$env:REAL2SIM_ASSETS = "E:\real2sim-assets"
$env:REAL2SIM_2DGS_ROOT = "D:\tools\2d-gaussian-splatting"
$env:REAL2SIM_2DGS_PYTHON = "$env:USERPROFILE\.conda\envs\surfel_splatting\python.exe"

uv run real2sim-verify-snapshot `
  --manifest reproducibility/2dgs.snapshot.json `
  --root $env:REAL2SIM_2DGS_ROOT
./scripts/check_stage2_env.ps1
```

### Run Stage 2

Inspect the complete command sequence without requiring a GPU or real input:

```powershell
uv run real2sim-mesh --config configs/stage2.tabletop_v1.toml --dry-run
```

Run each hardware stage separately so the 23-view dataset and 30,000-step surfel PLY can be
checked before TSDF fusion:

```powershell
uv run real2sim-mesh --config configs/stage2.tabletop_v1.toml --stage prepare
uv run real2sim-mesh --config configs/stage2.tabletop_v1.toml --stage train
uv run real2sim-mesh --config configs/stage2.tabletop_v1.toml --stage mesh
```

`--input-run-dir` and `--output-dir` override the TOML paths. The default bounded meshing
settings leave `depth_trunc`, `voxel_size`, and `sdf_trunc` unset so upstream estimates them
from camera scale; each can be explicitly added under `[mesh]` when tuning is required.

Stage 2 writes stable output names independently of the upstream directory depth:

```text
<stage2_output>/
  manifest.json                 # real2sim.mesh.v1
  trace.json
  provenance.json
  dataset/
    images/
    sparse/0/{cameras,images,points3D}.bin
    provenance.json
  2dgs/
    point_cloud/iteration_30000/point_cloud.ply
    provenance.json
  mesh/
    raw.ply
    post.ply
    preview.png                 # front, isometric, top
    provenance.json
  logs/
```

The mesh validator requires nonzero vertices and triangle faces plus RGB vertex colors, and
rejects a post-processed mesh larger than the raw mesh. A capture with incomplete view
coverage can still leave holes in unseen surfaces; Stage 2 does not claim simulation-ready
geometry or repair those regions.

## Stage 3

Stage 3 is the independent `real2sim-physics` flow. It consumes a completed Stage 2
`mesh/post.ply`, fits the dominant tabletop support plane, rotates it to MuJoCo z-up coordinates,
normalizes the scene's largest extent to one meter, and exports a static environment body. The
entire reconstructed tabletop remains one environment asset; Stage 3 does not add a free joint
or claim object-level rigid-body separation.

Install the exact CPU physics toolchain and verify it:

```powershell
uv sync --frozen --group dev --group physics
./scripts/check_physics_env.ps1
```

The pinned dependencies are [CoACD](https://github.com/SarahWeiii/CoACD) 1.0.11 (MIT),
[obj2mjcf](https://github.com/kevinzakka/obj2mjcf) 0.0.25 (MIT),
[MuJoCo](https://github.com/google-deepmind/mujoco) 3.10.0 (Apache-2.0),
[trimesh](https://github.com/mikedh/trimesh) 4.12.2 (MIT), and
[fast-simplification](https://github.com/pyvista/fast-simplification) 0.1.13 (MIT).
The machine-readable list is `reproducibility/physics.snapshot.json`.

Inspect or run the complete tabletop conversion:

```powershell
$env:REAL2SIM_ASSETS = "E:\real2sim-assets"
uv run real2sim-physics --config configs/stage3.tabletop_v1.toml --dry-run
uv run real2sim-physics --config configs/stage3.tabletop_v1.toml
```

Individual stages are available through `--stage prepare|decompose|mjcf|validate`; input and
output can be overridden with `--input-mesh-run-dir` and `--output-dir`. CoACD is invoked by a
separate helper with `real_metric=true`; obj2mjcf handles the visual OBJ/material and base MJCF,
then Stage 3 replaces its fallback collision geometry with the CoACD convex pieces. Final
acceptance directly calls `mujoco.MjModel.from_xml_path` and executes ten `mj_step` calls.

Render a reproducible MP4 preview from the validated scene. The temporary probe bodies are only
used for the video and are never written into the final static `scene.xml`:

```powershell
uv run python scripts/render_physics_video.py `
  --scene "$env:REAL2SIM_ASSETS\runs\tabletop_v1_physics\mjcf\scene.xml" `
  --output "$env:REAL2SIM_ASSETS\runs\tabletop_v1_physics\videos\scene_preview.mp4" `
  --manifest "$env:REAL2SIM_ASSETS\runs\tabletop_v1_physics\manifest.json"
```

```text
<stage3_output>/
  manifest.json                 # real2sim.physics.v1, final stage: validated
  trace.json
  provenance.json
  validation.json
  videos/
    scene_preview.mp4
    scene_preview.json
  source/
    scene.obj
    scene.mtl
    transform.json
  collision/
    scene_collision_000.obj
    ...
  mjcf/
    scene.xml
    scene.obj
    scene.mtl
    scene_collision_*.obj
  logs/
```

The one-meter extent is a reproducible demonstration scale, not real-world calibration. This
stage does not infer independent object motion, mass or inertia, material friction, joints,
automatic instance segmentation, grasp semantics, collision layers, or robot models.

## Reproducibility Contract

The Python package is locked by `uv.lock`. External GPU software is pinned or verified as
follows:

| Component | Reproduced version |
| --- | --- |
| Python | 3.11 (tested with 3.11.15) |
| COLMAP | 4.1.1 CUDA, commit `a0d785f` |
| HY-World 2.0 | commit `7f668e67c74338d50684e57be46a438459b6bbe1` |
| PyTorch | 2.7.1 + CUDA 12.8 |
| FFmpeg | modern build with H.264/HEVC decode support |
| CoACD / obj2mjcf | 1.0.11 / 0.0.25 |
| MuJoCo | 3.10.0 |

Large binaries, model weights, videos, and run outputs are deliberately excluded from Git.
Their locations are supplied through environment variables. The checked-in
`reproducibility/hyworld.snapshot.json` records the official repository, fixed archive URL,
and SHA256 hashes for the trainer, requirements, and custom gsplat entry point. Machine-
readable input hashes and acceptance metrics are stored in
`reproducibility/verified_runs.json`.

For the complete public download, source-verification, execution, and
acceptance procedure, see [the reproduction guide](reproducibility/REPRODUCE.md).

CUDA kernels and COLMAP bundle adjustment are not guaranteed to be bitwise deterministic
across GPUs and drivers. Reproduction means matching the pipeline stages, registered-camera
quality gate, artifact schema, and approximate metrics, not byte-identical PLY files.

## Install

Install the orchestration package and its locked development environment:

```powershell
git clone https://github.com/LimbusSpace/real2sim-demo.git
cd real2sim-demo
uv sync --frozen --group dev --group physics
```

Install COLMAP 4.1.1 CUDA from the official
[COLMAP releases](https://github.com/colmap/colmap/releases/tag/4.1.1) and install FFmpeg.
Both may live outside this repository.

### HY-World without git clone

Download this fixed archive with a browser or download manager such as FDM:

```text
https://github.com/Tencent-Hunyuan/HY-World-2.0/archive/7f668e67c74338d50684e57be46a438459b6bbe1.zip
```

Extract it to any disk. A full `git clone` is not required. Create the HY-World environment
using the pinned upstream files in that archive:

```powershell
conda create -n hyworld2 python=3.11.15 -y
conda activate hyworld2
python -m pip install -r D:\tools\HY-World-2.0\requirements.txt
python -m pip install --no-build-isolation -e D:\tools\HY-World-2.0\hyworld2\worldgen\third_party\gsplat_maskgaussian
python -m pip install --no-build-isolation -r D:\tools\HY-World-2.0\requirements_git.txt
```

The upstream project documents CUDA 12.8. Only the 3DGS trainer is used here; panorama,
navigation, and diffusion-model weights are not needed for Stage 1.

## Configure

Set these variables in the PowerShell session used to run the pipeline:

```powershell
$env:REAL2SIM_ASSETS = "D:\real2sim-assets"
$env:REAL2SIM_COLMAP = "D:\tools\colmap-4.1.1\COLMAP.bat"
$env:REAL2SIM_GAUSSIAN_PYTHON = "C:\Users\me\.conda\envs\hyworld2\python.exe"
$env:REAL2SIM_HYWORLD_ROOT = "D:\tools\HY-World-2.0"
$env:REAL2SIM_VIDEO = "D:\captures\tabletop.mp4"
```

`REAL2SIM_HYWORLD_DEPS` is optional. It may point to an existing dependency overlay when
the HY-World packages are not installed directly into `REAL2SIM_GAUSSIAN_PYTHON`.

Verify the downloaded source snapshot and the complete local environment:

```powershell
uv run real2sim-verify-snapshot `
  --manifest reproducibility/hyworld.snapshot.json `
  --root $env:REAL2SIM_HYWORLD_ROOT

./scripts/check_stage1_env.ps1
```

The environment check exits with code 1 for missing tools, source hash mismatches, a missing
asset directory, or a Python environment without CUDA-enabled PyTorch.

## Public Smoke Dataset

The public sample uses the small OpenMVG Sceaux Castle dataset at commit
`fde7f5faba555e3c54700477c304488613346a19`:

```text
https://github.com/openMVG/ImageDataset_SceauxCastle/archive/fde7f5faba555e3c54700477c304488613346a19.zip
```

Place the extracted dataset at:

```text
${REAL2SIM_ASSETS}/datasets/openmvg/ImageDataset_SceauxCastle
```

Create the video input from its 11 ordered JPEG files:

```powershell
$dataset = "$env:REAL2SIM_ASSETS\datasets\openmvg\ImageDataset_SceauxCastle"
ffmpeg -y -framerate 2 -start_number 7100 `
  -i "$dataset\images\100_%04d.JPG" `
  -c:v libx264 -pix_fmt yuv420p "$dataset\sceaux_castle.mp4"
```

Inspect all commands without executing external tools:

```powershell
uv run real2sim --config configs/stage1.sceaux.smoke.toml --dry-run
```

Run the 500-step public smoke reconstruction:

```powershell
uv run real2sim --config configs/stage1.sceaux.smoke.toml
```

The acceptance gate is 11 registered cameras, a `gaussian_trained` manifest, and a PLY at
`gaussian/ply/point_cloud_499.ply`. The verified machine produced 7,498 sparse points and
PSNR 15.208 / SSIM 0.6822 / LPIPS 0.421.

## Run a Phone Video

The portable example reads the input from `REAL2SIM_VIDEO` and stores outputs under
`REAL2SIM_ASSETS`:

```powershell
uv run real2sim --config configs/stage1.windows.example.toml --stage prepare
uv run real2sim --config configs/stage1.windows.example.toml --stage train
```

Preparation and training are separate so the COLMAP registration count can be checked in
`manifest.json` before spending GPU time. For a short phone orbit, exhaustive matching is
available in `configs/stage1.tabletop_v1.toml`.

The verified tabletop run used 64 extracted frames. COLMAP registered 23 frames with 553
sparse points; 5,000 HY-World steps produced 54,768 Gaussians and PSNR 30.986 / SSIM 0.9704
/ LPIPS 0.127 on three held-out registered views. The registered views render clearly, but
that capture does not provide complete 360-degree coverage.

## Artifact Layout

```text
<run_dir>/
  manifest.json
  trace.json
  frames_manifest.json
  frames/
  logs/
  colmap/
    database.db
    sparse/
    undistorted/
    model_txt/
  hyworld_dataset/
    images/
    cameras.json
    points.ply
    provenance.json
  gaussian/
    ckpts/
    ply/
      point_cloud_<step>.ply
    renders/
      val_step<step>_<view>.png
    provenance.json
```

The pipeline selects the COLMAP sparse model with the most registered images instead of
assuming `sparse/0` is the best reconstruction. Failures are recorded in the manifest and
the corresponding command log. A checkpoint without a PLY does not count as success.

## Capture Notes

- Record a slow 30-60 second orbit around a small tabletop scene.
- Keep exposure and focus locked and avoid motion blur.
- Use opaque, textured objects and keep static background features visible.
- Avoid transparent, mirror-like, or textureless objects for the first run.
- Move the camera position; do not only rotate in place.

Monocular COLMAP reconstruction has arbitrary global scale. Metric calibration is deferred
until the physics-asset milestone.

## Development

```powershell
uv sync --frozen --group dev
uv run pytest -q
uv run ruff check .
uv run mypy src
uv run real2sim-physics --config configs/stage3.tabletop_v1.toml --dry-run
```

GitHub Actions runs the same checks plus a full dry-run on every push and pull request.
