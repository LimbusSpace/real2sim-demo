# Public Reproduction Guide

This repository provides reproducible orchestration for three stages:

```text
video -> COLMAP/HY-World -> Gaussian PLY
      -> official 2DGS -> colored mesh
      -> CoACD/obj2mjcf -> static MuJoCo environment
```

It does not claim to reproduce the private `tabletop_v1` video or any later
ProbeSplat/Gaussian Grouping experiment.  Those inputs and outputs remain
outside Git by design.

## What Is Pinned

- Python dependencies: `uv.lock`.
- HY-World source: commit `7f668e67c74338d50684e57be46a438459b6bbe1`.
- Required HY-World file hashes: `hyworld.snapshot.json`.
- Public smoke data: OpenMVG Sceaux Castle, commit
  `fde7f5faba555e3c54700477c304488613346a19`.
- Expected public smoke metrics: `verified_runs.json`.

## Obtain Inputs

The hardware-backed path requires Git, `uv`, FFmpeg, COLMAP 4.1.1 CUDA,
Conda, an NVIDIA CUDA-capable GPU, and a CUDA-enabled HY-World Python
environment. The root `README.md` gives the exact HY-World environment setup.

Clone this repository and install the locked developer environment:

```powershell
git clone https://github.com/LimbusSpace/real2sim-demo.git
Set-Location real2sim-demo
uv sync --frozen --group dev --group physics
```

Download the fixed HY-World archive. On Windows, use Free Download Manager:

```powershell
& 'C:\Program Files\Softdeluxe\Free Download Manager\fdm.exe' 'https://github.com/Tencent-Hunyuan/HY-World-2.0/archive/7f668e67c74338d50684e57be46a438459b6bbe1.zip'
```

Download the public Sceaux Castle archive in the same way:

```powershell
& 'C:\Program Files\Softdeluxe\Free Download Manager\fdm.exe' 'https://github.com/openMVG/ImageDataset_SceauxCastle/archive/fde7f5faba555e3c54700477c304488613346a19.zip'
```

Extract HY-World anywhere and extract Sceaux Castle below
`$env:REAL2SIM_ASSETS\datasets\openmvg\ImageDataset_SceauxCastle`. Build the
input MP4 exactly as described in the root `README.md`.

## Verify the Environment

Set the environment variables in the root README, then verify the downloaded
HY-World source before training:

```powershell
uv run real2sim-verify-snapshot `
  --manifest reproducibility/hyworld.snapshot.json `
  --root $env:REAL2SIM_HYWORLD_ROOT
./scripts/check_stage1_env.ps1
```

The hash verification rejects a source tree that is at the right repository
revision but contains modified required trainer or dependency files.

Verify the locked Stage 3 CPU environment separately:

```powershell
./scripts/check_physics_env.ps1
```

## Run and Accept

First inspect the complete external command plan without running GPU software:

```powershell
uv run real2sim --config configs/stage1.sceaux.smoke.toml --dry-run
```

Then run the public 500-step smoke reconstruction:

```powershell
uv run real2sim --config configs/stage1.sceaux.smoke.toml
```

Accept the run only when its `manifest.json` records `gaussian_trained`, all
11 cameras are registered, and
`gaussian/ply/point_cloud_499.ply` exists. The reference machine reported
7,498 sparse points and PSNR 15.208 / SSIM 0.6822 / LPIPS 0.421. CUDA kernels
and bundle adjustment are not expected to be bitwise identical across GPUs.

## GitHub CI Scope

GitHub Actions runs the locked Python tests, Ruff, mypy, and dry-runs for both
the reconstruction and physics configurations. A synthetic cube fixture executes
CoACD, obj2mjcf, MuJoCo compilation, and `mj_step` on CPU. CI does not download
multi-gigabyte external reconstruction software or run COLMAP/CUDA training. The
commands above are the required hardware-backed reconstruction path; a completed
Stage 2 mesh can be converted with `real2sim-physics` as documented in the root README.
