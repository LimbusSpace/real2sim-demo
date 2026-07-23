param(
    [string]$SnapshotManifest = (Join-Path $PSScriptRoot "..\reproducibility\physics.snapshot.json")
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not (Test-Path -LiteralPath $SnapshotManifest -PathType Leaf)) {
    Write-Error "Physics snapshot manifest does not exist: $SnapshotManifest"
    exit 1
}

Push-Location $repoRoot
try {
    & uv run --frozen --group physics python -c @'
import importlib.metadata
import shutil
import sys

import coacd
import mujoco

expected = {
    "coacd": "1.0.11",
    "obj2mjcf": "0.0.25",
    "mujoco": "3.10.0",
    "trimesh": "4.12.2",
    "fast-simplification": "0.1.13",
}
if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"expected Python 3.11, got {sys.version.split()[0]}")
for package, version in expected.items():
    actual = importlib.metadata.version(package)
    if actual != version:
        raise SystemExit(f"{package}={actual}, expected={version}")
if not callable(coacd.run_coacd):
    raise SystemExit("coacd.run_coacd is not callable")
if shutil.which("obj2mjcf") is None:
    raise SystemExit("obj2mjcf CLI is not available")
model = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><geom type="plane" size="1 1 0.1"/></worldbody></mujoco>'
)
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
print(f"python={sys.version.split()[0]} mujoco={mujoco.__version__} time={data.time}")
'@
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
} finally {
    Pop-Location
}

Write-Output "stage3_physics_environment=OK"
