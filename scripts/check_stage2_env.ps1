param(
    [string]$TwoDGSPython = $env:REAL2SIM_2DGS_PYTHON,
    [string]$TwoDGSRoot = $env:REAL2SIM_2DGS_ROOT,
    [string]$SnapshotManifest = (Join-Path $PSScriptRoot "..\reproducibility\2dgs.snapshot.json")
)

$ErrorActionPreference = "Stop"
$failures = [System.Collections.Generic.List[string]]::new()
$expectedMain = "335ad612f2e783a4e57b9cbc4d1e167bd599fc98"
$expectedRasterizer = "e0ed0207b3e0669960cfad70852200a4a5847f61"
$expectedKnn = "f155ec04131cb579f53443a06879d37115f4612f"

function Resolve-Tool([string]$Name, [string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value)) {
        $script:failures.Add("$Name is not configured")
        return $null
    }
    if (Test-Path -LiteralPath $Value -PathType Leaf) {
        $resolved = (Resolve-Path -LiteralPath $Value).Path
        Write-Host "$Name=OK ($resolved)"
        return $resolved
    }
    $command = Get-Command $Value -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        $script:failures.Add("$Name is missing: $Value")
        return $null
    }
    Write-Host "$Name=OK ($($command.Source))"
    return $command.Source
}

function Test-GitRevision([string]$Name, [string]$Root, [string]$Expected) {
    $actual = (& git -C $Root rev-parse HEAD 2>$null).Trim()
    if ($LASTEXITCODE -ne 0) {
        $script:failures.Add("Cannot read $Name Git revision at $Root")
    } elseif ($actual -ne $Expected) {
        $script:failures.Add("$Name revision mismatch: $actual expected=$Expected")
    } else {
        Write-Output "$Name=OK ($actual)"
    }
}

$pythonPath = Resolve-Tool "2dgs_python" $TwoDGSPython
if ([string]::IsNullOrWhiteSpace($TwoDGSRoot)) {
    $failures.Add("REAL2SIM_2DGS_ROOT is not configured")
} elseif (-not (Test-Path -LiteralPath $TwoDGSRoot -PathType Container)) {
    $failures.Add("REAL2SIM_2DGS_ROOT does not exist: $TwoDGSRoot")
} elseif (-not (Test-Path -LiteralPath $SnapshotManifest -PathType Leaf)) {
    $failures.Add("2DGS snapshot manifest does not exist: $SnapshotManifest")
} else {
    $TwoDGSRoot = (Resolve-Path -LiteralPath $TwoDGSRoot).Path
    $snapshot = Get-Content -LiteralPath $SnapshotManifest -Raw | ConvertFrom-Json
    foreach ($item in $snapshot.files) {
        $relative = $item.path.Replace("/", [IO.Path]::DirectorySeparatorChar)
        $candidate = Join-Path $TwoDGSRoot $relative
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            $failures.Add("2DGS snapshot file is missing: $($item.path)")
            continue
        }
        $actual = (Get-FileHash -LiteralPath $candidate -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $item.sha256.ToLowerInvariant()) {
            $failures.Add("2DGS snapshot hash mismatch: $($item.path)")
        }
    }
    Test-GitRevision "2dgs_source" $TwoDGSRoot $expectedMain
    Test-GitRevision "diff_surfel_rasterization_source" `
        (Join-Path $TwoDGSRoot "submodules\diff-surfel-rasterization") $expectedRasterizer
    Test-GitRevision "simple_knn_source" `
        (Join-Path $TwoDGSRoot "submodules\simple-knn") $expectedKnn
}

if ($null -ne $pythonPath) {
    & $pythonPath -c @'
import importlib
import torch
import open3d
import matplotlib

required = ("diff_surfel_rasterization", "simple_knn._C")
for module in required:
    importlib.import_module(module)
print(
    f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
    f"open3d={open3d.__version__} matplotlib={matplotlib.__version__}"
)
if torch.__version__ != "2.7.1+cu128":
    raise SystemExit(f"expected torch=2.7.1+cu128, got {torch.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
if tuple(int(part) for part in matplotlib.__version__.split(".")[:2]) >= (3, 10):
    raise SystemExit("matplotlib must be older than 3.10 for upstream 2DGS")
'@
    if ($LASTEXITCODE -ne 0) {
        $failures.Add("2DGS Python environment check failed")
    } else {
        Write-Output "2dgs_python_packages=OK"
    }
}

if ($failures.Count -gt 0) {
    foreach ($failure in $failures) {
        Write-Error $failure -ErrorAction Continue
    }
    exit 1
}
Write-Output "stage2_environment=OK"
