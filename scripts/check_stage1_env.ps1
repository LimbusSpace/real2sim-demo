param(
    [string]$Colmap = $env:REAL2SIM_COLMAP,
    [string]$Ffmpeg = "ffmpeg",
    [string]$GaussianPython = $env:REAL2SIM_GAUSSIAN_PYTHON,
    [string]$HyWorldRoot = $env:REAL2SIM_HYWORLD_ROOT,
    [string]$AssetsRoot = $env:REAL2SIM_ASSETS,
    [string]$SnapshotManifest = (Join-Path $PSScriptRoot "..\reproducibility\hyworld.snapshot.json")
)

$ErrorActionPreference = "Stop"
$failures = [System.Collections.Generic.List[string]]::new()

function Resolve-Tool([string]$Name, [string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value)) {
        $script:failures.Add("$Name is not configured")
        return $null
    }
    if (Test-Path -LiteralPath $Value) {
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

$colmapPath = Resolve-Tool "colmap" $Colmap
$ffmpegPath = Resolve-Tool "ffmpeg" $Ffmpeg
$pythonPath = Resolve-Tool "gaussian_python" $GaussianPython

if ([string]::IsNullOrWhiteSpace($AssetsRoot)) {
    $failures.Add("REAL2SIM_ASSETS is not configured")
} elseif (-not (Test-Path -LiteralPath $AssetsRoot -PathType Container)) {
    $failures.Add("REAL2SIM_ASSETS does not exist: $AssetsRoot")
} else {
    Write-Output "assets_root=OK ($((Resolve-Path -LiteralPath $AssetsRoot).Path))"
}

if ([string]::IsNullOrWhiteSpace($HyWorldRoot)) {
    $failures.Add("REAL2SIM_HYWORLD_ROOT is not configured")
} elseif (-not (Test-Path -LiteralPath $HyWorldRoot -PathType Container)) {
    $failures.Add("REAL2SIM_HYWORLD_ROOT does not exist: $HyWorldRoot")
} elseif (-not (Test-Path -LiteralPath $SnapshotManifest -PathType Leaf)) {
    $failures.Add("HY-World snapshot manifest does not exist: $SnapshotManifest")
} else {
    $snapshot = Get-Content -LiteralPath $SnapshotManifest -Raw | ConvertFrom-Json
    foreach ($item in $snapshot.files) {
        $relative = $item.path.Replace("/", [IO.Path]::DirectorySeparatorChar)
        $candidate = Join-Path $HyWorldRoot $relative
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            $failures.Add("HY-World snapshot file is missing: $($item.path)")
            continue
        }
        $actual = (Get-FileHash -LiteralPath $candidate -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $item.sha256.ToLowerInvariant()) {
            $failures.Add("HY-World snapshot hash mismatch: $($item.path)")
        }
    }
    if ($failures.Count -eq 0) {
        Write-Output "hyworld_snapshot=OK ($($snapshot.source.commit))"
    }
}

if (-not [string]::IsNullOrWhiteSpace($env:REAL2SIM_HYWORLD_DEPS)) {
    if (Test-Path -LiteralPath $env:REAL2SIM_HYWORLD_DEPS -PathType Container) {
        Write-Output "hyworld_dependency_overlay=OK ($env:REAL2SIM_HYWORLD_DEPS)"
    } else {
        $failures.Add("REAL2SIM_HYWORLD_DEPS does not exist: $env:REAL2SIM_HYWORLD_DEPS")
    }
}

if ($null -ne $colmapPath) {
    $colmapVersion = & $colmapPath -h 2>&1
    $colmapExitCode = $LASTEXITCODE
    $colmapVersion | Select-Object -First 1
    if ($colmapExitCode -ne 0) {
        $failures.Add("COLMAP failed its version check")
    } elseif ($colmapVersion[0] -notmatch "COLMAP 4\.1\.1") {
        $failures.Add("COLMAP 4.1.1 is required; found: $($colmapVersion[0])")
    }
}
if ($null -ne $ffmpegPath) {
    $ffmpegVersion = & $ffmpegPath -version 2>&1
    $ffmpegExitCode = $LASTEXITCODE
    $ffmpegVersion | Select-Object -First 1
    if ($ffmpegExitCode -ne 0) {
        $failures.Add("FFmpeg failed its version check")
    }
}
if ($null -ne $pythonPath) {
    & $pythonPath -c 'import torch; print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}"); raise SystemExit(0 if torch.cuda.is_available() else 1)'
    if ($LASTEXITCODE -ne 0) {
        $failures.Add("Gaussian Python cannot load a CUDA-enabled torch")
    }
    if (-not [string]::IsNullOrWhiteSpace($HyWorldRoot)) {
        $env:REAL2SIM_HYWORLD_ROOT = $HyWorldRoot
        $launcher = Join-Path $PSScriptRoot "run_hyworld_trainer.py"
        $launcherHelp = & $pythonPath $launcher --help 2>&1
        $launcherExitCode = $LASTEXITCODE
        $launcherHelp | Select-Object -First 1
        if ($launcherExitCode -ne 0) {
            $failures.Add("HY-World trainer launcher failed to load")
        } else {
            Write-Output "hyworld_launcher=OK"
        }
    }
}

if ($failures.Count -gt 0) {
    foreach ($failure in $failures) {
        Write-Error $failure -ErrorAction Continue
    }
    exit 1
}
Write-Output "stage1_environment=OK"
