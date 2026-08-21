[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string]$ImageDir,
    [Parameter(Mandatory = $true)] [string]$Calib,
    [Parameter(Mandatory = $true)] [string]$Poses,
    [Parameter(Mandatory = $true)] [int]$StartFrame,
    [Parameter(Mandatory = $true)] [int]$EndFrame,
    [Parameter(Mandatory = $true)] [string]$ClrnetRoot,
    [string]$SequenceId = "01",
    [string]$ImagePattern = "{frame_id:06d}.png",
    [ValidateSet("cuda", "cpu")] [string]$Device = "cuda",
    [string]$Python = "python",
    [string]$OutputRoot = "workstation_outputs"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outputBase = Join-Path $repo $OutputRoot
New-Item -ItemType Directory -Force $outputBase | Out-Null
$runRoot = Join-Path (Resolve-Path $outputBase) "final_polynomial_$stamp"
$scanRoot = Join-Path $runRoot "01_lane_scan"
$fitRoot = Join-Path $runRoot "02_polynomial_windows"

New-Item -ItemType Directory -Force $scanRoot, $fitRoot | Out-Null

Write-Host "[1/2] Running CLRNet lane tracking..."
& $Python (Join-Path $repo "scripts\scan_lane_tracks.py") `
    --dataset-name "KITTI Odometry Sequence $SequenceId" `
    --image-dir $ImageDir `
    --image-pattern $ImagePattern `
    --frame-start $StartFrame `
    --frame-end $EndFrame `
    --calib $Calib `
    --poses $Poses `
    --output-dir $scanRoot `
    --candidate-selection-mode temporal_independent `
    --temporal-maximum-gap-frames 3 `
    --minimum-bev-points-per-side 4 `
    --clrnet-root $ClrnetRoot `
    --device $Device `
    --skip-recommendation
if ($LASTEXITCODE -ne 0) { throw "Lane scan failed with exit code $LASTEXITCODE." }

Write-Host "[2/2] Fitting polynomial windows and recording short gaps..."
& $Python (Join-Path $repo "scripts\fit_polynomial_windows.py") `
    --selected-lane-points (Join-Path $scanRoot "selected_lane_points.json") `
    --scan-json (Join-Path $scanRoot "scan.json") `
    --poses $Poses `
    --calib $Calib `
    --sequence-id $SequenceId `
    --start-frame $StartFrame `
    --end-frame $EndFrame `
    --window-length 15 `
    --window-stride 10 `
    --polynomial-degree 2 `
    --minimum-valid-frames-per-side 8 `
    --maximum-missing-run-frames 3 `
    --output-dir $fitRoot
if ($LASTEXITCODE -ne 0) { throw "Polynomial fitting failed with exit code $LASTEXITCODE." }

Write-Host "FINAL POLYNOMIAL RUN FINISHED"
Write-Host "Run root: $runRoot"
Write-Host "Read: $(Join-Path $fitRoot 'RESULT.json')"
Write-Host "Figures: $(Join-Path $fitRoot 'polynomial_windows_xz_overview.png')"
