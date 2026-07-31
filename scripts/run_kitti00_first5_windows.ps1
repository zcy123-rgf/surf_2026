[CmdletBinding()]
param(
    [string]$EnvName = "surf2026-win",
    [string]$DatasetRoot = "F:\BaiduNetdiskDownload\kitti\odometry",
    [string]$OutputDir = "",
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [string]$Weights = "0.2,0.4,0.6,0.8,1.0"
)

$ErrorActionPreference = "Stop"
$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RootDir

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found. Open Anaconda PowerShell Prompt and retry."
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputDir = Join-Path $RootDir "workstation_outputs\kitti00_first5_$Stamp"
}

. (Join-Path $PSScriptRoot "kitti00_workstation_input.ps1")
$Kitti = Resolve-Kitti00WorkstationInput -DatasetRoot $DatasetRoot
$Verification = Test-Kitti00FirstFiveInput `
    -InputPaths $Kitti `
    -RequireCompleteSequence
$ImageDir = $Kitti.ImageDir
$Calib = $Kitti.Calib
$Poses = $Kitti.Poses

$PythonArgs = @(
    "scripts\run_full_point_pipeline.py",
    "--image-dir", $ImageDir,
    "--image-pattern", "{frame_id:06d}.png",
    "--calib", $Calib,
    "--poses", $Poses,
    "--frame-ids", "0,1,2,3,4",
    "--reference-id", "4",
    "--local-x-range=-10,10",
    "--local-z-range=3,50",
    "--fusion-x-range=-15,15",
    "--fusion-z-range=-10,50",
    "--weights", $Weights,
    "--device", $Device,
    "--output-dir", $OutputDir
)

Write-Host "Running KITTI Odometry Sequence 00 frames 000000-000004."
Write-Host "Dataset root: $DatasetRoot"
Write-Host "Resolved layout: $($Verification.Layout)"
Write-Host "Verified full sequence: $($Verification.ImageCount) images, $($Verification.PoseRows) poses."
Write-Host "Reference frame: 000004"
Write-Host "Output directory: $OutputDir"
& conda run --no-capture-output -n $EnvName python @PythonArgs
if ($LASTEXITCODE -ne 0) {
    throw "Five-frame pose fusion failed with exit code $LASTEXITCODE."
}

$StatusPath = Join-Path $OutputDir "00_metadata\run_status.json"
if (-not (Test-Path -LiteralPath $StatusPath)) {
    throw "The runner exited without writing run_status.json."
}
$Status = Get-Content -LiteralPath $StatusPath -Raw | ConvertFrom-Json
if ($Status.status -ne "complete") {
    throw "The runner did not report complete status."
}

Write-Host ""
Write-Host "Completed. Check these files first:"
Write-Host "  04_pose_aligned_points\five_frame_metric_pose_fusion.png"
Write-Host "  05_fusion_without_denoise\accumulated_points_by_frame.png"
Write-Host "  05_fusion_without_denoise\weighted_score_heatmap.png"
Write-Host "  00_metadata\pose_alignment.csv"
Write-Host "  00_metadata\audit.json"
