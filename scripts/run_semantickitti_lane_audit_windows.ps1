[CmdletBinding()]
param(
    [string]$DatasetRoot = "F:\BaiduNetdiskDownload\kitti\odometry",
    [string]$SemanticKittiRoot = "F:\BaiduNetdiskDownload\SemanticKITTI",
    [string]$OutputRoot = "F:\2026_surf\generated_runs",
    [string]$EnvName = "surf2026-win",
    [switch]$DownloadLabels,
    [switch]$ExportPoints,
    [int]$ReferenceId = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$InputResolver = Join-Path $PSScriptRoot "kitti00_workstation_input.ps1"
. $InputResolver

$InputPaths = Resolve-Kitti00WorkstationInput -DatasetRoot $DatasetRoot
$InputAudit = Test-Kitti00FirstFiveInput `
    -InputPaths $InputPaths `
    -RequireCompleteSequence

$VelodyneCandidates = @(
    (Join-Path $DatasetRoot "data_odometry_velodyne\dataset\sequences\00\velodyne"),
    (Join-Path $DatasetRoot "dataset\sequences\00\velodyne"),
    (Join-Path $DatasetRoot "sequences\00\velodyne")
)
$VelodyneDir = $VelodyneCandidates |
    Where-Object { Test-Path -LiteralPath $_ -PathType Container } |
    Select-Object -First 1
if (-not $VelodyneDir) {
    throw "Could not find Sequence 00 Velodyne scans below $DatasetRoot"
}

$LabelCandidates = @(
    (Join-Path $SemanticKittiRoot "dataset\sequences\00\labels"),
    (Join-Path $SemanticKittiRoot "sequences\00\labels"),
    (Join-Path $DatasetRoot "data_odometry_velodyne\dataset\sequences\00\labels"),
    (Join-Path $DatasetRoot "data_odometry_labels\dataset\sequences\00\labels")
)
$LabelDir = $LabelCandidates |
    Where-Object { Test-Path -LiteralPath $_ -PathType Container } |
    Select-Object -First 1

if (-not $LabelDir -and $DownloadLabels) {
    New-Item -ItemType Directory -Force -Path $SemanticKittiRoot | Out-Null
    $Archive = Join-Path $SemanticKittiRoot "data_odometry_labels.zip"
    if (-not (Test-Path -LiteralPath $Archive -PathType Leaf)) {
        Write-Host "Downloading official SemanticKITTI labels (~179 MB)..."
        Invoke-WebRequest `
            -Uri "https://semantic-kitti.org/assets/data_odometry_labels.zip" `
            -OutFile $Archive
    }
    $ArchiveHash = (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-Host "Downloaded archive SHA256: $ArchiveHash"
    Write-Host "Extracting labels without changing the KITTI odometry directory..."
    Expand-Archive -LiteralPath $Archive -DestinationPath $SemanticKittiRoot -Force
    $LabelDir = $LabelCandidates |
        Where-Object { Test-Path -LiteralPath $_ -PathType Container } |
        Select-Object -First 1
}

if (-not $LabelDir) {
    throw @"
SemanticKITTI labels were not found. KITTI Odometry does not include them.
Rerun this script with -DownloadLabels, or extract the official archive under:
  $SemanticKittiRoot
Expected one of:
  $($LabelCandidates -join "`n  ")
"@
}

$VelodyneCount = (Get-ChildItem -LiteralPath $VelodyneDir -Filter *.bin -File).Count
$LabelCount = (Get-ChildItem -LiteralPath $LabelDir -Filter *.label -File).Count
if ($VelodyneCount -ne 4541) {
    throw "Sequence 00 should contain 4541 Velodyne files, found $VelodyneCount."
}
if ($LabelCount -ne 4541) {
    throw "SemanticKITTI Sequence 00 should contain 4541 label files, found $LabelCount."
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputDir = Join-Path $OutputRoot "semantickitti_lane_audit_$Stamp"
if (Test-Path -LiteralPath $OutputDir) {
    throw "New output path unexpectedly already exists: $OutputDir"
}

$PythonArgs = @(
    "scripts\audit_semantickitti_lane_markings.py",
    "--velodyne-dir", $VelodyneDir,
    "--label-dir", $LabelDir,
    "--image-dir", $InputPaths.ImageDir,
    "--output-dir", $OutputDir,
    "--expected-frames", "4541"
)
if ($ExportPoints) {
    $PythonArgs += @(
        "--export-points",
        "--poses", $InputPaths.Poses,
        "--calib", $InputPaths.Calib,
        "--reference-id", [string]$ReferenceId
    )
}

Write-Host "KITTI image frames: $($InputAudit.ImageCount)"
Write-Host "KITTI Velodyne frames: $VelodyneCount"
Write-Host "SemanticKITTI label frames: $LabelCount"
Write-Host "New output: $OutputDir"

Push-Location $RepoRoot
try {
    & conda run --no-capture-output -n $EnvName python @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "SemanticKITTI lane-marking audit failed."
    }
}
finally {
    Pop-Location
}

Write-Host "Audit complete. Open:"
Write-Host "  $OutputDir\summary.json"
Write-Host "  $OutputDir\frame_lane_marking_counts.csv"
Write-Host "  $OutputDir\contiguous_lane_marking_runs.csv"
if ($ExportPoints) {
    Write-Host "  $OutputDir\lane_marking_points_reference_camera.csv"
}
