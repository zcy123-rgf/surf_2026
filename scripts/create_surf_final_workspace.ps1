<#
.SYNOPSIS
Creates the reviewed workstation workspace at F:\surf_final.

.DESCRIPTION
Only the final pipeline, the retained five-frame RANSAC baseline, required
runtime modules, documentation and the external CLRNet runtime are copied.
Historical outputs, Git metadata, caches and development drafts are excluded.
The destination must be empty so an earlier result can never be mixed in.
#>

[CmdletBinding()]
param(
    [string]$Destination = "F:\surf_final",
    [string]$ClrnetSource = "F:\2026_surf\CLRNet"
)

$ErrorActionPreference = "Stop"
$SourceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if (-not (Test-Path -LiteralPath $ClrnetSource -PathType Container)) {
    throw "The proven Windows CLRNet runtime was not found: $ClrnetSource"
}
$ClrnetSource = (Resolve-Path -LiteralPath $ClrnetSource).Path

if (Test-Path -LiteralPath $Destination) {
    $Existing = @(Get-ChildItem -LiteralPath $Destination -Force)
    if ($Existing.Count -gt 0) {
        throw "Destination must be empty: $Destination"
    }
} else {
    New-Item -ItemType Directory -Path $Destination | Out-Null
}
$Destination = (Resolve-Path -LiteralPath $Destination).Path

function Copy-RequiredFile {
    param([string]$RelativePath)
    $Source = Join-Path $SourceRoot $RelativePath
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        throw "Required release file is missing: $Source"
    }
    $Target = Join-Path $Destination $RelativePath
    $Parent = Split-Path -Parent $Target
    New-Item -ItemType Directory -Force -Path $Parent | Out-Null
    Copy-Item -LiteralPath $Source -Destination $Target
}

function Copy-CleanDirectory {
    param([string]$Source, [string]$Target)
    New-Item -ItemType Directory -Force -Path $Target | Out-Null
    & robocopy $Source $Target /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP `
        /XD .git __pycache__ .pytest_cache .runtime workstation_outputs `
        /XF .git *.pyc *.pyo
    if ($LASTEXITCODE -gt 7) {
        throw "robocopy failed for $Source with exit code $LASTEXITCODE."
    }
}

$RootFiles = @(
    "requirements-windows.txt",
    "SURF_FINAL_PIPELINE_ZH.md",
    "SURF_FINAL_DECISIONS_ZH.md"
)
$FinalScripts = @(
    "scripts\kitti_odometry_workstation_input.ps1",
    "scripts\scan_clrnet_lane_counts.py",
    "scripts\analyze_pose_curvature.py",
    "scripts\run_pose_curvature_all_sequences_windows.ps1",
    "scripts\fit_first5_two_curves.py",
    "scripts\fit_adaptive_xz_piecewise.py",
    "scripts\run_adaptive_xz_piecewise_windows.ps1",
    "scripts\bridge_occluded_lane_segments.py",
    "scripts\run_surf_final_window_windows.ps1",
    "scripts\run_surf_final_sequence01_windows.ps1"
)
$RansacBaselineScripts = @(
    "scripts\kitti00_workstation_input.ps1",
    "scripts\run_full_point_pipeline.py",
    "scripts\evaluate_denoise_methods.py",
    "scripts\run_selected_ransac_reference.py",
    "scripts\optimize_ransac_parameters.py",
    "scripts\optimize_ransac_safety_expansion.py",
    "scripts\audit_ransac_shortlist.py",
    "scripts\run_ransac_improvement_windows.ps1"
)
$AnnotationFiles = @(
    "annotations\kitti00_first5_manual_annotations.json"
)

foreach ($RelativePath in @(
    $RootFiles + $FinalScripts + $RansacBaselineScripts + $AnnotationFiles
)) {
    Copy-RequiredFile $RelativePath
}

Copy-CleanDirectory `
    -Source (Join-Path $SourceRoot "surf_bev") `
    -Target (Join-Path $Destination "surf_bev")
Copy-CleanDirectory `
    -Source $ClrnetSource `
    -Target (Join-Path $Destination "CLRNet")
New-Item -ItemType Directory -Path (Join-Path $Destination "workstation_outputs") |
    Out-Null

$Commit = "archive_without_git_metadata"
if (
    (Get-Command git -ErrorAction SilentlyContinue) -and
    (Test-Path -LiteralPath (Join-Path $SourceRoot ".git"))
) {
    $CommitValue = & git -C $SourceRoot rev-parse HEAD 2>$null
    if ($LASTEXITCODE -eq 0) { $Commit = [string]$CommitValue }
}
$ClrnetCommit = "runtime_copy_with_windows_compatibility"
if (
    (Get-Command git -ErrorAction SilentlyContinue) -and
    (Test-Path -LiteralPath (Join-Path $ClrnetSource ".git"))
) {
    $ClrnetCommitValue = & git -C $ClrnetSource rev-parse HEAD 2>$null
    if ($LASTEXITCODE -eq 0) { $ClrnetCommit = [string]$ClrnetCommitValue }
}

$Manifest = [ordered]@{
    status = "complete"
    created_at = (Get-Date).ToString("o")
    destination = $Destination
    source_commit = $Commit.Trim()
    clrnet_commit = $ClrnetCommit.Trim()
    clrnet_policy = "separate runtime directory; final pipeline imports but does not edit it"
    contents = @(
        "final pose-curvature and adaptive X/Z lane pipeline",
        "five-frame improved-RANSAC baseline retained for comparison",
        "external CLRNet Windows runtime and weights",
        "empty workstation_outputs directory"
    )
    exclusions = @(
        "historical results",
        "development drafts",
        "Git metadata",
        "Python and test caches"
    )
}
$Manifest | ConvertTo-Json -Depth 6 |
    Set-Content -LiteralPath (Join-Path $Destination "SOURCE_MANIFEST.json") `
        -Encoding UTF8

Write-Host "SURF FINAL WORKSPACE CREATED"
Write-Host "Path: $Destination"
Write-Host "Start here: $(Join-Path $Destination 'SURF_FINAL_PIPELINE_ZH.md')"
Write-Host "Historical outputs copied: no"
