<#
.SYNOPSIS
Runs the curvature-adaptive lane pipeline over all KITTI Odometry Sequence 01.

.DESCRIPTION
This is the registered scale experiment for frames 0-1100.  Curvature
thresholds are calibrated from the same reviewed straight seeds used by the
formal 851-1005 experiment.  Straight windows use a quadratic parametric
polynomial; transition and curve windows compare that polynomial with a cubic
parametric B-spline on held-out frames.  Every run uses a new output folder.

This experiment measures coverage, consistency and scalability.  It is not an
official lane-accuracy benchmark and it does not turn project track IDs into
CLRNet semantic lane IDs.
#>

[CmdletBinding()]
param(
    [string]$EnvName = "surf2026-win",
    [string]$DatasetRoot = "F:\BaiduNetdiskDownload\kitti\odometry",
    [string]$ClrnetRoot = "",
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
if ([string]::IsNullOrWhiteSpace($ClrnetRoot)) {
    $ClrnetRoot = Join-Path $ProjectRoot "CLRNet"
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputRoot = Join-Path $ProjectRoot (
    "workstation_outputs\surf_final_sequence01_full_0000_1100_$Stamp"
)

& (Join-Path $PSScriptRoot "run_surf_final_window_windows.ps1") `
    -EnvName $EnvName `
    -DatasetRoot $DatasetRoot `
    -ClrnetRoot $ClrnetRoot `
    -Device $Device `
    -SequenceId "01" `
    -StartFrame 0 `
    -EndFrame 1100 `
    -StraightSeedRanges "851-875,991-1005" `
    -WindowLength 15 `
    -WindowStride 10 `
    -OutputRoot $OutputRoot
if ($LASTEXITCODE -ne 0) {
    throw "Full Sequence 01 experiment failed with exit code $LASTEXITCODE."
}

$Status = Get-Content (Join-Path $OutputRoot "FINAL_STATUS.json") -Raw |
    ConvertFrom-Json
$ScaleStatus = [ordered]@{
    status = [string]$Status.status
    experiment = "KITTI Odometry Sequence 01 full-sequence scale run"
    frames = @(0, 1100)
    frame_count = 1101
    straight_seed_ranges = @(@(851, 875), @(991, 1005))
    output = $OutputRoot
    metrics = Join-Path $OutputRoot "FINAL_METRICS.json"
    review_bundle = Join-Path $OutputRoot "final_seq01_0_1100_review_bundle.zip"
    interpretation = @(
        "curvature thresholds are project thresholds calibrated from reviewed straight seeds",
        "coverage and held-out errors are not official lane-position accuracy",
        "failed identity or continuity gates remain visible as skipped windows or segments"
    )
    previous_outputs_modified = $false
}
$ScaleStatus | ConvertTo-Json -Depth 8 |
    Set-Content -LiteralPath (Join-Path $OutputRoot "FULL_SEQUENCE_STATUS.json") `
        -Encoding UTF8

Write-Host ""
Write-Host "SURF FULL SEQUENCE 01 EXPERIMENT FINISHED"
Write-Host "Status: $($Status.status)"
Write-Host "Output root: $OutputRoot"
Write-Host "Metrics: $(Join-Path $OutputRoot 'FINAL_METRICS.json')"
Write-Host "Review ZIP: $(Join-Path $OutputRoot 'final_seq01_0_1100_review_bundle.zip')"
