<#
.SYNOPSIS
Runs the reviewed final SURF experiment for KITTI Odometry Sequence 01.

.DESCRIPTION
This is the fixed release entrypoint.  It keeps the reviewed 851-1005 range,
the registered straight seeds 851-875 and 991-1005, and creates timestamped
outputs.  Sequence 03 and Sequence 07 are not part of the formal result.

Use -RunFullPoseAudit to repeat the lightweight pose-only Sequence 00-10 audit
before the Sequence 01 CLRNet/fusion run.  Historical output directories are
never modified.
#>

[CmdletBinding()]
param(
    [string]$EnvName = "surf2026-win",
    [string]$DatasetRoot = "F:\BaiduNetdiskDownload\kitti\odometry",
    [string]$ClrnetRoot = "",
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cuda",
    [switch]$RunFullPoseAudit
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

if ([string]::IsNullOrWhiteSpace($ClrnetRoot)) {
    $ClrnetRoot = Join-Path $ProjectRoot "CLRNet"
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$SessionRoot = Join-Path $ProjectRoot "workstation_outputs\surf_final_sequence01_$Stamp"
New-Item -ItemType Directory -Path $SessionRoot | Out-Null

$PoseAuditRoot = $null
if ($RunFullPoseAudit) {
    $PoseAuditRoot = Join-Path $SessionRoot "00_pose_curvature_sequences00_10"
    & (Join-Path $PSScriptRoot "run_pose_curvature_all_sequences_windows.ps1") `
        -EnvName $EnvName `
        -DatasetRoot $DatasetRoot `
        -OutputRoot $PoseAuditRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Full pose-curvature audit failed with exit code $LASTEXITCODE."
    }
}

$SequenceRoot = Join-Path $SessionRoot "01_sequence01_frames0851_1005"
& (Join-Path $PSScriptRoot "run_surf_final_window_windows.ps1") `
    -EnvName $EnvName `
    -DatasetRoot $DatasetRoot `
    -ClrnetRoot $ClrnetRoot `
    -Device $Device `
    -SequenceId "01" `
    -StartFrame 851 `
    -EndFrame 1005 `
    -StraightSeedRanges "851-875,991-1005" `
    -WindowLength 15 `
    -WindowStride 10 `
    -OutputRoot $SequenceRoot
if ($LASTEXITCODE -ne 0) {
    throw "Final Sequence 01 run failed with exit code $LASTEXITCODE."
}

$FinalStatus = Get-Content (Join-Path $SequenceRoot "FINAL_STATUS.json") -Raw |
    ConvertFrom-Json
$SessionStatus = [ordered]@{
    status = [string]$FinalStatus.status
    formal_result = [ordered]@{
        dataset = "KITTI Odometry Sequence 01"
        frames = @(851, 1005)
        straight_seed_ranges = @(@(851, 875), @(991, 1005))
        output = $SequenceRoot
        metrics = Join-Path $SequenceRoot "FINAL_METRICS.json"
        review_bundle = Join-Path $SequenceRoot "final_seq01_851_1005_review_bundle.zip"
    }
    optional_pose_audit = $PoseAuditRoot
    excluded_from_formal_result = @(
        "Sequence 03 frames 29-128",
        "Sequence 07 frames 415-514"
    )
    previous_outputs_modified = $false
}
$SessionStatus | ConvertTo-Json -Depth 8 |
    Set-Content -LiteralPath (Join-Path $SessionRoot "FINAL_SESSION_STATUS.json") `
        -Encoding UTF8

Write-Host ""
Write-Host "SURF FINAL SEQUENCE 01 SESSION FINISHED"
Write-Host "Session root: $SessionRoot"
Write-Host "Formal result: $SequenceRoot"
Write-Host "Metrics: $(Join-Path $SequenceRoot 'FINAL_METRICS.json')"
Write-Host "Review ZIP: $(Join-Path $SequenceRoot 'final_seq01_851_1005_review_bundle.zip')"
