[CmdletBinding()]
param(
    [string]$EnvName = "surf2026-win",
    [string]$DatasetRoot = "F:\BaiduNetdiskDownload\kitti\odometry",
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RootDir

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputRoot = Join-Path $RootDir "workstation_outputs\meeting_followup_$Stamp"
$LogRoot = Join-Path $RootDir "workstation_outputs\automation_logs"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
$LogPath = Join-Path $LogRoot "meeting_followup_$Stamp.log"

# Follow-up ranges do not repeat the four previously selected windows. They
# test: a 30-frame extension of the straight baseline, both sides of the
# 143-157 and 1588-1602 windows, and a separate late-sequence marking run.
# All ranges are contiguous. They remain candidates until the same live
# detector, IPM, pose-aligned metric and visual-identity gates are passed.
$Ranges = @(
    "0-29",
    "118-142",
    "158-182",
    "1563-1587",
    "1603-1627",
    "4526-4540"
)

Start-Transcript -LiteralPath $LogPath -Force | Out-Null
try {
    & (Join-Path $PSScriptRoot "run_meeting_four_questions_windows.ps1") `
        -EnvName $EnvName `
        -DatasetRoot $DatasetRoot `
        -OutputRoot $OutputRoot `
        -Device $Device `
        -CurvedRanges $Ranges
} finally {
    Stop-Transcript | Out-Null
}

$StatusPath = Join-Path $OutputRoot "STATUS.json"
if (-not (Test-Path -LiteralPath $StatusPath)) {
    throw "The follow-up run ended without STATUS.json. Check $LogPath"
}
$Status = Get-Content -LiteralPath $StatusPath -Raw | ConvertFrom-Json
$Complete = @($Status.curved_candidates | Where-Object status -eq "complete")
$Skipped = @(
    $Status.curved_candidates |
        Where-Object status -eq "skipped_no_valid_two_candidate_run"
)
$Failed = @($Status.curved_candidates | Where-Object status -eq "failed")

Write-Host ""
Write-Host "FOLLOW-UP RUN FINISHED"
Write-Host "Output root: $OutputRoot"
Write-Host "Full log: $LogPath"
Write-Host "Completed candidates: $($Complete.Count)"
Write-Host "Skipped by input gate: $($Skipped.Count)"
Write-Host "Failed after input gate: $($Failed.Count)"
Write-Host "Status: $StatusPath"
