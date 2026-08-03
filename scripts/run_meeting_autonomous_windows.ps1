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
$OutputRoot = Join-Path $RootDir "workstation_outputs\meeting_autonomous_$Stamp"
$LogRoot = Join-Path $RootDir "workstation_outputs\automation_logs"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
$LogPath = Join-Path $LogRoot "meeting_autonomous_$Stamp.log"

# These are contiguous lightweight Sequence-00 candidates selected from the
# earlier SemanticKITTI class-60 coverage audit. Class 60 is only a search cue,
# not lane-polyline ground truth. Every range still has to pass live CLRNet,
# metric-IPM, pose-aligned point-count and visual identity gates.
$Ranges = @(
    "95-134",
    "143-167",
    "1549-1578",
    "1588-1607"
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
    throw "The run ended without STATUS.json. Check $LogPath"
}
$Status = Get-Content -LiteralPath $StatusPath -Raw | ConvertFrom-Json
$Complete = @($Status.curved_candidates | Where-Object status -eq "complete")
$Skipped = @(
    $Status.curved_candidates |
        Where-Object status -eq "skipped_no_valid_two_candidate_run"
)
$Failed = @($Status.curved_candidates | Where-Object status -eq "failed")

Write-Host ""
Write-Host "AUTONOMOUS RUN FINISHED"
Write-Host "Output root: $OutputRoot"
Write-Host "Full log: $LogPath"
Write-Host "Completed curved candidates: $($Complete.Count)"
Write-Host "Skipped by honest input gate: $($Skipped.Count)"
Write-Host "Failed after input gate: $($Failed.Count)"
Write-Host "Status: $StatusPath"
Write-Host ""
Write-Host "The straight first-20 analysis remains available even if every curved candidate is rejected."
