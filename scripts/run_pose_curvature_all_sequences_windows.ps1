<#
.SYNOPSIS
Runs a lightweight pose-curvature audit over KITTI Odometry Sequences 00-10.

.DESCRIPTION
This stage reads poses only; it does not run CLRNet.  With no registered
straight seeds it deliberately uses the lowest-curvature-quartile fallback.
The batch is for candidate discovery.  Final selected windows must be rerun
with reviewed straight seed ranges through run_surf_final_window_windows.ps1.
#>

[CmdletBinding()]
param(
    [string]$EnvName = "surf2026-win",
    [string]$DatasetRoot = "F:\BaiduNetdiskDownload\kitti\odometry",
    [string]$SequenceIds = "00,01,02,03,04,05,06,07,08,09,10",
    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found. Open Anaconda PowerShell Prompt and retry."
}

. (Join-Path $PSScriptRoot "kitti_odometry_workstation_input.ps1")
$Analyzer = Join-Path $PSScriptRoot "analyze_pose_curvature.py"
if (-not (Test-Path -LiteralPath $Analyzer -PathType Leaf)) {
    throw "Curvature analyzer is missing: $Analyzer"
}

$Ids = @(
    $SequenceIds.Split(",") |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ }
)
if (-not $Ids) { throw "SequenceIds is empty." }
foreach ($Id in $Ids) {
    if ($Id -notmatch "^\d{2}$") { throw "Bad sequence id: $Id" }
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputRoot = Join-Path $ProjectRoot "workstation_outputs\pose_curvature_all_$Stamp"
}
if (Test-Path -LiteralPath $OutputRoot) {
    if (Get-ChildItem -LiteralPath $OutputRoot -Force) {
        throw "OutputRoot must be new or empty: $OutputRoot"
    }
} else {
    New-Item -ItemType Directory -Path $OutputRoot | Out-Null
}
$OutputRoot = (Resolve-Path -LiteralPath $OutputRoot).Path

$Rows = @()
foreach ($Id in $Ids) {
    Write-Host ""
    Write-Host "[Sequence $Id pose curvature]"
    $Paths = Resolve-KittiOdometryWorkstationInput `
        -DatasetRoot $DatasetRoot `
        -SequenceId $Id
    $Input = Test-KittiOdometrySequenceInput -InputPaths $Paths
    $Output = Join-Path $OutputRoot "sequence_$Id"
    & conda run --no-capture-output -n $EnvName python $Analyzer `
        --poses $Input.Poses `
        --sequence-id $Id `
        --start-frame 0 `
        --end-frame $Input.MaximumFrame `
        --window-length 15 `
        --window-stride 10 `
        --output-dir $Output
    if ($LASTEXITCODE -ne 0) {
        throw "Sequence $Id curvature audit failed with exit code $LASTEXITCODE."
    }
    $Result = Get-Content (Join-Path $Output "CURVATURE_RESULT.json") -Raw |
        ConvertFrom-Json
    $Rows += [PSCustomObject]@{
        sequence_id = $Id
        frame_count = [int]$Result.counts.frames
        baseline_source = [string]$Result.thresholds.baseline_source
        straight_threshold_1pm = [double]$Result.thresholds.straight_threshold_abs_curvature_1pm
        curve_threshold_1pm = [double]$Result.thresholds.curve_threshold_abs_curvature_1pm
        straight_frames = [int]$Result.counts.state_frames.straight
        enter_transition_frames = [int]$Result.counts.state_frames.enter_transition
        curve_frames = [int]$Result.counts.state_frames.curve
        exit_transition_frames = [int]$Result.counts.state_frames.exit_transition
        result_directory = $Output
    }
}

$Rows | Export-Csv `
    -LiteralPath (Join-Path $OutputRoot "CURVATURE_BATCH_SUMMARY.csv") `
    -NoTypeInformation `
    -Encoding UTF8
$Status = [ordered]@{
    status = "complete"
    sequences = $Ids
    sequence_count = $Rows.Count
    threshold_scope = "automatic lowest-curvature-quartile fallback for discovery"
    final_selected_windows_require_registered_straight_seeds = $true
    previous_outputs_modified = $false
}
$Status | ConvertTo-Json -Depth 6 |
    Set-Content -LiteralPath (Join-Path $OutputRoot "STATUS.json") -Encoding UTF8

Write-Host ""
Write-Host "POSE CURVATURE BATCH FINISHED"
Write-Host "Output root: $OutputRoot"
Write-Host "Summary: $(Join-Path $OutputRoot 'CURVATURE_BATCH_SUMMARY.csv')"
Write-Host "Scope: discovery only; selected windows still require straight-seed calibration."

