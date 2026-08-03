[CmdletBinding()]
param(
    [string]$EnvName = "surf2026-win",
    [string]$DatasetRoot = "F:\BaiduNetdiskDownload\kitti\odometry",
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [string]$SelectionRoot = "",
    [string]$ManualJson = ""
)

$ErrorActionPreference = "Stop"
$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RootDir

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found. Open Anaconda PowerShell Prompt and retry."
}
. (Join-Path $PSScriptRoot "kitti00_workstation_input.ps1")
$Kitti = Resolve-Kitti00WorkstationInput -DatasetRoot $DatasetRoot
$null = Test-Kitti00FirstFiveInput -InputPaths $Kitti -RequireCompleteSequence

function Get-LatestDirectory {
    param([string]$Pattern)
    return Get-ChildItem (Join-Path $RootDir "workstation_outputs") -Directory |
        Where-Object Name -Like $Pattern |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
}

function Invoke-CheckedScript {
    param([string]$Path, [string[]]$Arguments)
    & $Path @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Path failed with exit code $LASTEXITCODE."
    }
}

if ([string]::IsNullOrWhiteSpace($SelectionRoot)) {
    $Overlap = Get-LatestDirectory -Pattern "hierarchy_overlap_selection_*"
    if ($Overlap) {
        $SelectionRoot = $Overlap.FullName
    }
}

if ([string]::IsNullOrWhiteSpace($SelectionRoot)) {
    $BaseSelection = Get-LatestDirectory -Pattern "hierarchy_selection_*"
    if (-not $BaseSelection) {
        Write-Host "No reusable scan exists. Running the CLRNet feasibility scan first."
        & (Join-Path $PSScriptRoot "run_hierarchy_frame_selection_windows.ps1") `
            -EnvName $EnvName -DatasetRoot $DatasetRoot -Device $Device
        if ($LASTEXITCODE -ne 0) {
            throw "Initial hierarchy selection failed with exit code $LASTEXITCODE."
        }
        $BaseSelection = Get-LatestDirectory -Pattern "hierarchy_selection_*"
    }
    if (-not $BaseSelection) {
        throw "No hierarchy_selection_* directory exists after the scan."
    }
    & (Join-Path $PSScriptRoot "rerun_hierarchy_selection_overlap_windows.ps1") `
        -EnvName $EnvName `
        -DatasetRoot $DatasetRoot `
        -PreviousSelectionRoot $BaseSelection.FullName
    if ($LASTEXITCODE -ne 0) {
        throw "Overlapping hierarchy selection failed with exit code $LASTEXITCODE."
    }
    $Overlap = Get-LatestDirectory -Pattern "hierarchy_overlap_selection_*"
    if (-not $Overlap) {
        throw "No hierarchy_overlap_selection_* directory was produced."
    }
    $SelectionRoot = $Overlap.FullName
}

$SelectionRoot = (Resolve-Path -LiteralPath $SelectionRoot).Path
$SelectionStatusPath = Join-Path $SelectionRoot "STATUS.json"
if (-not (Test-Path -LiteralPath $SelectionStatusPath -PathType Leaf)) {
    throw "Selection STATUS.json is missing: $SelectionStatusPath"
}
$SelectionStatus = Get-Content -LiteralPath $SelectionStatusPath -Raw |
    ConvertFrom-Json
if ($SelectionStatus.selection_status -ne "selected") {
    throw "Selection coverage gate was not met: $($SelectionStatus.selection_status)"
}
$SelectionJson = [string]$SelectionStatus.selection_json
if (-not (Test-Path -LiteralPath $SelectionJson -PathType Leaf)) {
    throw "Selection JSON is missing: $SelectionJson"
}
$AnnotationZip = [string]$SelectionStatus.annotation_zip
if (-not (Test-Path -LiteralPath $AnnotationZip -PathType Leaf)) {
    throw "Manual annotation package is missing: $AnnotationZip"
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputRoot = Join-Path $RootDir "workstation_outputs\weekly_lane_hierarchy_$Stamp"
$PythonArgs = @(
    "scripts\run_weekly_lane_hierarchy.py",
    "--selection-json", $SelectionJson,
    "--poses", $Kitti.Poses,
    "--calib", $Kitti.Calib,
    "--output-dir", $OutputRoot
)
if (-not [string]::IsNullOrWhiteSpace($ManualJson)) {
    $ManualJson = (Resolve-Path -LiteralPath $ManualJson).Path
    $PythonArgs += @("--manual-json", $ManualJson)
}

Write-Host ""
Write-Host "[Weekly meeting: ten-window lane hierarchy]"
& conda run --no-capture-output -n $EnvName python @PythonArgs
if ($LASTEXITCODE -ne 0) {
    throw "Weekly lane hierarchy failed with exit code $LASTEXITCODE."
}

$StatusPath = Join-Path $OutputRoot "STATUS.json"
$Status = Get-Content -LiteralPath $StatusPath -Raw | ConvertFrom-Json
$Bundle = Join-Path $OutputRoot "meeting_result_bundle.zip"
$BundleItems = @(
    (Join-Path $OutputRoot "MEETING_SUMMARY.md"),
    (Join-Path $OutputRoot "STATUS.json"),
    (Join-Path $OutputRoot "00_audit"),
    (Join-Path $OutputRoot "01_q1_two_curves"),
    (Join-Path $OutputRoot "02_q2_model_comparison"),
    (Join-Path $OutputRoot "03_q3_sparse_refusion"),
    (Join-Path $OutputRoot "04_q4_manual_evaluation")
)
Compress-Archive -LiteralPath $BundleItems -DestinationPath $Bundle -CompressionLevel Optimal

@{
    status = "complete"
    previous_outputs_modified = $false
    selection_root = $SelectionRoot
    output_root = $OutputRoot
    experiment_status = $Status
    meeting_summary = (Join-Path $OutputRoot "MEETING_SUMMARY.md")
    result_bundle = $Bundle
    manual_annotation_package = $AnnotationZip
} | ConvertTo-Json -Depth 8 | Set-Content `
    -LiteralPath (Join-Path $OutputRoot "RUN_STATUS.json") -Encoding UTF8

Write-Host ""
Write-Host "WEEKLY MEETING EXPERIMENT FINISHED"
Write-Host "Output root: $OutputRoot"
Write-Host "Completed windows: $($Status.windows_completed)"
Write-Host "Valid unique frames: $($Status.valid_unique_frames)"
Write-Host "Feature points per window per side: $($Status.selected_feature_points_per_window_per_side)"
Write-Host "Manual evaluation: $($Status.manual_evaluation_status)"
Write-Host "Meeting summary: $(Join-Path $OutputRoot 'MEETING_SUMMARY.md')"
Write-Host "Upload this small bundle for PPT and speech drafting: $Bundle"
Write-Host "Upload this image package to finish question 4: $AnnotationZip"
