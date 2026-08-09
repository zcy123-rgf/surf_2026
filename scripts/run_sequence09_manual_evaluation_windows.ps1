[CmdletBinding()]
param(
    [string]$EnvName = "surf2026-win",
    [string]$DatasetRoot = "F:\BaiduNetdiskDownload\kitti\odometry",
    [Parameter(Mandatory = $true)]
    [string]$BatchRoot,
    [string]$ManualJson = "",
    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"
$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RootDir

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found. Open Anaconda PowerShell Prompt and retry."
}

$BatchRoot = (Resolve-Path -LiteralPath $BatchRoot).Path
$SelectionJson = Join-Path $BatchRoot "sequence_09\01_ranked_option\selection.json"
if (-not (Test-Path -LiteralPath $SelectionJson -PathType Leaf)) {
    throw "Sequence 09 selection JSON is missing: $SelectionJson"
}

if ([string]::IsNullOrWhiteSpace($ManualJson)) {
    $ManualJson = Join-Path $RootDir `
        "annotations\kitti09_frames0051_0155_manual_boundaries.json"
}
$ManualJson = (Resolve-Path -LiteralPath $ManualJson).Path

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputRoot = Join-Path $RootDir `
        "workstation_outputs\sequence09_manual_evaluation_$Stamp"
}
if (Test-Path -LiteralPath $OutputRoot) {
    if (Get-ChildItem -LiteralPath $OutputRoot -Force) {
        throw "OutputRoot must be new or empty: $OutputRoot"
    }
} else {
    New-Item -ItemType Directory -Path $OutputRoot | Out-Null
}
$OutputRoot = (Resolve-Path -LiteralPath $OutputRoot).Path

. (Join-Path $PSScriptRoot "kitti_odometry_workstation_input.ps1")
$Kitti = Resolve-KittiOdometryWorkstationInput `
    -DatasetRoot $DatasetRoot `
    -SequenceId "09"
$null = Test-KittiOdometrySequenceInput -InputPaths $Kitti

Write-Host ""
Write-Host "[Sequence 09: manual pseudo-ground-truth evaluation]"
& conda run --no-capture-output -n $EnvName python `
    "scripts\run_weekly_lane_hierarchy.py" `
    --selection-json $SelectionJson `
    --poses $Kitti.Poses `
    --calib $Kitti.Calib `
    --manual-json $ManualJson `
    --output-dir $OutputRoot `
    --maximum-cv-folds 20 `
    --curve-samples 600 `
    --feature-count-grid "4,6,8,12"
if ($LASTEXITCODE -ne 0) {
    throw "Sequence 09 manual evaluation failed with exit code $LASTEXITCODE."
}

$Bundle = Join-Path $OutputRoot "SEQUENCE09_MANUAL_EVALUATION_RESULT.zip"
$BundleItems = @(
    (Join-Path $OutputRoot "STATUS.json"),
    (Join-Path $OutputRoot "MEETING_SUMMARY.md"),
    (Join-Path $OutputRoot "00_audit"),
    (Join-Path $OutputRoot "01_q1_two_curves"),
    (Join-Path $OutputRoot "02_q2_model_comparison"),
    (Join-Path $OutputRoot "03_q3_sparse_refusion"),
    (Join-Path $OutputRoot "04_q4_manual_evaluation"),
    $ManualJson
)
Compress-Archive -LiteralPath $BundleItems `
    -DestinationPath $Bundle -CompressionLevel Optimal

Write-Host ""
Write-Host "SEQUENCE 09 MANUAL EVALUATION FINISHED"
Write-Host "Output root: $OutputRoot"
Write-Host "Metrics: $(Join-Path $OutputRoot '04_q4_manual_evaluation\manual_metrics.csv')"
Write-Host "Result ZIP: $Bundle"
