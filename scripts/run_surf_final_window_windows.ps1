<#
.SYNOPSIS
Runs one clean final-window experiment without touching historical outputs.

.DESCRIPTION
The pipeline resolves official KITTI files, runs CLRNet, keeps left/right
observations independently, calculates pose curvature with registered
hysteresis thresholds, compares polynomial and B-spline curve models, checks
G1 window interfaces, and exports optional low-confidence occlusion bridges.
Every invocation creates a new timestamped output directory.
#>

[CmdletBinding()]
param(
    [string]$EnvName = "surf2026-win",
    [string]$DatasetRoot = "F:\BaiduNetdiskDownload\kitti\odometry",
    [string]$ClrnetRoot = "",
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cuda",
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^\d{2}$")]
    [string]$SequenceId,
    [Parameter(Mandatory = $true)]
    [int]$StartFrame,
    [Parameter(Mandatory = $true)]
    [int]$EndFrame,
    [string]$StraightSeedRanges = "",
    [int]$WindowLength = 15,
    [int]$WindowStride = 10,
    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

if ([string]::IsNullOrWhiteSpace($ClrnetRoot)) {
    $ClrnetRoot = Join-Path $ProjectRoot "CLRNet"
}

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found. Open Anaconda PowerShell Prompt and retry."
}
if ($StartFrame -lt 0 -or $EndFrame -lt $StartFrame) {
    throw "Frames must satisfy 0 <= StartFrame <= EndFrame."
}
if ($EndFrame - $StartFrame + 1 -lt $WindowLength) {
    throw "Requested range is shorter than one complete window."
}
if (-not (Test-Path -LiteralPath $ClrnetRoot -PathType Container)) {
    throw "External CLRNet directory was not found: $ClrnetRoot"
}

$RequiredScripts = @(
    "kitti_odometry_workstation_input.ps1",
    "scan_clrnet_lane_counts.py",
    "analyze_pose_curvature.py",
    "run_adaptive_xz_piecewise_windows.ps1",
    "bridge_occluded_lane_segments.py"
)
foreach ($Name in $RequiredScripts) {
    $Path = Join-Path $PSScriptRoot $Name
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required script is missing: $Path"
    }
}

. (Join-Path $PSScriptRoot "kitti_odometry_workstation_input.ps1")
$KittiPaths = Resolve-KittiOdometryWorkstationInput `
    -DatasetRoot $DatasetRoot `
    -SequenceId $SequenceId
$Kitti = Test-KittiOdometrySequenceInput -InputPaths $KittiPaths
if ($EndFrame -gt [int]$Kitti.MaximumFrame) {
    throw "Sequence $SequenceId ends at $($Kitti.MaximumFrame), requested $EndFrame."
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputRoot = Join-Path $ProjectRoot (
        "workstation_outputs\final_seq${SequenceId}_${StartFrame}_${EndFrame}_$Stamp"
    )
}
if (Test-Path -LiteralPath $OutputRoot) {
    if (Get-ChildItem -LiteralPath $OutputRoot -Force) {
        throw "OutputRoot must be new or empty: $OutputRoot"
    }
} else {
    New-Item -ItemType Directory -Path $OutputRoot | Out-Null
}
$OutputRoot = (Resolve-Path -LiteralPath $OutputRoot).Path

function Invoke-CondaPython {
    param([string]$Label, [string[]]$PythonArgs)
    Write-Host ""
    Write-Host "[$Label]"
    & conda run --no-capture-output -n $EnvName python @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

$ScanDir = Join-Path $OutputRoot "01_lane_detection_and_tracking"
Invoke-CondaPython -Label "1/4 CLRNet and independent left/right observations" `
    -PythonArgs @(
        "scripts\scan_clrnet_lane_counts.py",
        "--dataset-name", "KITTI Odometry Sequence $SequenceId",
        "--image-dir", $Kitti.ImageDir,
        "--image-pattern", "{frame_id:06d}.png",
        "--frame-start", [string]$StartFrame,
        "--frame-end", [string]$EndFrame,
        "--calib", $Kitti.Calib,
        "--poses", $Kitti.Poses,
        "--segment-size", "5",
        "--minimum-candidates", "2",
        "--candidate-selection-mode", "temporal_independent",
        "--temporal-maximum-match-cost-m", "1.50",
        "--temporal-maximum-gap-frames", "3",
        "--minimum-bev-points-per-side", "4",
        "--local-z-range=3,50",
        "--fusion-x-range=-20,20",
        "--fusion-z-range=-20,50",
        "--clrnet-root", $ClrnetRoot,
        "--device", $Device,
        "--skip-recommendation",
        "--output-dir", $ScanDir
    )

$CurvatureDir = Join-Path $OutputRoot "02_pose_curvature"
$CurvatureArguments = @(
    "scripts\analyze_pose_curvature.py",
    "--poses", $Kitti.Poses,
    "--sequence-id", $SequenceId,
    "--start-frame", [string]$StartFrame,
    "--end-frame", [string]$EndFrame,
    "--window-length", [string]$WindowLength,
    "--window-stride", [string]$WindowStride,
    "--output-dir", $CurvatureDir
)
if ($StraightSeedRanges) {
    $CurvatureArguments += @("--straight-seed-ranges", $StraightSeedRanges)
}
Invoke-CondaPython -Label "2/4 pose curvature and explicit thresholds" `
    -PythonArgs $CurvatureArguments

$AdaptiveDir = Join-Path $OutputRoot "03_curve_models_and_fusion"
Write-Host ""
Write-Host "[3/4 straight polynomial, curve model comparison, G1 diagnostics]"
& (Join-Path $PSScriptRoot "run_adaptive_xz_piecewise_windows.ps1") `
    -EnvName $EnvName `
    -SelectedLanePoints (Join-Path $ScanDir "selected_lane_points.json") `
    -ScanJson (Join-Path $ScanDir "scan.json") `
    -Poses $Kitti.Poses `
    -Calib $Kitti.Calib `
    -SequenceId $SequenceId `
    -StartFrame $StartFrame `
    -EndFrame $EndFrame `
    -CoreEndFrame $EndFrame `
    -ReferenceFrame $EndFrame `
    -WindowLength $WindowLength `
    -WindowStride $WindowStride `
    -CurveModelPolicy "compare" `
    -CurvatureWindowsCsv (Join-Path $CurvatureDir "pose_curvature_windows.csv") `
    -OutputDir $AdaptiveDir
if ($LASTEXITCODE -ne 0) {
    throw "Curve model and fusion stage failed with exit code $LASTEXITCODE."
}

$BridgeDir = Join-Path $OutputRoot "04_occlusion_hypotheses"
Invoke-CondaPython -Label "4/4 low-confidence pre/post-occlusion bridge audit" `
    -PythonArgs @(
        "scripts\bridge_occluded_lane_segments.py",
        "--blended-nodes", (Join-Path $AdaptiveDir "blended_lane_nodes.csv"),
        "--output-dir", $BridgeDir
    )

$Scan = Get-Content (Join-Path $ScanDir "scan.json") -Raw | ConvertFrom-Json
$Curvature = Get-Content (Join-Path $CurvatureDir "CURVATURE_RESULT.json") -Raw | ConvertFrom-Json
$Adaptive = Get-Content (Join-Path $AdaptiveDir "STATUS.json") -Raw | ConvertFrom-Json
$Bridge = Get-Content (Join-Path $BridgeDir "BRIDGE_RESULT.json") -Raw | ConvertFrom-Json
$ModelRows = @(Import-Csv (Join-Path $AdaptiveDir "model_comparison.csv"))
$SelectedModelRows = @(
    $ModelRows | Where-Object { $_.selected_for_output -eq "True" }
)
$SelectedModelCounts = [ordered]@{}
foreach ($Row in $SelectedModelRows) {
    $Key = "{0}|{1}" -f $Row.classification, $Row.model
    if (-not $SelectedModelCounts.Contains($Key)) {
        $SelectedModelCounts[$Key] = 0
    }
    $SelectedModelCounts[$Key] += 1
}
$ContinuityRows = @(Import-Csv (Join-Path $AdaptiveDir "window_continuity.csv"))
$ContinuityPassCount = @(
    $ContinuityRows | Where-Object { $_.passes_continuity_gate -eq "True" }
).Count
$ExpectedWindowSideFits = 2 * [int]$Curvature.counts.windows
$FrameCount = $EndFrame - $StartFrame + 1
$Metrics = [ordered]@{
    status = [string]$Adaptive.status
    sequence_id = $SequenceId
    frames = @($StartFrame, $EndFrame)
    frame_count = $FrameCount
    coordinate_system = "metric Cartesian common-reference X/Z"
    curvature_thresholds = $Curvature.thresholds
    curvature_state_frames = $Curvature.counts.state_frames
    observation_coverage = [ordered]@{
        left_count = [int]$Scan.frames_with_left_observation
        left_fraction = [double]$Scan.frames_with_left_observation / $FrameCount
        right_count = [int]$Scan.frames_with_right_observation
        right_fraction = [double]$Scan.frames_with_right_observation / $FrameCount
        both_count = [int]$Scan.frames_with_selected_pair
        both_fraction = [double]$Scan.frames_with_selected_pair / $FrameCount
    }
    fitting_coverage = [ordered]@{
        window_count = [int]$Curvature.counts.windows
        expected_window_side_fits = $ExpectedWindowSideFits
        completed_window_side_fits = [int]$Adaptive.window_side_fits
        completed_fraction = if ($ExpectedWindowSideFits -gt 0) {
            [double]$Adaptive.window_side_fits / $ExpectedWindowSideFits
        } else { 0.0 }
    }
    selected_model_counts = $SelectedModelCounts
    continuity = [ordered]@{
        checked_interfaces = $ContinuityRows.Count
        passed_interfaces = $ContinuityPassCount
        passed_fraction = if ($ContinuityRows.Count -gt 0) {
            [double]$ContinuityPassCount / $ContinuityRows.Count
        } else { 0.0 }
    }
    bridge_audit = [ordered]@{
        candidate_gaps = [int]$Bridge.candidate_gaps
        accepted_low_confidence_hypotheses = [int]$Bridge.accepted_hypothesis_bridges
        rejected_gaps = [int]$Bridge.rejected_gaps
    }
    interpretation = [ordered]@{
        curvature = "pose-derived project classification, not a KITTI annotation"
        held_out_errors = "internal consistency against omitted CLRNet/IPM observations, not real-world lane accuracy"
        bridges = "dashed low-confidence geometry hypotheses, never detected observations"
    }
}
$MetricsPath = Join-Path $OutputRoot "FINAL_METRICS.json"
$Metrics | ConvertTo-Json -Depth 12 |
    Set-Content -LiteralPath $MetricsPath -Encoding UTF8
$Status = [ordered]@{
    status = [string]$Adaptive.status
    sequence_id = $SequenceId
    frames = @($StartFrame, $EndFrame)
    frame_count = $EndFrame - $StartFrame + 1
    lane_observations = [ordered]@{
        left = [int]$Scan.frames_with_left_observation
        right = [int]$Scan.frames_with_right_observation
        both = [int]$Scan.frames_with_selected_pair
    }
    curvature_thresholds = $Curvature.thresholds
    window_side_fits = [int]$Adaptive.window_side_fits
    accepted_low_confidence_bridges = [int]$Bridge.accepted_hypothesis_bridges
    previous_outputs_modified = $false
    limitations = @(
        "temporal_independent track IDs are project hypotheses, not CLRNet semantic IDs",
        "bridges are dashed low-confidence hypotheses, not detected observations",
        "held-out metrics measure consistency, not official lane accuracy"
    )
}
$StatusPath = Join-Path $OutputRoot "FINAL_STATUS.json"
$Status | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $StatusPath -Encoding UTF8

$BundleDir = Join-Path $OutputRoot "05_review_bundle"
New-Item -ItemType Directory -Path $BundleDir | Out-Null
$Files = @(
    $StatusPath,
    $MetricsPath,
    (Join-Path $ScanDir "scan.json"),
    (Join-Path $ScanDir "lane_counts.csv"),
    (Join-Path $CurvatureDir "CURVATURE_RESULT.json"),
    (Join-Path $CurvatureDir "pose_curvature_frames.csv"),
    (Join-Path $CurvatureDir "pose_curvature_windows.csv"),
    (Join-Path $CurvatureDir "pose_curvature_segments.csv"),
    (Join-Path $CurvatureDir "pose_curvature_overview.png"),
    (Join-Path $AdaptiveDir "RESULT.json"),
    (Join-Path $AdaptiveDir "STATUS.json"),
    (Join-Path $AdaptiveDir "window_plan.csv"),
    (Join-Path $AdaptiveDir "model_comparison.csv"),
    (Join-Path $AdaptiveDir "window_continuity.csv"),
    (Join-Path $AdaptiveDir "output_segments.csv"),
    (Join-Path $AdaptiveDir "skipped_items.csv"),
    (Join-Path $AdaptiveDir "adaptive_piecewise_xz_overview.png"),
    (Join-Path $BridgeDir "BRIDGE_RESULT.json"),
    (Join-Path $BridgeDir "occlusion_bridge_diagnostics.csv"),
    (Join-Path $BridgeDir "occlusion_bridge_nodes.csv"),
    (Join-Path $BridgeDir "occlusion_bridge_overview.png")
)
foreach ($File in $Files) {
    if (Test-Path -LiteralPath $File -PathType Leaf) {
        Copy-Item -LiteralPath $File -Destination $BundleDir
    }
}
$BundleZip = Join-Path $OutputRoot (
    "final_seq${SequenceId}_${StartFrame}_${EndFrame}_review_bundle.zip"
)
Compress-Archive -LiteralPath $BundleDir -DestinationPath $BundleZip

Write-Host ""
Write-Host "SURF FINAL WINDOW RUN FINISHED"
Write-Host "Output root: $OutputRoot"
Write-Host "Review ZIP: $BundleZip"
