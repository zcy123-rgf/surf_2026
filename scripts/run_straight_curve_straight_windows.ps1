[CmdletBinding()]
param(
    [string]$EnvName = "surf2026-win",
    [string]$DatasetRoot = "F:\BaiduNetdiskDownload\kitti\odometry",
    [string]$ExistingBatchRoot = "F:\2026_surf\workstation_outputs\sequences01_09_ego_adjacent_20260809_141900",
    [ValidateSet("00", "01", "02", "03", "04", "05", "06", "07", "08", "09")]
    [string]$SequenceId = "02",
    [int]$WindowCount = 10,
    [int]$MinimumValidFramesPerBlock = 10,
    [double]$MaximumStraightTurnDegPerBlock = 1.5,
    [double]$MinimumCurveTurnDegPerBlock = 3.0,
    [int]$MinimumMiddleCurveBlocks = 2,
    [double]$TemporalMaximumMatchCostM = 1.5,
    [int]$TemporalMaximumGapFrames = 3,
    [double]$MaximumOverlapP95GapM = 1.5,
    [double]$MaximumOverlapP95TangentGapDeg = 20.0,
    [string]$ClrnetRoot = "F:\2026_surf\CLRNet",
    [string]$Device = "cuda",
    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"
$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RootDir

if ($WindowCount -lt 6) {
    throw "WindowCount must be at least 6 so entry, curve and exit phases exist."
}
if ($MinimumValidFramesPerBlock -lt 1 -or $MinimumValidFramesPerBlock -gt 15) {
    throw "MinimumValidFramesPerBlock must be between 1 and 15."
}
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found. Open Anaconda PowerShell Prompt and retry."
}

$Required = @(
    "scripts\kitti_odometry_workstation_input.ps1",
    "scripts\select_hierarchy_150_frames.py",
    "scripts\scan_clrnet_lane_counts.py",
    "scripts\fit_extended_polynomial.py",
    "scripts\fit_extended_bspline.py",
    "scripts\run_long_curved_road.py"
)
foreach ($Relative in $Required) {
    if (-not (Test-Path -LiteralPath (Join-Path $RootDir $Relative) -PathType Leaf)) {
        throw "Required script is missing: $Relative"
    }
}

. (Join-Path $PSScriptRoot "kitti_odometry_workstation_input.ps1")
$KittiPaths = Resolve-KittiOdometryWorkstationInput `
    -DatasetRoot $DatasetRoot `
    -SequenceId $SequenceId
$Kitti = Test-KittiOdometrySequenceInput -InputPaths $KittiPaths
if (-not (Test-Path -LiteralPath $ClrnetRoot -PathType Container)) {
    throw "External CLRNet directory was not found: $ClrnetRoot"
}
$ClrnetRoot = (Resolve-Path -LiteralPath $ClrnetRoot).Path

$SequenceRoot = Join-Path $ExistingBatchRoot "sequence_$SequenceId"
$ExistingScanJson = Join-Path $SequenceRoot "00_scan_frames\scan.json"
if (-not (Test-Path -LiteralPath $ExistingScanJson -PathType Leaf)) {
    throw (
        "Existing full-sequence scan was not found: $ExistingScanJson. " +
        "ExistingBatchRoot must point to the completed Sequence 00-09 batch."
    )
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputRoot = Join-Path $RootDir (
        "workstation_outputs\straight_curve_straight_sequence_{0}_{1}" -f `
            $SequenceId, $Stamp
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

$TotalUniqueFrames = 15 + ($WindowCount - 1) * 10
Write-Host "Dataset: KITTI Odometry Sequence $SequenceId"
Write-Host "Selection target: straight -> curve -> straight"
Write-Host "Window design: $WindowCount x 15 frames, stride 10"
Write-Host "Unique frame span: $TotalUniqueFrames"
Write-Host "Output: $OutputRoot"

# Stage 1 uses the completed full-sequence scan only to locate a suitable road.
$CoarseDir = Join-Path $OutputRoot "00_pose_and_coverage_selection"
Invoke-CondaPython -Label "1/7 locate a straight-curve-straight road" `
    -PythonArgs @(
        "scripts\select_hierarchy_150_frames.py",
        "--dataset-name", "KITTI Odometry Sequence $SequenceId",
        "--scan-json", $ExistingScanJson,
        "--poses", $Kitti.Poses,
        "--image-dir", $Kitti.ImageDir,
        "--image-pattern", "{frame_id:06d}.png",
        "--block-size", "15",
        "--block-count", [string]$WindowCount,
        "--block-stride", "10",
        "--minimum-valid-frames-per-block", [string]$MinimumValidFramesPerBlock,
        "--ranking-mode", "straight_curve_straight",
        "--maximum-straight-turn-deg-per-block", [string]$MaximumStraightTurnDegPerBlock,
        "--minimum-curve-turn-deg-per-block", [string]$MinimumCurveTurnDegPerBlock,
        "--minimum-middle-curve-blocks", [string]$MinimumMiddleCurveBlocks,
        "--selected-rank", "1",
        "--top-candidate-count", "100",
        "--output-dir", $CoarseDir
    )

$CoarseJson = Join-Path $CoarseDir "selection.json"
$Coarse = Get-Content -LiteralPath $CoarseJson -Raw | ConvertFrom-Json
if ($Coarse.status -ne "selected") {
    throw "No candidate passed the requested coverage gate. Review candidate_options.csv."
}
if (-not [bool]$Coarse.selected.straight_curve_straight_gate) {
    throw (
        "The best available candidate did not pass the registered straight-curve-straight gate. " +
        "Do not report it as that road pattern."
    )
}
$StartFrame = [int]$Coarse.selected.start_frame
$EndFrame = [int]$Coarse.selected.end_frame

# Stage 2 reruns only the selected span and creates project-level temporal IDs.
$TrackScanDir = Join-Path $OutputRoot "01_temporal_lane_tracks"
Invoke-CondaPython -Label "2/7 rerun CLRNet and associate project lane tracks" `
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
        "--candidate-selection-mode", "temporal_ego",
        "--temporal-maximum-match-cost-m", [string]$TemporalMaximumMatchCostM,
        "--temporal-maximum-gap-frames", [string]$TemporalMaximumGapFrames,
        "--minimum-bev-points-per-side", "4",
        "--local-z-range=3,50",
        "--fusion-x-range=-20,20",
        "--fusion-z-range=-20,50",
        "--clrnet-root", $ClrnetRoot,
        "--device", $Device,
        "--skip-recommendation",
        "--output-dir", $TrackScanDir
    )

# Stage 3 materializes windows from the newly tracked points, not the old pair.
$TrackedSelectionDir = Join-Path $OutputRoot "02_tracked_window_selection"
Invoke-CondaPython -Label "3/7 validate tracked windows" `
    -PythonArgs @(
        "scripts\select_hierarchy_150_frames.py",
        "--dataset-name", "KITTI Odometry Sequence $SequenceId",
        "--scan-json", (Join-Path $TrackScanDir "scan.json"),
        "--poses", $Kitti.Poses,
        "--image-dir", $Kitti.ImageDir,
        "--image-pattern", "{frame_id:06d}.png",
        "--block-size", "15",
        "--block-count", [string]$WindowCount,
        "--block-stride", "10",
        "--minimum-valid-frames-per-block", [string]$MinimumValidFramesPerBlock,
        "--ranking-mode", "straight_curve_straight",
        "--maximum-straight-turn-deg-per-block", [string]$MaximumStraightTurnDegPerBlock,
        "--minimum-curve-turn-deg-per-block", [string]$MinimumCurveTurnDegPerBlock,
        "--minimum-middle-curve-blocks", [string]$MinimumMiddleCurveBlocks,
        "--selected-rank", "1",
        "--top-candidate-count", "1",
        "--output-dir", $TrackedSelectionDir
    )

$TrackedSelectionJson = Join-Path $TrackedSelectionDir "selection.json"
$TrackedSelection = Get-Content -LiteralPath $TrackedSelectionJson -Raw | ConvertFrom-Json
if ($TrackedSelection.status -ne "selected") {
    throw (
        "Temporal association completed, but too few tracked frames passed a window gate. " +
        "Keep this output for diagnosis; do not claim a successful fusion."
    )
}

$PolynomialDir = Join-Path $OutputRoot "03_polynomial"
Invoke-CondaPython -Label "4/7 polynomial control" `
    -PythonArgs @(
        "scripts\fit_extended_polynomial.py",
        "--selection-json", $TrackedSelectionJson,
        "--poses", $Kitti.Poses,
        "--calib", $Kitti.Calib,
        "--output-dir", $PolynomialDir
    )

$BsplineDir = Join-Path $OutputRoot "04_bspline"
Invoke-CondaPython -Label "5/7 B-spline control" `
    -PythonArgs @(
        "scripts\fit_extended_bspline.py",
        "--selection-json", $TrackedSelectionJson,
        "--poses", $Kitti.Poses,
        "--calib", $Kitti.Calib,
        "--output-dir", $BsplineDir
    )

$PiecewiseDir = Join-Path $OutputRoot "05_piecewise_fusion"
Invoke-CondaPython -Label "6/7 overlap-blended local road fusion" `
    -PythonArgs @(
        "scripts\run_long_curved_road.py",
        "--selection-json", $TrackedSelectionJson,
        "--poses", $Kitti.Poses,
        "--calib", $Kitti.Calib,
        "--maximum-overlap-p95-gap-m", [string]$MaximumOverlapP95GapM,
        "--maximum-overlap-p95-tangent-gap-deg", [string]$MaximumOverlapP95TangentGapDeg,
        "--output-dir", $PiecewiseDir
    )

$MeetingDir = Join-Path $OutputRoot "06_meeting_figures"
New-Item -ItemType Directory -Force -Path $MeetingDir | Out-Null
$MiddleFrame = [int][math]::Floor(($StartFrame + $EndFrame) / 2)
$Copies = @(
    @{
        Source = Join-Path $TrackScanDir ("selected_pairs\frame_{0:D6}.png" -f $StartFrame)
        Name = "01_entry_straight_selected_pair.png"
    },
    @{
        Source = Join-Path $TrackScanDir ("selected_pairs\frame_{0:D6}.png" -f $MiddleFrame)
        Name = "02_middle_curve_selected_pair.png"
    },
    @{
        Source = Join-Path $TrackScanDir ("selected_pairs\frame_{0:D6}.png" -f $EndFrame)
        Name = "03_exit_straight_selected_pair.png"
    },
    @{
        Source = Join-Path $PolynomialDir "03_figures\polynomial_left_right.png"
        Name = "04_polynomial.png"
    },
    @{
        Source = Join-Path $BsplineDir "03_figures\bspline_left_right.png"
        Name = "05_bspline.png"
    },
    @{
        Source = Join-Path $PiecewiseDir "03_figures\polynomial_long_road.png"
        Name = "06_piecewise_polynomial.png"
    },
    @{
        Source = Join-Path $PiecewiseDir "03_figures\bspline_long_road.png"
        Name = "07_piecewise_bspline.png"
    }
)
foreach ($Item in $Copies) {
    if (-not (Test-Path -LiteralPath $Item.Source -PathType Leaf)) {
        throw "Expected result is missing: $($Item.Source)"
    }
    Copy-Item -LiteralPath $Item.Source -Destination (Join-Path $MeetingDir $Item.Name)
}
Copy-Item -LiteralPath (Join-Path $TrackScanDir "lane_counts.csv") `
    -Destination (Join-Path $MeetingDir "temporal_track_audit.csv")

$Scan = Get-Content -LiteralPath (Join-Path $TrackScanDir "scan.json") -Raw |
    ConvertFrom-Json
$TrackRows = @($Scan.counts)
$PiecewiseStatus = Get-Content -LiteralPath (Join-Path $PiecewiseDir "STATUS.json") -Raw |
    ConvertFrom-Json
$Status = [ordered]@{
    status = "complete"
    dataset = "KITTI Odometry Sequence $SequenceId"
    selected_frames = @($StartFrame, $EndFrame)
    selected_unique_frame_count = $EndFrame - $StartFrame + 1
    road_pattern_gate = "straight_curve_straight"
    entry_mean_trajectory_turn_deg = [double]$TrackedSelection.selected.entry_mean_trajectory_turn_deg
    middle_peak_trajectory_turn_deg = [double]$TrackedSelection.selected.middle_peak_trajectory_turn_deg
    exit_mean_trajectory_turn_deg = [double]$TrackedSelection.selected.exit_mean_trajectory_turn_deg
    frames_with_more_than_two_clrnet_candidates = @(
        $TrackRows | Where-Object { [int]$_.candidate_count -gt 2 }
    ).Count
    frames_with_valid_temporal_pair = @(
        $TrackRows | Where-Object { [bool]$_.eligible_for_two_curve_fit }
    ).Count
    project_track_ids_are_official = $false
    project_track_id_definition = (
        "ego_left/ego_right continuity IDs created after CLRNet by pose-aligned " +
        "metric curve matching; CLRNet candidate indices are frame-local"
    )
    polynomial_output = $PolynomialDir
    bspline_output = $BsplineDir
    piecewise_output = $PiecewiseDir
    road_segment_count_after_continuity_gates = [int]$PiecewiseStatus.road_segment_count
    previous_outputs_modified = $false
    warnings = @(
        "A project track ID is not a semantic road-map lane ID.",
        "Temporal consistency alone cannot prove that a candidate is not a sidewalk edge.",
        "All accuracy values remain consistency diagnostics until manual labels are supplied."
    )
}
$Status | ConvertTo-Json -Depth 8 | Set-Content `
    -LiteralPath (Join-Path $OutputRoot "STATUS.json") -Encoding UTF8

$ZipPath = "$OutputRoot.zip"
Compress-Archive -LiteralPath $OutputRoot -DestinationPath $ZipPath

Write-Host ""
Write-Host "STRAIGHT-CURVE-STRAIGHT EXPERIMENT FINISHED"
Write-Host "Selected frames: $StartFrame-$EndFrame"
Write-Host "Valid temporal pairs: $($Status.frames_with_valid_temporal_pair)"
Write-Host "Meeting figures: $MeetingDir"
Write-Host "Upload this ZIP: $ZipPath"
