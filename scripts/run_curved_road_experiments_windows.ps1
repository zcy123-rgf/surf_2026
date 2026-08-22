[CmdletBinding()]
param(
    [string]$EnvName = "surf2026-win",
    [string]$DatasetRoot = "F:\BaiduNetdiskDownload\kitti\odometry",
    [string]$ExistingBatchRoot = "F:\2026_surf\workstation_outputs\sequences01_09_ego_adjacent_20260809_141900",
    [ValidateSet("00", "01", "02", "03", "04", "05", "06", "07", "08", "09")]
    [string]$SequenceId = "01",
    [int]$WindowCount = 30,
    [int]$SelectedRank = 1,
    [int]$TopCandidateCount = 100,
    [int]$MinimumValidFramesPerBlock = 10,
    [double]$MinimumTrajectoryTurnDegPerBlock = 1.0,
    [double]$MaximumOverlapP95GapM = 1.50,
    [double]$MaximumOverlapP95TangentGapDeg = 20.0,
    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"
$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RootDir

if ($WindowCount -lt 10) {
    throw "WindowCount must be at least 10. Use 30 for 305 unique frames."
}
if ($MinimumValidFramesPerBlock -lt 1 -or $MinimumValidFramesPerBlock -gt 15) {
    throw "MinimumValidFramesPerBlock must be between 1 and 15."
}
if ($MaximumOverlapP95GapM -le 0 -or $MaximumOverlapP95TangentGapDeg -le 0) {
    throw "Continuity gates must be positive."
}
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found. Open Anaconda PowerShell Prompt and retry."
}

$Required = @(
    "scripts\select_hierarchy_150_frames.py",
    "scripts\run_long_curved_road.py",
    "scripts\fit_extended_polynomial.py",
    "scripts\fit_extended_bspline.py"
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

$SequenceRoot = Join-Path $ExistingBatchRoot "sequence_$SequenceId"
$ScanJson = Join-Path $SequenceRoot "00_scan_frames\scan.json"
$SelectedPoints = Join-Path $SequenceRoot "00_scan_frames\selected_lane_points.json"
if (-not (Test-Path -LiteralPath $ScanJson -PathType Leaf)) {
    throw (
        "Existing full-sequence scan was not found: $ScanJson. " +
        "Pass -ExistingBatchRoot pointing to the completed Sequence 00-09 scan."
    )
}
if (-not (Test-Path -LiteralPath $SelectedPoints -PathType Leaf)) {
    throw "Saved CLRNet/IPM points were not found: $SelectedPoints"
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputRoot = Join-Path $RootDir (
        "workstation_outputs\curved_road_sequence_{0}_{1}windows_{2}" -f `
            $SequenceId, $WindowCount, $Stamp
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

Write-Host "Dataset: KITTI Odometry Sequence $SequenceId"
Write-Host "Full sequence frames: $($Kitti.FrameCount)"
Write-Host "Reusing scan: $ScanJson"
Write-Host "Window design: $WindowCount x 15 frames, stride 10"
Write-Host "Selected unique span: $((15 + ($WindowCount - 1) * 10)) frames"
Write-Host "Output: $OutputRoot"

$SelectionDir = Join-Path $OutputRoot "01_long_road_selection"
Invoke-CondaPython -Label "1/4 select a long road containing sustained curves" `
    -PythonArgs @(
        "scripts\select_hierarchy_150_frames.py",
        "--dataset-name", "KITTI Odometry Sequence $SequenceId",
        "--scan-json", $ScanJson,
        "--poses", $Kitti.Poses,
        "--image-dir", $Kitti.ImageDir,
        "--image-pattern", "{frame_id:06d}.png",
        "--block-size", "15",
        "--block-count", [string]$WindowCount,
        "--block-stride", "10",
        "--minimum-valid-frames-per-block", [string]$MinimumValidFramesPerBlock,
        "--selected-rank", [string]$SelectedRank,
        "--top-candidate-count", [string]$TopCandidateCount,
        "--ranking-mode", "sustained_curve",
        "--minimum-trajectory-turn-deg-per-block", [string]$MinimumTrajectoryTurnDegPerBlock,
        "--output-dir", $SelectionDir
    )

$SelectionJson = Join-Path $SelectionDir "selection.json"
$Selection = Get-Content -LiteralPath $SelectionJson -Raw | ConvertFrom-Json
if ($Selection.status -ne "selected") {
    throw (
        "No long-road candidate met the requested coverage gate. " +
        "Selection status: $($Selection.status). Review candidate_options.csv."
    )
}

$PolynomialDir = Join-Path $OutputRoot "02_global_polynomial_control"
Invoke-CondaPython -Label "2/4 global polynomial control" `
    -PythonArgs @(
        "scripts\fit_extended_polynomial.py",
        "--selection-json", $SelectionJson,
        "--poses", $Kitti.Poses,
        "--calib", $Kitti.Calib,
        "--output-dir", $PolynomialDir
    )

$BsplineDir = Join-Path $OutputRoot "03_global_bspline_control"
Invoke-CondaPython -Label "3/4 global B-spline control" `
    -PythonArgs @(
        "scripts\fit_extended_bspline.py",
        "--selection-json", $SelectionJson,
        "--poses", $Kitti.Poses,
        "--calib", $Kitti.Calib,
        "--output-dir", $BsplineDir
    )

$PiecewiseDir = Join-Path $OutputRoot "04_piecewise_long_road"
Invoke-CondaPython -Label "4/4 overlap-blended local polynomial and B-spline roads" `
    -PythonArgs @(
        "scripts\run_long_curved_road.py",
        "--selection-json", $SelectionJson,
        "--poses", $Kitti.Poses,
        "--calib", $Kitti.Calib,
        "--maximum-overlap-p95-gap-m", [string]$MaximumOverlapP95GapM,
        "--maximum-overlap-p95-tangent-gap-deg", [string]$MaximumOverlapP95TangentGapDeg,
        "--output-dir", $PiecewiseDir
    )

$MeetingDir = Join-Path $OutputRoot "05_meeting_figures"
New-Item -ItemType Directory -Force -Path $MeetingDir | Out-Null
$FigureCopies = @(
    @{
        Source = Join-Path $PolynomialDir "03_figures\polynomial_left_right.png"
        Name = "01_global_polynomial_control.png"
    },
    @{
        Source = Join-Path $BsplineDir "03_figures\bspline_left_right.png"
        Name = "02_global_bspline_control.png"
    },
    @{
        Source = Join-Path $PiecewiseDir "03_figures\polynomial_long_road.png"
        Name = "03_piecewise_polynomial_long_road.png"
    },
    @{
        Source = Join-Path $PiecewiseDir "03_figures\bspline_long_road.png"
        Name = "04_piecewise_bspline_long_road.png"
    }
)
foreach ($Item in $FigureCopies) {
    if (-not (Test-Path -LiteralPath $Item.Source -PathType Leaf)) {
        throw "Expected result image is missing: $($Item.Source)"
    }
    Copy-Item -LiteralPath $Item.Source -Destination (Join-Path $MeetingDir $Item.Name)
}

$PiecewiseStatus = Get-Content `
    -LiteralPath (Join-Path $PiecewiseDir "STATUS.json") `
    -Raw | ConvertFrom-Json
$Status = [ordered]@{
    status = "complete"
    dataset = "KITTI Odometry Sequence $SequenceId"
    full_sequence_frame_count = [int]$Kitti.FrameCount
    reused_full_sequence_clrnet_scan = $true
    selected_frames = $PiecewiseStatus.selected_frames
    window_count = [int]$PiecewiseStatus.windows_completed
    window_size = 15
    window_stride = 10
    road_segment_count = [int]$PiecewiseStatus.road_segment_count
    previous_outputs_modified = $false
    selection_json = $SelectionJson
    polynomial_global_control = $PolynomialDir
    bspline_global_control = $BsplineDir
    piecewise_long_road = $PiecewiseDir
    meeting_figures = $MeetingDir
}
$Status | ConvertTo-Json -Depth 6 | Set-Content `
    -LiteralPath (Join-Path $OutputRoot "STATUS.json") `
    -Encoding UTF8

$ZipPath = "$OutputRoot.zip"
if (Test-Path -LiteralPath $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
}
Compress-Archive -LiteralPath $OutputRoot -DestinationPath $ZipPath

Write-Host ""
Write-Host "CURVED ROAD EXPERIMENT FINISHED"
Write-Host "Selected frames: $($PiecewiseStatus.selected_frames -join '-')"
Write-Host "Windows completed: $($PiecewiseStatus.windows_completed)"
Write-Host "Road segments after continuity gates: $($PiecewiseStatus.road_segment_count)"
Write-Host "Meeting figures: $MeetingDir"
Write-Host "Upload this ZIP: $ZipPath"
