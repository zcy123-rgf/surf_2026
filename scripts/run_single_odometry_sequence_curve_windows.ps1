[CmdletBinding()]
param(
    [string]$EnvName = "surf2026-win",
    [string]$DatasetRoot = "F:\BaiduNetdiskDownload\kitti\odometry",
    [Parameter(Mandatory = $true)]
    [ValidateSet("00", "01", "02", "03", "04", "05", "06", "07", "08", "09")]
    [string]$SequenceId,
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [int]$StartFrame = 0,
    [int]$EndFrame = -1,
    [int]$TopCandidateCount = 100,
    [int]$SelectedRank = 1,
    [int]$MinimumValidFramesPerBlock = 10,
    [ValidateSet("coverage", "sustained_curve")]
    [string]$RankingMode = "sustained_curve",
    [double]$MinimumTrajectoryTurnDegPerBlock = 1.0,
    [ValidateSet("outermost", "ego_adjacent")]
    [string]$CandidateSelectionMode = "ego_adjacent",
    [ValidateSet("both", "polynomial", "bspline")]
    [string]$Model = "both",
    [string]$OutputDir = "",
    [switch]$SkipPreflight
)

$ErrorActionPreference = "Stop"
$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RootDir

if ($StartFrame -lt 0) {
    throw "StartFrame must be non-negative."
}
if ($TopCandidateCount -lt 1 -or $SelectedRank -lt 1) {
    throw "TopCandidateCount and SelectedRank must be positive."
}
if ($SelectedRank -gt $TopCandidateCount) {
    throw "SelectedRank cannot exceed TopCandidateCount."
}
if ($MinimumValidFramesPerBlock -lt 1 -or $MinimumValidFramesPerBlock -gt 15) {
    throw "MinimumValidFramesPerBlock must be between 1 and 15."
}
if ($MinimumTrajectoryTurnDegPerBlock -le 0) {
    throw "MinimumTrajectoryTurnDegPerBlock must be positive."
}
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found. Open Anaconda PowerShell Prompt and retry."
}

. (Join-Path $PSScriptRoot "kitti_odometry_workstation_input.ps1")
$KittiPaths = Resolve-KittiOdometryWorkstationInput `
    -DatasetRoot $DatasetRoot `
    -SequenceId $SequenceId
$Kitti = Test-KittiOdometrySequenceInput -InputPaths $KittiPaths

if ($EndFrame -lt 0) {
    $EndFrame = [int]$Kitti.MaximumFrame
}
if ($EndFrame -lt $StartFrame) {
    throw "Frame range must satisfy StartFrame <= EndFrame."
}
if ($EndFrame -gt [int]$Kitti.MaximumFrame) {
    throw (
        "Sequence $SequenceId ends at frame $($Kitti.MaximumFrame), " +
        "but EndFrame $EndFrame was requested."
    )
}
if (($EndFrame - $StartFrame + 1) -lt 105) {
    throw "At least 105 consecutive frames are required for ten overlapping windows."
}

$RequiredScripts = @(
    "check_windows_env.py",
    "scan_clrnet_lane_counts.py",
    "select_hierarchy_150_frames.py",
    "extended_curve_model_common.py",
    "fit_extended_polynomial.py",
    "fit_extended_bspline.py",
    "run_weekly_lane_hierarchy.py",
    "fit_first5_two_curves.py"
)
foreach ($Name in $RequiredScripts) {
    $Path = Join-Path $PSScriptRoot $Name
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required script is missing: $Path"
    }
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputDir = Join-Path $RootDir (
        "workstation_outputs\sequence_{0}_curve_{1:000000}_{2:000000}_{3}" -f `
            $SequenceId, $StartFrame, $EndFrame, $Stamp
    )
}
if (Test-Path -LiteralPath $OutputDir) {
    if (Get-ChildItem -LiteralPath $OutputDir -Force) {
        throw "Output directory must be new or empty: $OutputDir"
    }
} else {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path
$DatasetName = "KITTI Odometry Sequence $SequenceId"

function Invoke-CondaPython {
    param([string]$Label, [string[]]$PythonArgs)
    Write-Host ""
    Write-Host "[$SequenceId][$Label]"
    & conda run --no-capture-output -n $EnvName python @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Sequence $SequenceId $Label failed with exit code $LASTEXITCODE."
    }
}

if (-not $SkipPreflight) {
    Invoke-CondaPython -Label "1/5 CUDA and CLRNet preflight" `
        -PythonArgs @(
            "scripts\check_windows_env.py",
            "--device", $Device,
            "--run-clrnet"
        )
}

$ScanDir = Join-Path $OutputDir "00_scan_frames"
Invoke-CondaPython -Label "2/5 CLRNet scan $StartFrame-$EndFrame" `
    -PythonArgs @(
        "scripts\scan_clrnet_lane_counts.py",
        "--dataset-name", $DatasetName,
        "--image-dir", $Kitti.ImageDir,
        "--image-pattern", "{frame_id:06d}.png",
        "--frame-start", [string]$StartFrame,
        "--frame-end", [string]$EndFrame,
        "--calib", $Kitti.Calib,
        "--poses", $Kitti.Poses,
        "--segment-size", "5",
        "--minimum-candidates", "2",
        "--candidate-selection-mode", $CandidateSelectionMode,
        "--minimum-bev-points-per-side", "4",
        "--local-z-range=3,50",
        "--fusion-x-range=-20,20",
        "--fusion-z-range=-20,50",
        "--device", $Device,
        "--skip-recommendation",
        "--output-dir", $ScanDir
    )
$ScanJson = Join-Path $ScanDir "scan.json"

$SelectionDir = Join-Path $OutputDir "01_ranked_option"
Invoke-CondaPython -Label "3/5 Rank candidate road sections" `
    -PythonArgs @(
        "scripts\select_hierarchy_150_frames.py",
        "--dataset-name", $DatasetName,
        "--scan-json", $ScanJson,
        "--poses", $Kitti.Poses,
        "--image-dir", $Kitti.ImageDir,
        "--image-pattern", "{frame_id:06d}.png",
        "--block-size", "15",
        "--block-count", "10",
        "--block-stride", "10",
        "--minimum-valid-frames-per-block", `
            [string]$MinimumValidFramesPerBlock,
        "--selected-rank", [string]$SelectedRank,
        "--top-candidate-count", [string]$TopCandidateCount,
        "--ranking-mode", $RankingMode,
        "--minimum-trajectory-turn-deg-per-block", `
            [string]$MinimumTrajectoryTurnDegPerBlock,
        "--output-dir", $SelectionDir
    )

$SelectionJson = Join-Path $SelectionDir "selection.json"
$Selection = Get-Content -LiteralPath $SelectionJson -Raw | ConvertFrom-Json
if ($Selection.status -ne "selected") {
    @{
        status = "stopped_at_coverage_gate"
        dataset = $DatasetName
        sequence_id = $SequenceId
        previous_outputs_modified = $false
        search_frames = @($StartFrame, $EndFrame)
        selected_rank = $SelectedRank
        selection_status = $Selection.status
        candidate_options_csv = $Selection.candidate_options_csv
        selection_json = $SelectionJson
        scan_json = $ScanJson
    } | ConvertTo-Json -Depth 7 | Set-Content `
        -LiteralPath (Join-Path $OutputDir "RUN_STATUS.json") -Encoding UTF8
    throw (
        "Sequence $SequenceId selected option did not meet the coverage gate. " +
        "Candidate files were preserved in $SelectionDir."
    )
}

$SharedModelArgs = @(
    "--selection-json", $SelectionJson,
    "--poses", $Kitti.Poses,
    "--calib", $Kitti.Calib,
    "--maximum-cv-folds", "20",
    "--curve-samples", "600"
)
$PolynomialOutput = $null
$BsplineOutput = $null

if ($Model -eq "both" -or $Model -eq "polynomial") {
    $PolynomialOutput = Join-Path $OutputDir "02_polynomial_only"
    Invoke-CondaPython -Label "4/5 Separate robust polynomial" `
        -PythonArgs (@(
            "scripts\fit_extended_polynomial.py",
            "--output-dir", $PolynomialOutput
        ) + $SharedModelArgs)
}

if ($Model -eq "both" -or $Model -eq "bspline") {
    $BsplineOutput = Join-Path $OutputDir "03_bspline_only"
    Invoke-CondaPython -Label "5/5 Separate robust cubic B-spline" `
        -PythonArgs (@(
            "scripts\fit_extended_bspline.py",
            "--output-dir", $BsplineOutput
        ) + $SharedModelArgs)
}

$RunStatus = @{
    status = "complete"
    dataset = $DatasetName
    sequence_id = $SequenceId
    previous_outputs_modified = $false
    search_frame_start = $StartFrame
    search_frame_end_inclusive = $EndFrame
    searched_frame_count = $EndFrame - $StartFrame + 1
    sequence_frame_count = [int]$Kitti.FrameCount
    selected_rank = $SelectedRank
    ranking_mode = $RankingMode
    candidate_selection_mode = $CandidateSelectionMode
    minimum_trajectory_turn_deg_per_block = $MinimumTrajectoryTurnDegPerBlock
    selected_frames = @(
        [int]$Selection.selected.start_frame,
        [int]$Selection.selected.end_frame
    )
    available_candidate_count = [int]$Selection.available_candidate_count
    exported_candidate_count = [int]$Selection.exported_candidate_count
    candidate_options_csv = [string]$Selection.candidate_options_csv
    scan_json = $ScanJson
    selection_json = $SelectionJson
    polynomial_output = $PolynomialOutput
    bspline_output = $BsplineOutput
    coordinate_changes = "none"
    output_root = $OutputDir
}
$RunStatus | ConvertTo-Json -Depth 7 | Set-Content `
    -LiteralPath (Join-Path $OutputDir "RUN_STATUS.json") -Encoding UTF8

Write-Host ""
Write-Host "SEQUENCE $SequenceId FINISHED"
Write-Host "Frames: $StartFrame-$EndFrame"
Write-Host "Selected: $($Selection.selected.start_frame)-$($Selection.selected.end_frame)"
Write-Host "Output: $OutputDir"

[PSCustomObject]$RunStatus
