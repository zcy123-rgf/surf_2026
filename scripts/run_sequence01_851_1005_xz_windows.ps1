<#
.SYNOPSIS
Runs the registered Sequence 01 metric-X/Z straight-curve-straight experiment.

.DESCRIPTION
Resolves the official KITTI Odometry Sequence 01 files, runs CLRNet on frames
851-1005, tracks ego-left and ego-right independently through short one-side
occlusions, and then runs adaptive polynomial/B-spline fitting directly in a
common metric Cartesian X/Z frame.  Every invocation creates a timestamped
output directory.  Existing outputs are never overwritten.

The reported core interval is 851-990.  Frames 991-1005 provide the terminal
15-frame window needed to evaluate and blend the exit portion without changing
the requested core conclusion.

.PARAMETER DatasetRoot
KITTI Odometry root containing the official separated downloads.

.PARAMETER ClrnetRoot
External CLRNet checkout containing configs and weights.  It is read only.

.PARAMETER OutputRoot
Optional new/empty result directory.  When omitted, a timestamped directory is
created below workstation_outputs.

.EXAMPLE
Set-Location F:\2026_surf
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& .\scripts\run_sequence01_851_1005_xz_windows.ps1

.EXAMPLE
Get-Help .\scripts\run_sequence01_851_1005_xz_windows.ps1 -Detailed
#>


[CmdletBinding()]
param(
    [string]$EnvName = "surf2026-win",
    [string]$DatasetRoot = "F:\BaiduNetdiskDownload\kitti\odometry",
    [string]$ClrnetRoot = "F:\2026_surf\CLRNet",
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [int]$StartFrame = 851,
    [int]$EndFrame = 1005,
    [int]$CoreEndFrame = 990,
    [double]$TemporalMaximumMatchCostM = 1.50,
    [int]$TemporalMaximumGapFrames = 3,
    [int]$MinimumValidFramesPerSide = 8,
    [int]$MaximumMissingRunFrames = 3,
    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

if ($StartFrame -lt 0 -or $EndFrame -lt $StartFrame) {
    throw "Frames must satisfy 0 <= StartFrame <= EndFrame."
}
if ($CoreEndFrame -lt $StartFrame -or $CoreEndFrame -gt $EndFrame) {
    throw "CoreEndFrame must fall inside the requested frame range."
}
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found. Open Anaconda PowerShell Prompt and retry."
}

$RequiredScripts = @(
    "kitti_odometry_workstation_input.ps1",
    "scan_clrnet_lane_counts.py",
    "run_adaptive_xz_piecewise_windows.ps1",
    "fit_adaptive_xz_piecewise.py"
)
foreach ($Name in $RequiredScripts) {
    $Path = Join-Path $PSScriptRoot $Name
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required script is missing: $Path"
    }
}
if (-not (Test-Path -LiteralPath $ClrnetRoot -PathType Container)) {
    throw "External CLRNet directory was not found: $ClrnetRoot"
}
$ClrnetRoot = (Resolve-Path -LiteralPath $ClrnetRoot).Path

. (Join-Path $PSScriptRoot "kitti_odometry_workstation_input.ps1")
$KittiPaths = Resolve-KittiOdometryWorkstationInput `
    -DatasetRoot $DatasetRoot `
    -SequenceId "01"
$Kitti = Test-KittiOdometrySequenceInput -InputPaths $KittiPaths
if ($EndFrame -gt [int]$Kitti.MaximumFrame) {
    throw (
        "Sequence 01 ends at frame $($Kitti.MaximumFrame), " +
        "but EndFrame $EndFrame was requested."
    )
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputRoot = Join-Path $ProjectRoot (
        "workstation_outputs\sequence01_851_1005_adaptive_xz_$Stamp"
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

$ScanDir = Join-Path $OutputRoot "01_temporal_independent_scan"
Invoke-CondaPython -Label "1/2 CLRNet scan and independent side tracking" `
    -PythonArgs @(
        "scripts\scan_clrnet_lane_counts.py",
        "--dataset-name", "KITTI Odometry Sequence 01",
        "--image-dir", $Kitti.ImageDir,
        "--image-pattern", "{frame_id:06d}.png",
        "--frame-start", [string]$StartFrame,
        "--frame-end", [string]$EndFrame,
        "--calib", $Kitti.Calib,
        "--poses", $Kitti.Poses,
        "--segment-size", "5",
        "--minimum-candidates", "2",
        "--candidate-selection-mode", "temporal_independent",
        "--temporal-maximum-match-cost-m", [string]$TemporalMaximumMatchCostM,
        "--temporal-maximum-gap-frames", [string]$TemporalMaximumGapFrames,
        "--minimum-bev-points-per-side", "4",
        "--local-z-range=3,50",
        "--fusion-x-range=-20,20",
        "--fusion-z-range=-20,50",
        "--clrnet-root", $ClrnetRoot,
        "--device", $Device,
        "--skip-recommendation",
        "--output-dir", $ScanDir
    )

$SelectedPoints = Join-Path $ScanDir "selected_lane_points.json"
$ScanJson = Join-Path $ScanDir "scan.json"
$AdaptiveDir = Join-Path $OutputRoot "02_adaptive_xz_piecewise"
Write-Host ""
Write-Host "[2/2 adaptive polynomial/B-spline fitting in metric X/Z]"
& (Join-Path $PSScriptRoot "run_adaptive_xz_piecewise_windows.ps1") `
    -EnvName $EnvName `
    -SelectedLanePoints $SelectedPoints `
    -ScanJson $ScanJson `
    -Poses $Kitti.Poses `
    -Calib $Kitti.Calib `
    -SequenceId "01" `
    -StartFrame $StartFrame `
    -EndFrame $EndFrame `
    -CoreEndFrame $CoreEndFrame `
    -ReferenceFrame $EndFrame `
    -WindowLength 15 `
    -WindowStride 10 `
    -MinimumValidFramesPerSide $MinimumValidFramesPerSide `
    -MaximumMissingRunFrames $MaximumMissingRunFrames `
    -OutputDir $AdaptiveDir
if ($LASTEXITCODE -ne 0) {
    throw "Adaptive metric X/Z fitting failed with exit code $LASTEXITCODE."
}

$ScanStatus = Get-Content -LiteralPath $ScanJson -Raw | ConvertFrom-Json
$AdaptiveStatusPath = Join-Path $AdaptiveDir "STATUS.json"
$AdaptiveResultPath = Join-Path $AdaptiveDir "RESULT.json"
$AdaptiveStatus = Get-Content -LiteralPath $AdaptiveStatusPath -Raw |
    ConvertFrom-Json

$MeetingDir = Join-Path $OutputRoot "03_meeting_bundle"
New-Item -ItemType Directory -Path $MeetingDir | Out-Null
$MeetingFiles = @(
    @{ Source = $ScanJson; Name = "scan_summary.json" },
    @{ Source = (Join-Path $ScanDir "lane_counts.csv"); Name = "lane_counts.csv" },
    @{ Source = $AdaptiveStatusPath; Name = "adaptive_STATUS.json" },
    @{ Source = $AdaptiveResultPath; Name = "adaptive_RESULT.json" },
    @{ Source = (Join-Path $AdaptiveDir "adaptive_piecewise_xz_overview.png"); Name = "adaptive_piecewise_xz_overview.png" },
    @{ Source = (Join-Path $AdaptiveDir "window_plan.csv"); Name = "window_plan.csv" },
    @{ Source = (Join-Path $AdaptiveDir "model_comparison.csv"); Name = "model_comparison.csv" },
    @{ Source = (Join-Path $AdaptiveDir "window_continuity.csv"); Name = "window_continuity.csv" },
    @{ Source = (Join-Path $AdaptiveDir "blended_lane_nodes.csv"); Name = "blended_lane_nodes.csv" },
    @{ Source = (Join-Path $AdaptiveDir "skipped_items.csv"); Name = "skipped_items.csv" }
)
foreach ($Item in $MeetingFiles) {
    if (Test-Path -LiteralPath $Item.Source -PathType Leaf) {
        Copy-Item -LiteralPath $Item.Source -Destination (Join-Path $MeetingDir $Item.Name)
    }
}

$RepresentativeFrames = @($StartFrame, 900, 950, $CoreEndFrame, $EndFrame) |
    Sort-Object -Unique
foreach ($FrameId in $RepresentativeFrames) {
    $Name = "frame_{0:D6}.png" -f $FrameId
    foreach ($Kind in @("original_frames", "all_candidates", "selected_pairs")) {
        $Source = Join-Path (Join-Path $ScanDir $Kind) $Name
        if (Test-Path -LiteralPath $Source -PathType Leaf) {
            Copy-Item -LiteralPath $Source -Destination (
                Join-Path $MeetingDir ("{0}_{1}" -f $Kind, $Name)
            )
        }
    }
}

$RunStatus = [ordered]@{
    status = [string]$AdaptiveStatus.status
    dataset = "KITTI Odometry Sequence 01"
    requested_frames = @($StartFrame, $EndFrame)
    reported_core_frames = @($StartFrame, $CoreEndFrame)
    candidate_selection_mode = "temporal_independent"
    coordinate_system = "metric Cartesian X/Z; X right, Z forward"
    frames_with_left_observation = [int]$ScanStatus.frames_with_left_observation
    frames_with_right_observation = [int]$ScanStatus.frames_with_right_observation
    frames_with_selected_pair = [int]$ScanStatus.frames_with_selected_pair
    frames_with_at_least_one_independent_side = [int]$ScanStatus.frames_with_at_least_one_independent_side
    window_side_fits = [int]$AdaptiveStatus.window_side_fits
    scan_output = $ScanDir
    adaptive_output = $AdaptiveDir
    previous_outputs_modified = $false
    warnings = @(
        "Project track IDs are not CLRNet or KITTI semantic lane IDs.",
        "An empty side slot is a missing observation, not a synthesized lane.",
        "Held-out errors measure consistency with CLRNet/IPM points, not official lane-boundary accuracy."
    )
}
$RunStatusPath = Join-Path $OutputRoot "RUN_STATUS.json"
$RunStatus | ConvertTo-Json -Depth 8 | Set-Content `
    -LiteralPath $RunStatusPath -Encoding UTF8
Copy-Item -LiteralPath $RunStatusPath -Destination (Join-Path $MeetingDir "RUN_STATUS.json")

$BundleZip = Join-Path $OutputRoot "sequence01_851_1005_meeting_bundle.zip"
Compress-Archive -LiteralPath $MeetingDir -DestinationPath $BundleZip

Write-Host ""
Write-Host "SEQUENCE 01 ADAPTIVE X/Z RUN FINISHED"
Write-Host "Status: $($RunStatus.status)"
Write-Host "Left observations: $($RunStatus.frames_with_left_observation)"
Write-Host "Right observations: $($RunStatus.frames_with_right_observation)"
Write-Host "Two-side frames: $($RunStatus.frames_with_selected_pair)"
Write-Host "Output root: $OutputRoot"
Write-Host "Upload this small ZIP: $BundleZip"
