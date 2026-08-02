[CmdletBinding()]
param(
    [string]$EnvName = "surf2026-win",
    [string]$DatasetRoot = "F:\BaiduNetdiskDownload\kitti\odometry",
    [string]$OutputRoot = "",
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [string]$SmoothingGrid = "0.0025,0.01,0.04,0.16"
)

$ErrorActionPreference = "Stop"
$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RootDir

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found. Open Anaconda PowerShell Prompt and retry."
}

. (Join-Path $PSScriptRoot "kitti00_workstation_input.ps1")
$Kitti = Resolve-Kitti00WorkstationInput -DatasetRoot $DatasetRoot
$Verification = Test-Kitti00FirstFiveInput `
    -InputPaths $Kitti `
    -RequireCompleteSequence

$FrameIdArray = 0..19
$FrameIds = $FrameIdArray -join ","
$ReferenceId = 19
$UniformWeights = (($FrameIdArray | ForEach-Object { "1.0" }) -join ",")

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputRoot = Join-Path $RootDir "workstation_outputs\first20_two_curves_$Stamp"
}

$Required = @($Kitti.ImageDir, $Kitti.Calib, $Kitti.Poses)
foreach ($FrameId in $FrameIdArray) {
    $Required += Join-Path $Kitti.ImageDir ($FrameId.ToString("000000") + ".png")
}
$Missing = $Required | Where-Object { -not (Test-Path -LiteralPath $_) }
if ($Missing) {
    throw "Required inputs are missing:`n$($Missing -join [Environment]::NewLine)"
}

# Every attempt receives a new root.  A non-empty path is rejected instead of
# being reused, so earlier five-frame and RANSAC outputs cannot be overwritten.
if (Test-Path -LiteralPath $OutputRoot) {
    if (Get-ChildItem -LiteralPath $OutputRoot -Force) {
        throw "OutputRoot must be new or empty: $OutputRoot"
    }
} else {
    New-Item -ItemType Directory -Path $OutputRoot | Out-Null
}

function Invoke-CondaPython {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [string[]]$PythonArgs
    )
    Write-Host ""
    Write-Host "[$Label]"
    & conda run --no-capture-output -n $EnvName python @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

Write-Host "Dataset root: $DatasetRoot"
Write-Host "Verified layout: $($Verification.Layout)"
Write-Host "Verified sequence: $($Verification.ImageCount) images and $($Verification.PoseRows) poses"
Write-Host "Frames: 000000-000019"
Write-Host "Reference frame: 000019"
Write-Host "New isolated output: $OutputRoot"

$PipelineDir = Join-Path $OutputRoot "01_from_scratch_pipeline"
Invoke-CondaPython -Label "1/2 Live CLRNet, point IPM and frame-19 pose alignment" `
    -PythonArgs @(
        "scripts\run_full_point_pipeline.py",
        "--image-dir", $Kitti.ImageDir,
        "--image-pattern", "{frame_id:06d}.png",
        "--calib", $Kitti.Calib,
        "--poses", $Kitti.Poses,
        "--frame-ids", $FrameIds,
        "--reference-id", "$ReferenceId",
        "--local-x-range=-10,10",
        "--local-z-range=3,50",
        "--fusion-x-range=-15,15",
        "--fusion-z-range=-20,50",
        "--weights", $UniformWeights,
        "--device", $Device,
        "--output-dir", $PipelineDir
    )

$AlignedJson = Join-Path $PipelineDir "00_metadata\aligned_lane_points.json"
$CurveDir = Join-Path $OutputRoot "02_two_curve_experiment"
Invoke-CondaPython -Label "2/2 Robust left/right cubic B-spline experiment" `
    -PythonArgs @(
        "scripts\fit_first5_two_curves.py",
        "--aligned-json", $AlignedJson,
        "--frame-ids", $FrameIds,
        "--reference-id", "$ReferenceId",
        "--x-range=-15,15",
        "--z-range=-20,50",
        "--smoothing-grid", $SmoothingGrid,
        "--output-dir", $CurveDir
    )

$StatusPath = Join-Path $OutputRoot "STATUS.json"
@{
    status = "complete"
    dataset = "KITTI Odometry Sequence 00"
    frames = $FrameIdArray
    reference_frame = $ReferenceId
    official_pose_path_length_m = 16.39456210618509
    official_pose_net_heading_change_deg = -2.183932679512209
    pose_geometry_interpretation = "near-straight candidate; image topology still requires inspection"
    previous_outputs_modified = $false
    pipeline_output = $PipelineDir
    curve_output = $CurveDir
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $StatusPath -Encoding UTF8

Write-Host ""
Write-Host "Completed. Previous outputs were not read or modified."
Write-Host "Output root: $OutputRoot"
Write-Host "Inspect lane identity first:"
Write-Host "  01_from_scratch_pipeline\02_clrnet_points\all_candidates\"
Write-Host "  01_from_scratch_pipeline\02_clrnet_points\selected_two\"
Write-Host "Open curve result:"
Write-Host "  02_two_curve_experiment\03_selected_result\20_frames_two_smooth_curves.png"
Write-Host "Check numerical data:"
Write-Host "  02_two_curve_experiment\03_selected_result\curve_data\left_curve_xz.csv"
Write-Host "  02_two_curve_experiment\03_selected_result\curve_data\right_curve_xz.csv"
Write-Host "Check audit:"
Write-Host "  02_two_curve_experiment\00_audit\audit.json"
Write-Host "  02_two_curve_experiment\00_audit\smoothing_selection.csv"
