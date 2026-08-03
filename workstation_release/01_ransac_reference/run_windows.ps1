[CmdletBinding()]
param(
    [string]$EnvName = "surf2026-win",
    [string]$DatasetRoot = "F:\BaiduNetdiskDownload\kitti\odometry",
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [string]$DetectedJson = "",
    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"
$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RootDir

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found. Open Anaconda PowerShell Prompt and retry."
}
. (Join-Path $RootDir "scripts\kitti00_workstation_input.ps1")
$Kitti = Resolve-Kitti00WorkstationInput -DatasetRoot $DatasetRoot
$null = Test-Kitti00FirstFiveInput -InputPaths $Kitti -RequireCompleteSequence

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputRoot = Join-Path $RootDir "clean_outputs\ransac_reference_$Stamp"
}
if (Test-Path -LiteralPath $OutputRoot) {
    if (Get-ChildItem -LiteralPath $OutputRoot -Force) {
        throw "OutputRoot must be new or empty: $OutputRoot"
    }
} else {
    New-Item -ItemType Directory -Path $OutputRoot | Out-Null
}

function Invoke-CondaPython {
    param([string]$Label, [string[]]$PythonArgs)
    Write-Host ""
    Write-Host "[$Label]"
    & conda run --no-capture-output -n $EnvName python @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

if ([string]::IsNullOrWhiteSpace($DetectedJson)) {
    $PipelineDir = Join-Path $OutputRoot "01_live_clrnet_ipm_pose"
    Invoke-CondaPython -Label "1/2 Live CLRNet, point IPM and pose alignment" `
        -PythonArgs @(
            "scripts\run_full_point_pipeline.py",
            "--image-dir", $Kitti.ImageDir,
            "--image-pattern", "{frame_id:06d}.png",
            "--calib", $Kitti.Calib,
            "--poses", $Kitti.Poses,
            "--frame-ids", "0,1,2,3,4",
            "--reference-id", "4",
            "--local-x-range=-10,10",
            "--local-z-range=3,50",
            "--fusion-x-range=-15,15",
            "--fusion-z-range=-10,50",
            "--weights", "0.2,0.4,0.6,0.8,1.0",
            "--device", $Device,
            "--output-dir", $PipelineDir
        )
    $DetectedJson = Join-Path $PipelineDir "00_metadata\detected_lane_points.json"
} else {
    $DetectedJson = (Resolve-Path -LiteralPath $DetectedJson).Path
}

$RansacDir = Join-Path $OutputRoot "02_selected_optimized_ransac"
Invoke-CondaPython -Label "2/2 Fixed safety-audited optimized RANSAC" `
    -PythonArgs @(
        "scripts\run_selected_ransac_reference.py",
        "--clrnet-json", $DetectedJson,
        "--manual-json", (Join-Path $RootDir "annotations\kitti00_first5_manual_annotations.json"),
        "--calib", $Kitti.Calib,
        "--poses", $Kitti.Poses,
        "--output-dir", $RansacDir
    )

@{
    status = "complete"
    selected_method = "safety-audited side-aware cubic RANSAC"
    selected_config = "d3_b0.350_s0.015_c0.80_i1000_z10.0_qbalanced_r1"
    b_spline_used = $false
    output_root = $OutputRoot
    result = $RansacDir
} | ConvertTo-Json -Depth 4 | Set-Content `
    -LiteralPath (Join-Path $OutputRoot "STATUS.json") -Encoding UTF8

Write-Host ""
Write-Host "RANSAC REFERENCE FINISHED"
Write-Host "Output root: $OutputRoot"
Write-Host "Open: $RansacDir\01_points\metric_plot.png"
Write-Host "Audit: $RansacDir\00_audit\result.json"
Write-Host "Coordinates: $RansacDir\03_data\denoised_points.json"
