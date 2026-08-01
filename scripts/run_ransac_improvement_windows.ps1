[CmdletBinding()]
param(
    [string]$EnvName = "surf2026-win",
    [string]$DatasetRoot = "F:\BaiduNetdiskDownload\kitti\odometry",
    [string]$ImageDir = "",
    [string]$ImagePattern = "{frame_id:06d}.png",
    [string]$Calib = "",
    [string]$Poses = "",
    [string]$ManualJson = "",
    [string]$Provenance = "",
    [string]$OutputRoot = "",
    [ValidateSet("Compare", "Full")]
    [string]$Mode = "Compare",
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [string]$Weights = "0.2,0.4,0.6,0.8,1.0"
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
if ([string]::IsNullOrWhiteSpace($ImageDir)) { $ImageDir = $Kitti.ImageDir }
if ([string]::IsNullOrWhiteSpace($Calib)) { $Calib = $Kitti.Calib }
if ([string]::IsNullOrWhiteSpace($Poses)) { $Poses = $Kitti.Poses }
if ([string]::IsNullOrWhiteSpace($ManualJson)) {
    $ManualJson = Join-Path $RootDir "annotations\kitti00_first5_manual_annotations.json"
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputRoot = Join-Path $RootDir "workstation_outputs\ransac_improvement_$Stamp"
}

$Required = @($ImageDir, $Calib, $Poses, $ManualJson)
for ($FrameId = 0; $FrameId -lt 5; $FrameId++) {
    $FrameName = $ImagePattern.Replace(
        "{frame_id:06d}",
        $FrameId.ToString("000000")
    )
    $Required += Join-Path $ImageDir $FrameName
}
$Missing = $Required | Where-Object { -not (Test-Path -LiteralPath $_) }
if ($Missing) {
    throw "Required inputs are missing:`n$($Missing -join [Environment]::NewLine)"
}

Write-Host "Dataset root: $DatasetRoot"
Write-Host "Resolved layout: $($Verification.Layout)"
Write-Host "Verified full sequence: $($Verification.ImageCount) images, $($Verification.PoseRows) poses."
if (
    -not [string]::IsNullOrWhiteSpace($Provenance) -and
    -not (Test-Path -LiteralPath $Provenance)
) {
    throw "Provenance file does not exist: $Provenance"
}

if (Test-Path -LiteralPath $OutputRoot) {
    $Existing = Get-ChildItem -LiteralPath $OutputRoot -Force
    if ($Existing) {
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

$PipelineDir = Join-Path $OutputRoot "01_from_scratch_pipeline"
$PipelineArgs = @(
    "scripts\run_full_point_pipeline.py",
    "--image-dir", $ImageDir,
    "--image-pattern", $ImagePattern,
    "--calib", $Calib,
    "--poses", $Poses,
    "--frame-ids", "0,1,2,3,4",
    "--reference-id", "4",
    "--local-x-range=-10,10",
    "--local-z-range=3,50",
    "--fusion-x-range=-15,15",
    "--fusion-z-range=-10,50",
    "--weights", $Weights,
    "--device", $Device,
    "--output-dir", $PipelineDir
)
if (-not [string]::IsNullOrWhiteSpace($Provenance)) {
    $PipelineArgs += @("--provenance", $Provenance)
}
Invoke-CondaPython -Label "1/2 Live CLRNet, point BEV and pose alignment" `
    -PythonArgs $PipelineArgs

$ClrnetJson = Join-Path $PipelineDir "00_metadata\detected_lane_points.json"
$CompareDir = Join-Path $OutputRoot "02_ransac_method_comparison"
$CompareArgs = @(
    "scripts\evaluate_denoise_methods.py",
    "--clrnet-json", $ClrnetJson,
    "--manual-json", $ManualJson,
    "--calib", $Calib,
    "--poses", $Poses,
    "--output-dir", $CompareDir
)
Invoke-CondaPython -Label "2/2 RANSAC and nearby-method comparison" `
    -PythonArgs $CompareArgs

if ($Mode -eq "Full") {
    Write-Host ""
    Write-Host "Full mode performs the complete CPU-heavy parameter search and held-out audits."

    $SearchDir = Join-Path $OutputRoot "03_ransac_parameter_search"
    Invoke-CondaPython -Label "3/5 Full RANSAC parameter search" -PythonArgs @(
        "scripts\optimize_ransac_parameters.py",
        "--clrnet-json", $ClrnetJson,
        "--manual-json", $ManualJson,
        "--calib", $Calib,
        "--poses", $Poses,
        "--output-dir", $SearchDir
    )

    $SafetyDir = Join-Path $OutputRoot "04_ransac_safety_expansion"
    Invoke-CondaPython -Label "4/5 Safety-expanded RANSAC audit" -PythonArgs @(
        "scripts\optimize_ransac_safety_expansion.py",
        "--clrnet-json", $ClrnetJson,
        "--manual-json", $ManualJson,
        "--calib", $Calib,
        "--poses", $Poses,
        "--output-dir", $SafetyDir
    )

    $HeldoutDir = Join-Path $OutputRoot "05_ransac_heldout_audit"
    Invoke-CondaPython -Label "5/5 Independent shortlist audit" -PythonArgs @(
        "scripts\audit_ransac_shortlist.py",
        "--clrnet-json", $ClrnetJson,
        "--manual-json", $ManualJson,
        "--calib", $Calib,
        "--poses", $Poses,
        "--source-dir", $SearchDir,
        "--output-dir", $HeldoutDir
    )
}

Write-Host ""
Write-Host "Completed without reading any pre-generated result directory."
Write-Host "Output root: $OutputRoot"
Write-Host "Check first:"
Write-Host "  01_from_scratch_pipeline\00_metadata\audit.json"
Write-Host "  01_from_scratch_pipeline\04_pose_aligned_points\five_frame_metric_pose_fusion.png"
Write-Host "  02_ransac_method_comparison\comparison\main_comparison.png"
Write-Host "  02_ransac_method_comparison\00_audit\evaluation.json"
Write-Host "  02_ransac_method_comparison\00_audit\denoised_point_sets.json"
Write-Host "  02_ransac_method_comparison\00_audit\point_decisions.csv"
if ($Mode -eq "Full") {
    Write-Host "  04_ransac_safety_expansion\FINAL_RANSAC_REPORT_ZH.md"
    Write-Host "  05_ransac_heldout_audit\HELDOUT_AUDIT_REPORT_ZH.md"
}
