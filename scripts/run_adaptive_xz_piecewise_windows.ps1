[CmdletBinding()]
param(
    [string]$EnvName = "surf2026-win",
    [Parameter(Mandatory = $true)]
    [string]$SelectedLanePoints,
    [string]$ScanJson = "",
    [Parameter(Mandatory = $true)]
    [string]$Poses,
    [Parameter(Mandatory = $true)]
    [string]$Calib,
    [string]$SequenceId = "01",
    [int]$StartFrame = 851,
    [int]$EndFrame = 1005,
    [int]$CoreEndFrame = 990,
    [int]$ReferenceFrame = -1,
    [int]$WindowLength = 15,
    [int]$WindowStride = 10,
    [double]$StraightMaxHeadingDeg = 1.5,
    [double]$CurveMinHeadingDeg = 3.0,
    [double]$TransitionRmseTieM = 0.005,
    [double]$ContinuityP95GateM = 0.75,
    [double]$ContinuityAngleGateDeg = 30.0,
    [double]$ContinuityRouteGapGateM = 0.50,
    [int]$PolynomialDegree = 2,
    [int]$MinimumValidFramesPerSide = 8,
    [int]$MaximumMissingRunFrames = 3,
    [double]$MaximumInternalOrderGapFrames = 3.0,
    [double]$MaximumInternalNodeGapM = 5.0,
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found. Open Anaconda PowerShell Prompt and retry."
}

foreach ($InputPath in @($SelectedLanePoints, $Poses, $Calib)) {
    if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
        throw "Required input file is missing: $InputPath"
    }
}
if ($ScanJson -and -not (Test-Path -LiteralPath $ScanJson -PathType Leaf)) {
    throw "Optional scan JSON is missing: $ScanJson"
}
if (-not $OutputDir) {
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputDir = Join-Path $ProjectRoot "workstation_outputs\adaptive_xz_sequence${SequenceId}_${StartFrame}_${EndFrame}_$Stamp"
}
if (Test-Path -LiteralPath $OutputDir) {
    $Existing = @(Get-ChildItem -LiteralPath $OutputDir -Force -ErrorAction SilentlyContinue)
    if ($Existing.Count -gt 0) {
        throw "Output directory must be new or empty: $OutputDir"
    }
}

$Arguments = @(
    "run", "--no-capture-output", "-n", $EnvName,
    "python", (Join-Path $PSScriptRoot "fit_adaptive_xz_piecewise.py"),
    "--selected-lane-points", $SelectedLanePoints,
    "--poses", $Poses,
    "--calib", $Calib,
    "--sequence-id", $SequenceId,
    "--start-frame", [string]$StartFrame,
    "--end-frame", [string]$EndFrame,
    "--core-end-frame", [string]$CoreEndFrame,
    "--window-length", [string]$WindowLength,
    "--window-stride", [string]$WindowStride,
    "--straight-max-heading-deg", [string]$StraightMaxHeadingDeg,
    "--curve-min-heading-deg", [string]$CurveMinHeadingDeg,
    "--transition-rmse-tie-m", [string]$TransitionRmseTieM,
    "--continuity-p95-gate-m", [string]$ContinuityP95GateM,
    "--continuity-angle-gate-deg", [string]$ContinuityAngleGateDeg,
    "--continuity-route-gap-gate-m", [string]$ContinuityRouteGapGateM,
    "--polynomial-degree", [string]$PolynomialDegree,
    "--minimum-valid-frames-per-side", [string]$MinimumValidFramesPerSide,
    "--maximum-missing-run-frames", [string]$MaximumMissingRunFrames,
    "--maximum-internal-order-gap-frames", [string]$MaximumInternalOrderGapFrames,
    "--maximum-internal-node-gap-m", [string]$MaximumInternalNodeGapM,
    "--output-dir", $OutputDir
)
if ($ReferenceFrame -ge 0) {
    $Arguments += @("--reference-frame", [string]$ReferenceFrame)
}
if ($ScanJson) {
    $Arguments += @("--scan-json", $ScanJson)
}

Write-Host "Running adaptive metric X/Z piecewise experiment..."
Write-Host "Frames: $StartFrame-$EndFrame; common reference: $(if ($ReferenceFrame -ge 0) { $ReferenceFrame } else { $EndFrame })"
Write-Host "Output: $OutputDir"
& conda @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Adaptive metric X/Z piecewise experiment failed with exit code $LASTEXITCODE."
}

$StatusPath = Join-Path $OutputDir "STATUS.json"
if (-not (Test-Path -LiteralPath $StatusPath -PathType Leaf)) {
    throw "Run returned without STATUS.json: $StatusPath"
}
$Status = Get-Content -LiteralPath $StatusPath -Raw | ConvertFrom-Json
Write-Host "ADAPTIVE X/Z EXPERIMENT FINISHED"
Write-Host "Status: $($Status.status)"
Write-Host "Window-side fits: $($Status.window_side_fits)"
Write-Host "Output root: $OutputDir"
