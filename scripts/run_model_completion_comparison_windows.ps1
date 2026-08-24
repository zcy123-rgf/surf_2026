param(
    [string]$EnvName = "surf2026-win",
    [string]$ProjectRoot = "F:\surf_final",
    [string]$WorkstationOutputs = "F:\surf_final\workstation_outputs",
    [string]$BsplineResultDir = "",
    [string]$PolynomialResultDir = ""
)

$ErrorActionPreference = "Stop"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputDir = Join-Path $WorkstationOutputs "sequence01_model_completion_comparison_$Stamp"
$Script = Join-Path $ProjectRoot "scripts\compare_model_completion_variants.py"

if (-not (Test-Path -LiteralPath $Script)) {
    throw "Comparison script not found: $Script"
}
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot "scripts\bridge_occluded_lane_segments.py"))) {
    throw "Bridge script not found under: $ProjectRoot\scripts"
}

$Arguments = @(
    "run", "--no-capture-output", "-n", $EnvName,
    "python", $Script,
    "--output-dir", $OutputDir
)

if ($BsplineResultDir -and $PolynomialResultDir) {
    $Arguments += @(
        "--bspline-result-dir", $BsplineResultDir,
        "--polynomial-result-dir", $PolynomialResultDir
    )
} elseif ($BsplineResultDir -or $PolynomialResultDir) {
    throw "Provide both BsplineResultDir and PolynomialResultDir, or leave both empty."
} else {
    $Arguments += @("--workstation-outputs", $WorkstationOutputs)
}

& conda @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Four-way comparison failed with exit code $LASTEXITCODE."
}

Write-Host "FOUR-WAY MODEL COMPLETION COMPARISON FINISHED"
Write-Host "Output root: $OutputDir"
Write-Host "Upload this ZIP: $OutputDir.zip"
