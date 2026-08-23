<#
.SYNOPSIS
Builds the fixed SURF evaluation scorecard from an existing completed batch.

.DESCRIPTION
This script does not rerun CLRNet and does not modify historical outputs. It
reads the saved full-Sequence result directories and writes a timestamped
post-hoc internal-consistency report.
#>

[CmdletBinding()]
param(
    [string]$EnvName = "surf2026-win",
    [string]$BatchRoot = "",
    [string]$OutputRoot = "",
    [double]$ResampleSpacingM = 0.5
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found. Open Anaconda PowerShell Prompt and retry."
}

if ([string]::IsNullOrWhiteSpace($BatchRoot)) {
    $Latest = Get-ChildItem (Join-Path $ProjectRoot "workstation_outputs") `
        -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "surf_final_all_sequences_*" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $Latest) {
        throw "No completed surf_final_all_sequences_* directory was found."
    }
    $BatchRoot = $Latest.FullName
}

$BatchSummary = Join-Path $BatchRoot "BATCH_SUMMARY.csv"
if (-not (Test-Path -LiteralPath $BatchSummary -PathType Leaf)) {
    throw "BATCH_SUMMARY.csv was not found: $BatchSummary"
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputRoot = Join-Path $ProjectRoot (
        "workstation_outputs\fixed_project_metrics_$Stamp"
    )
}
if (Test-Path -LiteralPath $OutputRoot) {
    if (Get-ChildItem -LiteralPath $OutputRoot -Force) {
        throw "OutputRoot must be new or empty: $OutputRoot"
    }
} else {
    New-Item -ItemType Directory -Path $OutputRoot | Out-Null
}

& conda run --no-capture-output -n $EnvName python `
    (Join-Path $PSScriptRoot "evaluate_fixed_project_metrics.py") `
    --batch-summary $BatchSummary `
    --output-dir $OutputRoot `
    --resample-spacing-m $ResampleSpacingM
if ($LASTEXITCODE -ne 0) {
    throw "Fixed project metric evaluation failed with exit code $LASTEXITCODE."
}

$Status = Get-Content (Join-Path $OutputRoot "STATUS.json") -Raw |
    ConvertFrom-Json
$Zip = "${OutputRoot}.zip"
Compress-Archive -LiteralPath $OutputRoot -DestinationPath $Zip

Write-Host ""
Write-Host "FIXED PROJECT METRICS FINISHED"
Write-Host "Evaluated Sequences: $($Status.evaluated_sequence_count)"
Write-Host "Failed Sequences: $($Status.failed_sequence_count)"
Write-Host "Historical outputs modified: $($Status.previous_outputs_modified)"
Write-Host "Output root: $OutputRoot"
Write-Host "Upload this ZIP: $Zip"
