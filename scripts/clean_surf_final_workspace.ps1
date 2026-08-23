<#
.SYNOPSIS
Removes only the exact SURF final-workspace items approved by the 2026-08-23 audit.

.DESCRIPTION
The script refuses to run unless the all-Sequence batch, protocol-2.0 metric
result and source/runtime backup are present. It preserves every formal model,
Sequence result, metric result and backup except the explicitly superseded
153156 metric directory. It then runs the read-only workspace audit again.
#>

[CmdletBinding()]
param(
    [string]$ProjectRoot = "F:\surf_final",
    [switch]$ConfirmCleanup
)

$ErrorActionPreference = "Stop"
if (-not $ConfirmCleanup) {
    throw "Cleanup was not confirmed. Rerun with -ConfirmCleanup after reviewing the audit."
}
if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    throw "SURF final workspace was not found: $ProjectRoot"
}
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
if ($ProjectRoot -ne "F:\surf_final") {
    throw "Safety gate rejected non-final path: $ProjectRoot"
}
$RootPrefix = $ProjectRoot.TrimEnd("\") + "\"

function Assert-SafeTarget {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    $Resolved = (Resolve-Path -LiteralPath $Path).Path
    if (-not $Resolved.StartsWith($RootPrefix)) {
        throw "Safety gate rejected target outside F:\surf_final: $Resolved"
    }
    return $Resolved
}

$OutputsRoot = Join-Path $ProjectRoot "workstation_outputs"
$LatestBatch = Get-ChildItem -LiteralPath $OutputsRoot -Directory |
    Where-Object Name -Like "surf_final_all_sequences_*" |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $LatestBatch) { throw "Completed all-Sequence batch was not found." }
$Batch = Get-Content -LiteralPath (Join-Path $LatestBatch.FullName "BATCH_STATUS.json") `
    -Raw | ConvertFrom-Json
if (
    $Batch.status -ne "complete" -or [int]$Batch.completed_sequences -ne 11 -or
    [int]$Batch.failed_sequences -ne 0
) { throw "All-Sequence batch safety gate failed." }

$LatestMetrics = Get-ChildItem -LiteralPath $OutputsRoot -Directory |
    Where-Object Name -Like "fixed_project_metrics_*" |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $LatestMetrics) { throw "Final fixed metric result was not found." }
$Metrics = Get-Content -LiteralPath (Join-Path $LatestMetrics.FullName "STATUS.json") `
    -Raw | ConvertFrom-Json
if (
    $Metrics.status -ne "complete" -or
    $Metrics.metric_protocol_version -ne "2.0-domain-aligned" -or
    [int]$Metrics.evaluated_sequence_count -ne 11 -or
    [int]$Metrics.failed_sequence_count -ne 0
) { throw "Protocol-2.0 fixed metric safety gate failed." }

$Backup = Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "backups") -File `
    -Filter "SURF_FINAL_SOURCE_RUNTIME_*.zip" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $Backup -or $Backup.Length -lt 1MB) {
    throw "Source/runtime backup safety gate failed."
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$CleanupDir = Join-Path $OutputsRoot "workspace_cleanup_$Stamp"
New-Item -ItemType Directory -Path $CleanupDir | Out-Null
$RemovedRows = @()

$ExactRelativeTargets = @(
    ".fixed_metrics_patch_17fda7c",
    "SURF全量实验审核包_20260823_140103.zip",
    "SURF固定评测脚本_17fda7c.zip",
    "SURF固定评测脚本_v2_513eff4.zip",
    "workstation_outputs\fixed_project_metrics_20260823_153156"
)
foreach ($RelativePath in $ExactRelativeTargets) {
    $Candidate = Join-Path $ProjectRoot $RelativePath
    $Safe = Assert-SafeTarget $Candidate
    if ($null -eq $Safe) { continue }
    $Kind = if ((Get-Item -LiteralPath $Safe).PSIsContainer) { "directory" } else { "file" }
    Remove-Item -LiteralPath $Safe -Recurse -Force
    $RemovedRows += [pscustomobject]@{
        category = "approved_exact_redundancy"
        kind = $Kind
        relative_path = $RelativePath
    }
}

$CacheDirectories = @(Get-ChildItem -LiteralPath $ProjectRoot -Directory -Force `
    -Recurse -ErrorAction SilentlyContinue | Where-Object Name -eq "__pycache__" |
    Sort-Object { $_.FullName.Length } -Descending)
foreach ($Directory in $CacheDirectories) {
    if (-not (Test-Path -LiteralPath $Directory.FullName)) { continue }
    $Safe = Assert-SafeTarget $Directory.FullName
    Remove-Item -LiteralPath $Safe -Recurse -Force
    $RemovedRows += [pscustomobject]@{
        category = "regenerable_python_cache"
        kind = "directory"
        relative_path = $Safe.Substring($RootPrefix.Length)
    }
}
$LooseBytecode = @(Get-ChildItem -LiteralPath $ProjectRoot -File -Force -Recurse `
    -ErrorAction SilentlyContinue | Where-Object Extension -in @(".pyc", ".pyo"))
foreach ($File in $LooseBytecode) {
    if (-not (Test-Path -LiteralPath $File.FullName)) { continue }
    $Safe = Assert-SafeTarget $File.FullName
    Remove-Item -LiteralPath $Safe -Force
    $RemovedRows += [pscustomobject]@{
        category = "regenerable_python_bytecode"
        kind = "file"
        relative_path = $Safe.Substring($RootPrefix.Length)
    }
}

$RemovedRows | Export-Csv -LiteralPath (Join-Path $CleanupDir "removed_items.csv") `
    -NoTypeInformation -Encoding UTF8
$CleanupStatus = [ordered]@{
    status = "complete"
    project_root = $ProjectRoot
    removed_item_count = $RemovedRows.Count
    preserved_full_sequence01 = "workstation_outputs\surf_final_sequence01_full_0000_1100_20260823_015856"
    preserved_all_sequences = $LatestBatch.FullName
    preserved_fixed_metrics = $LatestMetrics.FullName
    preserved_source_runtime_backup = $Backup.FullName
    policy = "only exact audited redundancies and regenerable Python caches were removed"
}
$CleanupStatus | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath `
    (Join-Path $CleanupDir "CLEANUP_STATUS.json") -Encoding UTF8

Write-Host "SURF FINAL WORKSPACE CLEANUP FINISHED"
Write-Host "Removed audited redundancies/cache entries: $($RemovedRows.Count)"
Write-Host "Cleanup report: $CleanupDir"
Write-Host "Running final read-only audit..."
& (Join-Path $PSScriptRoot "audit_surf_final_workspace.ps1") `
    -ProjectRoot $ProjectRoot
