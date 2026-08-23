<#
.SYNOPSIS
Audits F:\surf_final without deleting or moving any project file.

.DESCRIPTION
Checks the final source/runtime, the completed Sequence 00-10 batch, the
domain-aligned metric report, root-level extras, caches and unregistered output
directories. Optionally creates a source/runtime backup that includes the
separate CLRNet runtime and its checkpoint but excludes large experiment
outputs. Every report is written under workstation_outputs with a timestamp.
#>

[CmdletBinding()]
param(
    [string]$ProjectRoot = "F:\surf_final",
    [switch]$CreateSourceBackup
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    throw "SURF final workspace was not found: $ProjectRoot"
}
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
if ($ProjectRoot -ne "F:\surf_final") {
    Write-Warning "Auditing a non-default workspace: $ProjectRoot"
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputsRoot = Join-Path $ProjectRoot "workstation_outputs"
New-Item -ItemType Directory -Force -Path $OutputsRoot | Out-Null
$ReportDir = Join-Path $OutputsRoot "workspace_audit_$Stamp"
New-Item -ItemType Directory -Path $ReportDir | Out-Null

$RequiredFiles = @(
    "requirements-windows.txt",
    "README_SURF_FINAL_ZH.md",
    "SURF_FINAL_PIPELINE_ZH.md",
    "SURF_FINAL_DECISIONS_ZH.md",
    "FIXED_EVALUATION_PROTOCOL_ZH.md",
    "FINAL_WORKSPACE_INVENTORY_ZH.md",
    "SOURCE_MANIFEST.json",
    "scripts\run_surf_final_sequence01_windows.ps1",
    "scripts\run_surf_final_sequence01_full_windows.ps1",
    "scripts\run_surf_final_all_sequences_windows.ps1",
    "scripts\run_fixed_project_metrics_windows.ps1",
    "scripts\evaluate_fixed_project_metrics.py",
    "scripts\audit_surf_final_workspace.ps1",
    "scripts\clean_surf_final_workspace.ps1",
    "surf_bev\detectors.py",
    "surf_bev\geometry.py",
    "CLRNet\configs\clrnet\clr_resnet18_culane.py",
    "CLRNet\weights\culane_r18.pth"
)
$RequiredDirectories = @(
    "annotations", "CLRNet", "scripts", "surf_bev", "workstation_outputs"
)

$RequiredRows = @()
foreach ($RelativePath in $RequiredDirectories) {
    $FullPath = Join-Path $ProjectRoot $RelativePath
    $RequiredRows += [pscustomobject]@{
        kind = "directory"
        relative_path = $RelativePath
        exists = Test-Path -LiteralPath $FullPath -PathType Container
        bytes = $null
    }
}
foreach ($RelativePath in $RequiredFiles) {
    $FullPath = Join-Path $ProjectRoot $RelativePath
    $Exists = Test-Path -LiteralPath $FullPath -PathType Leaf
    $RequiredRows += [pscustomobject]@{
        kind = "file"
        relative_path = $RelativePath
        exists = $Exists
        bytes = if ($Exists) { (Get-Item -LiteralPath $FullPath).Length } else { $null }
    }
}
$RequiredRows | Export-Csv -LiteralPath (Join-Path $ReportDir "required_items.csv") `
    -NoTypeInformation -Encoding UTF8

$AllowedTopLevel = @(
    "annotations", "backups", "CLRNet", "results", "scripts", "surf_bev",
    "workstation_outputs", "requirements-windows.txt", "README_SURF_FINAL_ZH.md",
    "SURF_FINAL_PIPELINE_ZH.md", "SURF_FINAL_DECISIONS_ZH.md",
    "FIXED_EVALUATION_PROTOCOL_ZH.md", "FINAL_WORKSPACE_INVENTORY_ZH.md",
    "SOURCE_MANIFEST.json"
)
$TopRows = @(Get-ChildItem -LiteralPath $ProjectRoot -Force | ForEach-Object {
    [pscustomobject]@{
        name = $_.Name
        kind = if ($_.PSIsContainer) { "directory" } else { "file" }
        allowed_final_item = $_.Name -in $AllowedTopLevel
        full_path = $_.FullName
    }
})
$TopRows | Export-Csv -LiteralPath (Join-Path $ReportDir "top_level_inventory.csv") `
    -NoTypeInformation -Encoding UTF8

$CacheNames = @(".git", ".idea", ".pytest_cache", "__pycache__", ".runtime")
$CacheRows = @(Get-ChildItem -LiteralPath $ProjectRoot -Force -Recurse `
    -ErrorAction SilentlyContinue | Where-Object {
        $_.Name -in $CacheNames -or $_.Extension -in @(".pyc", ".pyo")
    } | ForEach-Object {
        [pscustomobject]@{
            name = $_.Name
            kind = if ($_.PSIsContainer) { "directory" } else { "file" }
            full_path = $_.FullName
        }
    })
if ($CacheRows.Count -gt 0) {
    $CacheRows | Export-Csv -LiteralPath (Join-Path $ReportDir "cache_items.csv") `
        -NoTypeInformation -Encoding UTF8
}

$RegisteredOutputPrefixes = @(
    "final_seq01_851_1005_",
    "final_seq01_0_1100_",
    "surf_final_sequence01_851_1005_",
    "surf_final_sequence01_full_0000_1100_",
    "surf_final_all_sequences_",
    "fixed_project_metrics_",
    "pose_curvature_all_sequences_",
    "workspace_audit_",
    "workspace_cleanup_"
)
$OutputRows = @(Get-ChildItem -LiteralPath $OutputsRoot -Directory | ForEach-Object {
    $Registered = $false
    foreach ($Prefix in $RegisteredOutputPrefixes) {
        if ($_.Name.StartsWith($Prefix)) { $Registered = $true; break }
    }
    [pscustomobject]@{
        name = $_.Name
        registered_final_output = $Registered
        last_write_time = $_.LastWriteTime.ToString("o")
        full_path = $_.FullName
    }
})
$OutputRows | Export-Csv -LiteralPath (Join-Path $ReportDir "output_inventory.csv") `
    -NoTypeInformation -Encoding UTF8

$LatestBatch = Get-ChildItem -LiteralPath $OutputsRoot -Directory |
    Where-Object Name -Like "surf_final_all_sequences_*" |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
$BatchCheck = [ordered]@{found=$false;path=$null;status=$null;completed_sequences=$null;failed_sequences=$null}
if ($null -ne $LatestBatch) {
    $BatchStatusPath = Join-Path $LatestBatch.FullName "BATCH_STATUS.json"
    if (Test-Path -LiteralPath $BatchStatusPath -PathType Leaf) {
        $Batch = Get-Content -LiteralPath $BatchStatusPath -Raw | ConvertFrom-Json
        $BatchCheck = [ordered]@{
            found = $true
            path = $LatestBatch.FullName
            status = $Batch.status
            completed_sequences = $Batch.completed_sequences
            failed_sequences = $Batch.failed_sequences
        }
    }
}

$LatestMetrics = Get-ChildItem -LiteralPath $OutputsRoot -Directory |
    Where-Object Name -Like "fixed_project_metrics_*" |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
$MetricCheck = [ordered]@{found=$false;path=$null;status=$null;protocol=$null;evaluated_sequences=$null;failed_sequences=$null}
if ($null -ne $LatestMetrics) {
    $MetricStatusPath = Join-Path $LatestMetrics.FullName "STATUS.json"
    if (Test-Path -LiteralPath $MetricStatusPath -PathType Leaf) {
        $Metrics = Get-Content -LiteralPath $MetricStatusPath -Raw | ConvertFrom-Json
        $MetricCheck = [ordered]@{
            found = $true
            path = $LatestMetrics.FullName
            status = $Metrics.status
            protocol = $Metrics.metric_protocol_version
            evaluated_sequences = $Metrics.evaluated_sequence_count
            failed_sequences = $Metrics.failed_sequence_count
        }
    }
}

$Backup = $null
if ($CreateSourceBackup) {
    $BackupRoot = Join-Path $ProjectRoot "backups"
    New-Item -ItemType Directory -Force -Path $BackupRoot | Out-Null
    $BackupPath = Join-Path $BackupRoot "SURF_FINAL_SOURCE_RUNTIME_$Stamp.zip"
    $BackupItems = @(
        "annotations", "CLRNet", "scripts", "surf_bev",
        "requirements-windows.txt", "README_SURF_FINAL_ZH.md",
        "SURF_FINAL_PIPELINE_ZH.md", "SURF_FINAL_DECISIONS_ZH.md",
        "FIXED_EVALUATION_PROTOCOL_ZH.md", "FINAL_WORKSPACE_INVENTORY_ZH.md",
        "SOURCE_MANIFEST.json"
    ) | ForEach-Object { Join-Path $ProjectRoot $_ } |
        Where-Object { Test-Path -LiteralPath $_ }
    Compress-Archive -LiteralPath $BackupItems -DestinationPath $BackupPath
    $Hash = Get-FileHash -LiteralPath $BackupPath -Algorithm SHA256
    $Backup = [ordered]@{
        path = $BackupPath
        bytes = (Get-Item -LiteralPath $BackupPath).Length
        sha256 = $Hash.Hash
        includes_clrnet_runtime_and_checkpoint = (
            Test-Path -LiteralPath (Join-Path $ProjectRoot "CLRNet\weights\culane_r18.pth")
        )
        excludes_large_workstation_outputs = $true
    }
}

$MissingRequired = @($RequiredRows | Where-Object { -not $_.exists })
$UnexpectedTop = @($TopRows | Where-Object { -not $_.allowed_final_item })
$UnregisteredOutputs = @($OutputRows | Where-Object { -not $_.registered_final_output })
$BatchValid = (
    $BatchCheck.found -and $BatchCheck.status -eq "complete" -and
    [int]$BatchCheck.completed_sequences -eq 11 -and
    [int]$BatchCheck.failed_sequences -eq 0
)
$MetricsValid = (
    $MetricCheck.found -and $MetricCheck.status -eq "complete" -and
    $MetricCheck.protocol -eq "2.0-domain-aligned" -and
    [int]$MetricCheck.evaluated_sequences -eq 11 -and
    [int]$MetricCheck.failed_sequences -eq 0
)
$Ready = (
    $MissingRequired.Count -eq 0 -and $UnexpectedTop.Count -eq 0 -and
    $CacheRows.Count -eq 0 -and $UnregisteredOutputs.Count -eq 0 -and
    $BatchValid -and $MetricsValid
)

$Status = [ordered]@{
    status = if ($Ready) { "ready_for_final_retention" } else { "review_required" }
    project_root = $ProjectRoot
    audit_is_read_only_except_new_report_and_optional_backup = $true
    missing_required_count = $MissingRequired.Count
    unexpected_top_level_count = $UnexpectedTop.Count
    cache_item_count = $CacheRows.Count
    unregistered_output_directory_count = $UnregisteredOutputs.Count
    all_sequence_batch = $BatchCheck
    fixed_metrics = $MetricCheck
    source_runtime_backup = $Backup
    deletion_performed = $false
    warning = "Do not delete any flagged item until the audit report has been reviewed."
}
$Status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath `
    (Join-Path $ReportDir "WORKSPACE_AUDIT.json") -Encoding UTF8

$ReportZip = "$ReportDir.zip"
Compress-Archive -LiteralPath $ReportDir -DestinationPath $ReportZip
Write-Host "SURF FINAL WORKSPACE AUDIT FINISHED"
Write-Host "Status: $($Status.status)"
Write-Host "Missing required: $($Status.missing_required_count)"
Write-Host "Unexpected top-level: $($Status.unexpected_top_level_count)"
Write-Host "Cache items: $($Status.cache_item_count)"
Write-Host "Unregistered outputs: $($Status.unregistered_output_directory_count)"
if ($null -ne $Backup) { Write-Host "Source/runtime backup: $($Backup.path)" }
Write-Host "Upload this audit ZIP: $ReportZip"
