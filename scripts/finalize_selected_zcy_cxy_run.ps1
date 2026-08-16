[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $OutputRoot -PathType Container)) {
    throw "OutputRoot does not exist: $OutputRoot"
}
$OutputRoot = (Resolve-Path -LiteralPath $OutputRoot).Path
$WindowDirs = @(
    Get-ChildItem -LiteralPath $OutputRoot -Directory |
        Where-Object Name -Like "sequence_*" |
        Sort-Object Name
)
if ($WindowDirs.Count -eq 0) {
    throw "No sequence result directory exists under: $OutputRoot"
}

$Rows = @()
foreach ($WindowDir in $WindowDirs) {
    $Parts = $WindowDir.Name -split "_"
    $SequenceId = $Parts[1]
    $StartFrame = [int]$Parts[2]
    $EndFrame = [int]$Parts[3]
    $Purpose = if ($Parts.Count -gt 4) { ($Parts[4..($Parts.Count - 1)] -join "_") } else { "selected" }
    $IdentityPath = Join-Path $WindowDir.FullName "03_zcy_identity_comparison\RESULT.json"
    $SemanticPath = Join-Path $WindowDir.FullName "07_cxy_semantic_evaluation\RESULT.json"
    if ((Test-Path -LiteralPath $IdentityPath -PathType Leaf) -and
        (Test-Path -LiteralPath $SemanticPath -PathType Leaf)) {
        $Identity = Get-Content -LiteralPath $IdentityPath -Raw | ConvertFrom-Json
        $Semantic = Get-Content -LiteralPath $SemanticPath -Raw | ConvertFrom-Json
        $Rows += [PSCustomObject]@{
            purpose = $Purpose
            sequence = $SequenceId
            start_frame = $StartFrame
            end_frame = $EndFrame
            status = "complete"
            ego_valid_rate = [double]$Identity.mode_summaries[0].valid_two_lane_rate
            temporal_valid_rate = [double]$Identity.mode_summaries[1].valid_two_lane_rate
            ego_continuity_p90_m = $Identity.mode_summaries[0].continuity_p90_m
            temporal_continuity_p90_m = $Identity.mode_summaries[1].continuity_p90_m
            semantic_status = [string]$Semantic.status
            semantic_left_reference_points = [int]$Semantic.association.assigned_left
            semantic_right_reference_points = [int]$Semantic.association.assigned_right
            output = $WindowDir.FullName
        }
    } else {
        $Missing = @($IdentityPath, $SemanticPath) |
            Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) }
        $Rows += [PSCustomObject]@{
            purpose = $Purpose
            sequence = $SequenceId
            start_frame = $StartFrame
            end_frame = $EndFrame
            status = "failed_or_incomplete"
            error = "Missing final result: $($Missing -join '; ')"
            output = $WindowDir.FullName
        }
    }
}

$Rows | Export-Csv -LiteralPath (Join-Path $OutputRoot "BATCH_SUMMARY.csv") `
    -NoTypeInformation -Encoding UTF8
$Complete = @($Rows | Where-Object { $_.status -eq "complete" }).Count
$Failed = $Rows.Count - $Complete
$Status = [ordered]@{
    status = if ($Failed -eq 0) { "complete" } elseif ($Complete -gt 0) { "partial" } else { "failed" }
    completed_windows = $Complete
    failed_windows = $Failed
    results = $Rows
    recovered_by_finalize_script = $true
    previous_outputs_modified = $false
}
$Status | ConvertTo-Json -Depth 8 | Set-Content `
    -LiteralPath (Join-Path $OutputRoot "STATUS.json") -Encoding UTF8

$BundleDir = Join-Path $OutputRoot "MEETING_BUNDLE"
if (Test-Path -LiteralPath $BundleDir) {
    throw "Meeting bundle already exists; use the existing bundle: $BundleDir"
}
New-Item -ItemType Directory -Path $BundleDir | Out-Null
Copy-Item -LiteralPath (Join-Path $OutputRoot "BATCH_SUMMARY.csv") -Destination $BundleDir
Copy-Item -LiteralPath (Join-Path $OutputRoot "STATUS.json") -Destination $BundleDir
foreach ($Row in $Rows | Where-Object { $_.status -eq "complete" }) {
    $Name = "sequence_{0}_{1:000000}_{2:000000}" -f $Row.sequence, $Row.start_frame, $Row.end_frame
    $Target = Join-Path $BundleDir $Name
    New-Item -ItemType Directory -Path $Target | Out-Null
    $Source = [string]$Row.output
    Copy-Item -LiteralPath (Join-Path $Source "03_zcy_identity_comparison\01_metrics") `
        -Destination (Join-Path $Target "identity_metrics") -Recurse
    Copy-Item -LiteralPath (Join-Path $Source "03_zcy_identity_comparison\02_figures") `
        -Destination (Join-Path $Target "identity_figures") -Recurse
    Copy-Item -LiteralPath (Join-Path $Source "03_zcy_identity_comparison\RESULT.json") `
        -Destination (Join-Path $Target "identity_RESULT.json")
    Copy-Item -LiteralPath (Join-Path $Source "05_polynomial\03_figures\polynomial_left_right.png") `
        -Destination (Join-Path $Target "polynomial.png")
    Copy-Item -LiteralPath (Join-Path $Source "06_bspline\03_figures\bspline_left_right.png") `
        -Destination (Join-Path $Target "bspline.png")
    Copy-Item -LiteralPath (Join-Path $Source "07_cxy_semantic_evaluation\01_metrics") `
        -Destination (Join-Path $Target "semantic_metrics") -Recurse
    Copy-Item -LiteralPath (Join-Path $Source "07_cxy_semantic_evaluation\02_figures") `
        -Destination (Join-Path $Target "semantic_figures") -Recurse
    Copy-Item -LiteralPath (Join-Path $Source "07_cxy_semantic_evaluation\RESULT.json") `
        -Destination (Join-Path $Target "semantic_RESULT.json")
}
$BundleZip = Join-Path $OutputRoot "zcy_cxy_meeting_bundle.zip"
Compress-Archive -LiteralPath $BundleDir -DestinationPath $BundleZip

Write-Host ""
Write-Host "EXISTING ZCY + CXY RUN FINALIZED"
Write-Host "Output root: $OutputRoot"
Write-Host "Completed windows: $Complete; incomplete windows: $Failed"
Write-Host "Upload this ZIP: $BundleZip"
if ($Complete -eq 0) {
    throw "No completed window was found. Review STATUS.json."
}
