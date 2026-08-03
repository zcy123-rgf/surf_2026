[CmdletBinding()]
param(
    [string]$EnvName = "surf2026-win",
    [string]$SelectionRoot = ""
)

$ErrorActionPreference = "Stop"
$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RootDir

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found. Open Anaconda PowerShell Prompt and retry."
}

if ([string]::IsNullOrWhiteSpace($SelectionRoot)) {
    $Selection = Get-ChildItem (Join-Path $RootDir "workstation_outputs") -Directory |
        Where-Object Name -Like "hierarchy_selection_*" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $Selection) {
        throw "No hierarchy_selection_* output exists."
    }
    $SelectionRoot = $Selection.FullName
}
$SelectionRoot = (Resolve-Path -LiteralPath $SelectionRoot).Path
$ScanRoot = Join-Path $SelectionRoot "00_search_scans"
if (-not (Test-Path -LiteralPath $ScanRoot -PathType Container)) {
    throw "Search scan directory is missing: $ScanRoot"
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputRoot = Join-Path $RootDir "workstation_outputs\lane_semantic_review_$Stamp"
$OriginalOut = Join-Path $OutputRoot "01_original_samples"
$CandidateOut = Join-Path $OutputRoot "02_clrnet_candidate_samples"
New-Item -ItemType Directory -Force $OriginalOut,$CandidateOut | Out-Null

$Groups = @(
    @{
        Name = "negative_control_000000_000104"
        Scan = "frames_000000_000169"
        Frames = @(0, 25, 50, 75, 104)
    },
    @{
        Name = "candidate_000143_000164"
        Scan = "frames_000000_000169"
        Frames = @(143, 147, 151, 155, 160, 164)
    },
    @{
        Name = "candidate_001549_001578"
        Scan = "frames_001480_001649"
        Frames = @(1549, 1555, 1561, 1567, 1573, 1578)
    },
    @{
        Name = "candidate_001588_001608"
        Scan = "frames_001480_001649"
        Frames = @(1588, 1592, 1596, 1600, 1604, 1608)
    }
)

$Manifest = @()
foreach ($Group in $Groups) {
    $CurrentScan = Join-Path $ScanRoot $Group.Scan
    if (-not (Test-Path -LiteralPath $CurrentScan -PathType Container)) {
        throw "Required scan is missing: $CurrentScan"
    }
    foreach ($FrameId in $Group.Frames) {
        $FrameName = "frame_{0:000000}.png" -f $FrameId
        $OriginalSource = Join-Path $CurrentScan "original_frames\$FrameName"
        $CandidateSource = Join-Path $CurrentScan "all_candidates\$FrameName"
        if (-not (Test-Path -LiteralPath $OriginalSource -PathType Leaf)) {
            throw "Original sample is missing: $OriginalSource"
        }
        if (-not (Test-Path -LiteralPath $CandidateSource -PathType Leaf)) {
            throw "CLRNet candidate sample is missing: $CandidateSource"
        }
        $TargetName = "$($Group.Name)_$FrameName"
        $OriginalTarget = Join-Path $OriginalOut $TargetName
        $CandidateTarget = Join-Path $CandidateOut $TargetName
        Copy-Item -LiteralPath $OriginalSource -Destination $OriginalTarget
        Copy-Item -LiteralPath $CandidateSource -Destination $CandidateTarget
        $Manifest += [PSCustomObject]@{
            group = $Group.Name
            frame_id = $FrameId
            original_file = "01_original_samples\$TargetName"
            candidate_file = "02_clrnet_candidate_samples\$TargetName"
            original_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $OriginalTarget).Hash.ToLower()
            candidate_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $CandidateTarget).Hash.ToLower()
        }
    }
}
$Manifest | Export-Csv -LiteralPath (Join-Path $OutputRoot "manifest.csv") `
    -NoTypeInformation -Encoding UTF8

& conda run --no-capture-output -n $EnvName python `
    scripts\make_image_contact_sheet.py `
    --input-dir $OriginalOut `
    --output (Join-Path $OutputRoot "03_original_contact_sheet.png")
if ($LASTEXITCODE -ne 0) {
    throw "Original contact sheet failed with exit code $LASTEXITCODE."
}
& conda run --no-capture-output -n $EnvName python `
    scripts\make_image_contact_sheet.py `
    --input-dir $CandidateOut `
    --output (Join-Path $OutputRoot "04_clrnet_candidate_contact_sheet.png")
if ($LASTEXITCODE -ne 0) {
    throw "CLRNet candidate contact sheet failed with exit code $LASTEXITCODE."
}

$CopiedAudit = Join-Path $OutputRoot "05_scan_audit"
New-Item -ItemType Directory -Force $CopiedAudit | Out-Null
foreach ($ScanName in @("frames_000000_000169", "frames_001480_001649")) {
    $CurrentScan = Join-Path $ScanRoot $ScanName
    Copy-Item -LiteralPath (Join-Path $CurrentScan "scan.json") `
        -Destination (Join-Path $CopiedAudit "$ScanName`_scan.json")
    Copy-Item -LiteralPath (Join-Path $CurrentScan "lane_counts.csv") `
        -Destination (Join-Path $CopiedAudit "$ScanName`_lane_counts.csv")
}

$ZipPath = Join-Path $OutputRoot "lane_semantic_review_package.zip"
$Items = Get-ChildItem -LiteralPath $OutputRoot | Where-Object Name -ne $ZipPath
Compress-Archive -LiteralPath $Items.FullName -DestinationPath $ZipPath `
    -CompressionLevel Optimal

@{
    status = "complete"
    purpose = "manual visual gate before calling CLRNet candidates lane boundaries"
    previous_outputs_modified = $false
    source_selection_root = $SelectionRoot
    sampled_groups = @($Groups | ForEach-Object { $_.Name })
    sample_count = $Manifest.Count
    output_root = $OutputRoot
    package_zip = $ZipPath
    instruction = "Upload the ZIP; do not choose a final experiment range from CLRNet counts alone."
} | ConvertTo-Json -Depth 6 | Set-Content `
    -LiteralPath (Join-Path $OutputRoot "STATUS.json") -Encoding UTF8

Write-Host ""
Write-Host "LANE SEMANTIC REVIEW PACKAGE FINISHED"
Write-Host "Output root: $OutputRoot"
Write-Host "Samples: $($Manifest.Count) originals + candidate overlays"
Write-Host "Upload this ZIP: $ZipPath"
