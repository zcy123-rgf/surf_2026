[CmdletBinding()]
param(
    [string]$EnvName = "surf2026-win",
    [string]$DatasetRoot = "F:\BaiduNetdiskDownload\kitti\odometry",
    [Parameter(Mandatory = $true)]
    [string]$BatchRoot,
    [ValidateSet("01", "02", "03", "04", "05", "06", "07", "08", "09")]
    [string[]]$HierarchySequenceIds = @("01", "06"),
    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"
$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RootDir

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found. Open Anaconda PowerShell Prompt and retry."
}
$BatchRoot = (Resolve-Path -LiteralPath $BatchRoot).Path
if (-not (Test-Path -LiteralPath (Join-Path $BatchRoot "BATCH_STATUS.json"))) {
    throw "BATCH_STATUS.json is missing from BatchRoot: $BatchRoot"
}
if ($HierarchySequenceIds.Count -ne (@($HierarchySequenceIds | Sort-Object -Unique)).Count) {
    throw "HierarchySequenceIds must not contain duplicates."
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputRoot = Join-Path $RootDir "workstation_outputs\next_stage_lane_$Stamp"
}
if (Test-Path -LiteralPath $OutputRoot) {
    if (Get-ChildItem -LiteralPath $OutputRoot -Force) {
        throw "OutputRoot must be new or empty: $OutputRoot"
    }
} else {
    New-Item -ItemType Directory -Path $OutputRoot | Out-Null
}
$OutputRoot = (Resolve-Path -LiteralPath $OutputRoot).Path

. (Join-Path $PSScriptRoot "kitti_odometry_workstation_input.ps1")

function Invoke-CondaPython {
    param([string]$Label, [string[]]$PythonArgs)
    Write-Host ""
    Write-Host "[$Label]"
    & conda run --no-capture-output -n $EnvName python @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

function Resolve-SequenceInput {
    param([string]$SequenceId)
    $Paths = Resolve-KittiOdometryWorkstationInput `
        -DatasetRoot $DatasetRoot `
        -SequenceId $SequenceId
    $null = Test-KittiOdometrySequenceInput -InputPaths $Paths
    return $Paths
}

# Question 4: export independent images before any manual labels exist.
$AnnotationSequence = "09"
$AnnotationFrames = @(51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151, 155)
$AnnotationInput = Resolve-SequenceInput -SequenceId $AnnotationSequence
$AnnotationRoot = Join-Path $OutputRoot "sequence09_manual_annotation_package"
$AnnotationImages = Join-Path $AnnotationRoot "images"
New-Item -ItemType Directory -Path $AnnotationImages -Force | Out-Null

$ManifestRows = @()
foreach ($FrameId in $AnnotationFrames) {
    $Name = "{0:D6}.png" -f $FrameId
    $Source = Join-Path $AnnotationInput.ImageDir $Name
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        throw "Manual-annotation source image is missing: $Source"
    }
    $Destination = Join-Path $AnnotationImages $Name
    Copy-Item -LiteralPath $Source -Destination $Destination
    $ManifestRows += [PSCustomObject]@{
        sequence_id = $AnnotationSequence
        frame_id = $FrameId
        filename = $Name
        sha256 = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLower()
    }
}
$ManifestRows | Export-Csv `
    -LiteralPath (Join-Path $AnnotationRoot "manifest.csv") `
    -NoTypeInformation -Encoding UTF8

Add-Type -AssemblyName System.Drawing
$FirstImage = [System.Drawing.Image]::FromFile(
    (Join-Path $AnnotationImages ("{0:D6}.png" -f $AnnotationFrames[0]))
)
try {
    $ImageSize = @([int]$FirstImage.Width, [int]$FirstImage.Height)
} finally {
    $FirstImage.Dispose()
}
$AnnotationTemplate = @{
    dataset = "KITTI Odometry Sequence 09"
    frames = $AnnotationFrames
    image_size_wh_px = $ImageSize
    annotation_type = "manual apparent left/right driving-boundary pseudo-labels"
    annotation_note = "Independent manual pseudo-ground-truth; not official KITTI ground truth. Empty arrays must be filled by visual annotation, not copied from CLRNet."
    coordinate_convention = "image pixel [x, y] from top-left"
    frames_xy = @(
        foreach ($FrameId in $AnnotationFrames) {
            @{
                frame_id = $FrameId
                left = @()
                right = @()
                occlusion_note = ""
            }
        }
    )
}
$AnnotationTemplate | ConvertTo-Json -Depth 8 | Set-Content `
    -LiteralPath (Join-Path $AnnotationRoot "annotation_template.json") `
    -Encoding UTF8

@'
Sequence 09 manual pseudo-ground-truth package

1. Mark the visually apparent left and right driving boundaries independently.
2. Use image pixel coordinates [x, y] from the top-left.
3. Do not copy CLRNet points into the annotation.
4. Record occluded or ambiguous boundaries in occlusion_note.
5. These annotations are pseudo-ground-truth, not official KITTI truth.
'@ | Set-Content `
    -LiteralPath (Join-Path $AnnotationRoot "ANNOTATION_INSTRUCTIONS.txt") `
    -Encoding UTF8

Invoke-CondaPython -Label "Sequence 09 annotation contact sheet" `
    -PythonArgs @(
        "scripts\make_image_contact_sheet.py",
        "--input-dir", $AnnotationImages,
        "--output", (Join-Path $AnnotationRoot "contact_sheet.png")
    )
$AnnotationZip = Join-Path $OutputRoot "sequence09_manual_annotation_package.zip"
Compress-Archive -Path (Join-Path $AnnotationRoot "*") `
    -DestinationPath $AnnotationZip -CompressionLevel Optimal

# Question 3 and the long-curve failure audit: ten overlapping 15-frame windows,
# sparse arc-length anchors, then a second pose-aligned fusion.
$HierarchyRows = @()
foreach ($SequenceId in $HierarchySequenceIds) {
    $SequenceRoot = Join-Path $BatchRoot "sequence_$SequenceId"
    $SelectionJson = Join-Path $SequenceRoot "01_ranked_option\selection.json"
    if (-not (Test-Path -LiteralPath $SelectionJson -PathType Leaf)) {
        $HierarchyRows += [PSCustomObject]@{
            sequence_id = $SequenceId
            status = "failed"
            output = ""
            error = "Selection JSON is missing: $SelectionJson"
        }
        continue
    }
    $SequenceInput = Resolve-SequenceInput -SequenceId $SequenceId
    $HierarchyOutput = Join-Path $OutputRoot "hierarchy_sequence_$SequenceId"
    try {
        Invoke-CondaPython -Label "Sequence $SequenceId sparse hierarchical refusion" `
            -PythonArgs @(
                "scripts\run_weekly_lane_hierarchy.py",
                "--selection-json", $SelectionJson,
                "--poses", $SequenceInput.Poses,
                "--calib", $SequenceInput.Calib,
                "--output-dir", $HierarchyOutput,
                "--maximum-cv-folds", "20",
                "--curve-samples", "600",
                "--feature-count-grid", "4,6,8,12"
            )
        $HierarchyStatus = Get-Content `
            -LiteralPath (Join-Path $HierarchyOutput "STATUS.json") `
            -Raw | ConvertFrom-Json
        $HierarchyRows += [PSCustomObject]@{
            sequence_id = $SequenceId
            status = "complete"
            windows_completed = $HierarchyStatus.windows_completed
            valid_unique_frames = $HierarchyStatus.valid_unique_frames
            selected_feature_points_per_window_per_side = `
                $HierarchyStatus.selected_feature_points_per_window_per_side
            output = $HierarchyOutput
            error = ""
        }
    } catch {
        $HierarchyRows += [PSCustomObject]@{
            sequence_id = $SequenceId
            status = "failed"
            windows_completed = $null
            valid_unique_frames = $null
            selected_feature_points_per_window_per_side = $null
            output = $HierarchyOutput
            error = $_.Exception.Message
        }
        Write-Warning "Sequence $SequenceId hierarchy failed: $($_.Exception.Message)"
    }
}

$HierarchyRows | Export-Csv `
    -LiteralPath (Join-Path $OutputRoot "HIERARCHY_SUMMARY.csv") `
    -NoTypeInformation -Encoding UTF8
$FailedCount = @($HierarchyRows | Where-Object status -eq "failed").Count
$FinalStatus = if ($FailedCount -eq 0) { "complete" } else { "complete_with_failures" }
$Status = @{
    status = $FinalStatus
    previous_outputs_modified = $false
    source_batch = $BatchRoot
    hierarchy_method = "ten overlapping 15-frame local fits; sparse arc-length anchors; second pose-aligned fusion"
    hierarchy_results = $HierarchyRows
    manual_annotation = @{
        sequence_id = $AnnotationSequence
        frames = $AnnotationFrames
        status = "pending_manual_annotation"
        package_zip = $AnnotationZip
        warning = "Manual labels are pseudo-ground-truth, not official KITTI ground truth."
    }
}
$Status | ConvertTo-Json -Depth 8 | Set-Content `
    -LiteralPath (Join-Path $OutputRoot "STATUS.json") -Encoding UTF8

$ResultBundle = Join-Path $OutputRoot "NEXT_STAGE_RESULT_BUNDLE.zip"
$BundleItems = @(
    (Join-Path $OutputRoot "STATUS.json"),
    (Join-Path $OutputRoot "HIERARCHY_SUMMARY.csv")
)
foreach ($SequenceId in $HierarchySequenceIds) {
    $Path = Join-Path $OutputRoot "hierarchy_sequence_$SequenceId"
    if (Test-Path -LiteralPath $Path) {
        $BundleItems += $Path
    }
}
Compress-Archive -LiteralPath $BundleItems `
    -DestinationPath $ResultBundle -CompressionLevel Optimal

Write-Host ""
Write-Host "NEXT STAGE FINISHED"
Write-Host "Status: $FinalStatus"
Write-Host "Output root: $OutputRoot"
Write-Host "Sparse-refusion bundle: $ResultBundle"
Write-Host "Manual-annotation package: $AnnotationZip"
Write-Host "Upload both ZIP files for final analysis and manual annotation."

