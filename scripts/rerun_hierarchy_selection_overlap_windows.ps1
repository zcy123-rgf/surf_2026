[CmdletBinding()]
param(
    [string]$EnvName = "surf2026-win",
    [string]$DatasetRoot = "F:\BaiduNetdiskDownload\kitti\odometry",
    [string]$PreviousSelectionRoot = ""
)

$ErrorActionPreference = "Stop"
$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RootDir

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found. Open Anaconda PowerShell Prompt and retry."
}
. (Join-Path $PSScriptRoot "kitti00_workstation_input.ps1")
$Kitti = Resolve-Kitti00WorkstationInput -DatasetRoot $DatasetRoot

if ([string]::IsNullOrWhiteSpace($PreviousSelectionRoot)) {
    $Previous = Get-ChildItem (Join-Path $RootDir "workstation_outputs") -Directory |
        Where-Object Name -Like "hierarchy_selection_*" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $Previous) {
        throw "No hierarchy_selection_* output was found."
    }
    $PreviousSelectionRoot = $Previous.FullName
}
$PreviousSelectionRoot = (Resolve-Path -LiteralPath $PreviousSelectionRoot).Path
$ScanJsons = @(Get-ChildItem `
    (Join-Path $PreviousSelectionRoot "00_search_scans") `
    -Recurse -Filter "scan.json" | Sort-Object FullName)
if ($ScanJsons.Count -ne 3) {
    throw "Expected exactly three reusable scan.json files; found $($ScanJsons.Count)."
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputRoot = Join-Path $RootDir "workstation_outputs\hierarchy_overlap_selection_$Stamp"
$SelectionDir = Join-Path $OutputRoot "01_selected_10x15_overlap"
$SelectArgs = @(
    "scripts\select_hierarchy_150_frames.py",
    "--poses", $Kitti.Poses,
    "--image-dir", $Kitti.ImageDir,
    "--image-pattern", "{frame_id:06d}.png",
    "--block-size", "15",
    "--block-count", "10",
    "--block-stride", "10",
    "--minimum-valid-frames-per-block", "5",
    "--output-dir", $SelectionDir
)
foreach ($ScanJson in $ScanJsons) {
    $SelectArgs += @("--scan-json", $ScanJson.FullName)
}

& conda run --no-capture-output -n $EnvName python @SelectArgs
if ($LASTEXITCODE -ne 0) {
    throw "Overlapping hierarchy selection failed with exit code $LASTEXITCODE."
}
& conda run --no-capture-output -n $EnvName python `
    scripts\make_image_contact_sheet.py `
    --input-dir (Join-Path $SelectionDir "manual_annotation_package\images") `
    --output (Join-Path $SelectionDir "manual_annotation_contact_sheet.png")
if ($LASTEXITCODE -ne 0) {
    throw "Manual annotation contact sheet failed with exit code $LASTEXITCODE."
}

$SelectionPath = Join-Path $SelectionDir "selection.json"
$Selection = Get-Content -LiteralPath $SelectionPath -Raw | ConvertFrom-Json
@{
    status = "complete"
    previous_outputs_modified = $false
    reused_scans = $PreviousSelectionRoot
    selection_status = $Selection.status
    selected_start = $Selection.selected.start_frame
    selected_end = $Selection.selected.end_frame
    unique_frames = $Selection.total_unique_frames
    block_size = $Selection.block_size
    block_stride = $Selection.block_stride
    block_count = $Selection.block_count
    minimum_valid_frames_in_a_block = $Selection.selected.minimum_valid_frames_in_a_block
    total_valid_frame_uses = $Selection.selected.total_valid_frames
    selection_json = $SelectionPath
    annotation_zip = $Selection.package_zip
} | ConvertTo-Json -Depth 5 | Set-Content `
    -LiteralPath (Join-Path $OutputRoot "STATUS.json") -Encoding UTF8

Write-Host ""
Write-Host "OVERLAPPING HIERARCHY SELECTION FINISHED"
Write-Host "Output root: $OutputRoot"
Write-Host "Selection status: $($Selection.status)"
Write-Host "Selected unique frames: $($Selection.selected.start_frame)-$($Selection.selected.end_frame)"
Write-Host "Design: 10 windows x 15 frames, stride 10, adjacent overlap 5"
Write-Host "Minimum valid frames in one window: $($Selection.selected.minimum_valid_frames_in_a_block)"
Write-Host "Manual annotation ZIP: $($Selection.package_zip)"
