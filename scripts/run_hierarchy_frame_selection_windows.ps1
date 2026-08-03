[CmdletBinding()]
param(
    [string]$EnvName = "surf2026-win",
    [string]$DatasetRoot = "F:\BaiduNetdiskDownload\kitti\odometry",
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RootDir

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found. Open Anaconda PowerShell Prompt and retry."
}
. (Join-Path $PSScriptRoot "kitti00_workstation_input.ps1")
$Kitti = Resolve-Kitti00WorkstationInput -DatasetRoot $DatasetRoot
$Verification = Test-Kitti00FirstFiveInput -InputPaths $Kitti -RequireCompleteSequence

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputRoot = Join-Path $RootDir "workstation_outputs\hierarchy_selection_$Stamp"
New-Item -ItemType Directory -Path $OutputRoot | Out-Null

function Invoke-CondaPython {
    param([string]$Label, [string[]]$PythonArgs)
    Write-Host ""
    Write-Host "[$Label]"
    & conda run --no-capture-output -n $EnvName python @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

# Three 170-frame search regions: the stable sequence start, the region around
# the successful 1549/1588 experiments, and the longest late class-60 cue.
$SearchRanges = @(
    @{ Start = 0; End = 169 },
    @{ Start = 1480; End = 1649 },
    @{ Start = 4371; End = 4540 }
)
$ScanJsons = @()
foreach ($Range in $SearchRanges) {
    $FrameArray = $Range.Start..$Range.End
    $FrameIds = $FrameArray -join ","
    $Name = "frames_{0:000000}_{1:000000}" -f $Range.Start, $Range.End
    $ScanDir = Join-Path $OutputRoot "00_search_scans\$Name"
    Invoke-CondaPython -Label "Live CLRNet/IPM feasibility scan $($Range.Start)-$($Range.End)" `
        -PythonArgs @(
            "scripts\scan_clrnet_lane_counts.py",
            "--image-dir", $Kitti.ImageDir,
            "--image-pattern", "{frame_id:06d}.png",
            "--frame-ids", $FrameIds,
            "--calib", $Kitti.Calib,
            "--poses", $Kitti.Poses,
            "--segment-size", "15",
            "--minimum-candidates", "2",
            "--minimum-bev-points-per-side", "4",
            "--local-z-range=3,50",
            "--fusion-x-range=-20,20",
            "--fusion-z-range=-20,50",
            "--device", $Device,
            "--output-dir", $ScanDir
        )
    $ScanJsons += (Join-Path $ScanDir "scan.json")
}

$SelectionDir = Join-Path $OutputRoot "01_selected_150_frames"
$SelectArgs = @(
    "scripts\select_hierarchy_150_frames.py",
    "--poses", $Kitti.Poses,
    "--image-dir", $Kitti.ImageDir,
    "--image-pattern", "{frame_id:06d}.png",
    "--block-size", "15",
    "--block-count", "10",
    "--minimum-valid-frames-per-block", "5",
    "--output-dir", $SelectionDir
)
foreach ($ScanJson in $ScanJsons) {
    $SelectArgs += @("--scan-json", $ScanJson)
}
Invoke-CondaPython -Label "Select one continuous 150-frame hierarchy experiment" `
    -PythonArgs $SelectArgs

Invoke-CondaPython -Label "Build 30-frame manual-annotation contact sheet" `
    -PythonArgs @(
        "scripts\make_image_contact_sheet.py",
        "--input-dir", (Join-Path $SelectionDir "manual_annotation_package\images"),
        "--output", (Join-Path $SelectionDir "manual_annotation_contact_sheet.png")
    )

$SelectionPath = Join-Path $SelectionDir "selection.json"
$Selection = Get-Content -LiteralPath $SelectionPath -Raw | ConvertFrom-Json
@{
    status = "complete"
    dataset = "KITTI Odometry Sequence 00"
    previous_outputs_modified = $false
    selection_status = $Selection.status
    selected_start = $Selection.selected.start_frame
    selected_end = $Selection.selected.end_frame
    minimum_valid_frames_in_a_block = $Selection.selected.minimum_valid_frames_in_a_block
    total_valid_frames = $Selection.selected.total_valid_frames
    selection_json = $SelectionPath
    annotation_zip = $Selection.package_zip
} | ConvertTo-Json -Depth 5 | Set-Content `
    -LiteralPath (Join-Path $OutputRoot "STATUS.json") -Encoding UTF8

Write-Host ""
Write-Host "HIERARCHY FRAME SELECTION FINISHED"
Write-Host "Output root: $OutputRoot"
Write-Host "Selection status: $($Selection.status)"
Write-Host "Selected frames: $($Selection.selected.start_frame)-$($Selection.selected.end_frame)"
Write-Host "Minimum valid frames in one 15-frame block: $($Selection.selected.minimum_valid_frames_in_a_block)"
Write-Host "Manual annotation ZIP: $($Selection.package_zip)"
Write-Host "Upload that ZIP for manual pseudo-ground-truth annotation."
