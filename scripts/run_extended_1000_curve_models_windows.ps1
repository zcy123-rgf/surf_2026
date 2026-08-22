[CmdletBinding()]
param(
    [string]$EnvName = "surf2026-win",
    [string]$DatasetRoot = "F:\BaiduNetdiskDownload\kitti\odometry",
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [int]$StartFrame = 0,
    [int]$EndFrame = 1000,
    [int]$TopCandidateCount = 100,
    [int]$SelectedRank = 1,
    [int]$MinimumValidFramesPerBlock = 5,
    [ValidateSet("both", "polynomial", "bspline")]
    [string]$Model = "both",
    [string]$ExistingScanJson = ""
)

$ErrorActionPreference = "Stop"
$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RootDir

if ($StartFrame -lt 0 -or $EndFrame -lt $StartFrame) {
    throw "Frame range must satisfy 0 <= StartFrame <= EndFrame."
}
if ($TopCandidateCount -lt 1 -or $SelectedRank -lt 1) {
    throw "TopCandidateCount and SelectedRank must be positive."
}
if ($SelectedRank -gt $TopCandidateCount) {
    throw "SelectedRank cannot exceed TopCandidateCount."
}
if ($MinimumValidFramesPerBlock -lt 1 -or $MinimumValidFramesPerBlock -gt 15) {
    throw "MinimumValidFramesPerBlock must be between 1 and 15."
}
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found. Open Anaconda PowerShell Prompt and retry."
}

. (Join-Path $PSScriptRoot "kitti00_workstation_input.ps1")
$Kitti = Resolve-Kitti00WorkstationInput -DatasetRoot $DatasetRoot
$null = Test-Kitti00FirstFiveInput -InputPaths $Kitti -RequireCompleteSequence

$RequiredScripts = @(
    "check_windows_env.py",
    "scan_clrnet_lane_counts.py",
    "select_hierarchy_150_frames.py",
    "extended_curve_model_common.py",
    "fit_extended_polynomial.py",
    "fit_extended_bspline.py",
    "run_weekly_lane_hierarchy.py",
    "fit_first5_two_curves.py"
)
foreach ($Name in $RequiredScripts) {
    $Path = Join-Path $PSScriptRoot $Name
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required script is missing: $Path"
    }
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputRoot = Join-Path $RootDir (
    "workstation_outputs\extended_curve_{0:000000}_{1:000000}_{2}" -f `
        $StartFrame, $EndFrame, $Stamp
)
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

Invoke-CondaPython -Label "1/5 Workstation CUDA and CLRNet preflight" `
    -PythonArgs @(
        "scripts\check_windows_env.py",
        "--device", $Device,
        "--run-clrnet"
    )

if ([string]::IsNullOrWhiteSpace($ExistingScanJson)) {
    $ScanDir = Join-Path $OutputRoot "00_scan_frames"
    Invoke-CondaPython -Label "2/5 CLRNet scan $StartFrame-$EndFrame (inclusive)" `
        -PythonArgs @(
            "scripts\scan_clrnet_lane_counts.py",
            "--image-dir", $Kitti.ImageDir,
            "--image-pattern", "{frame_id:06d}.png",
            "--frame-start", [string]$StartFrame,
            "--frame-end", [string]$EndFrame,
            "--calib", $Kitti.Calib,
            "--poses", $Kitti.Poses,
            "--segment-size", "5",
            "--minimum-candidates", "2",
            "--minimum-bev-points-per-side", "4",
            "--local-z-range=3,50",
            "--fusion-x-range=-20,20",
            "--fusion-z-range=-20,50",
            "--device", $Device,
            "--skip-recommendation",
            "--output-dir", $ScanDir
        )
    $ScanJson = Join-Path $ScanDir "scan.json"
} else {
    $ScanJson = (Resolve-Path -LiteralPath $ExistingScanJson).Path
    $ScanDocument = Get-Content -LiteralPath $ScanJson -Raw | ConvertFrom-Json
    if ($ScanDocument.status -ne "complete") {
        throw "Existing scan is not complete: $ScanJson"
    }
    $FirstScanned = [int]$ScanDocument.requested_frame_ids[0]
    $LastScanned = [int]$ScanDocument.requested_frame_ids[-1]
    if ($FirstScanned -ne $StartFrame -or $LastScanned -ne $EndFrame) {
        throw (
            "Existing scan range $FirstScanned-$LastScanned must exactly match " +
            "$StartFrame-$EndFrame so candidate ranking cannot silently use extra frames."
        )
    }
    Write-Host "Reusing scan: $ScanJson"
}

$SelectionDir = Join-Path $OutputRoot "01_ranked_option"
Invoke-CondaPython -Label "3/5 Rank up to $TopCandidateCount road-section options" `
    -PythonArgs @(
        "scripts\select_hierarchy_150_frames.py",
        "--scan-json", $ScanJson,
        "--poses", $Kitti.Poses,
        "--image-dir", $Kitti.ImageDir,
        "--image-pattern", "{frame_id:06d}.png",
        "--block-size", "15",
        "--block-count", "10",
        "--block-stride", "10",
        "--minimum-valid-frames-per-block", [string]$MinimumValidFramesPerBlock,
        "--selected-rank", [string]$SelectedRank,
        "--top-candidate-count", [string]$TopCandidateCount,
        "--output-dir", $SelectionDir
    )

$SelectionJson = Join-Path $SelectionDir "selection.json"
$Selection = Get-Content -LiteralPath $SelectionJson -Raw | ConvertFrom-Json
if ($Selection.status -ne "selected") {
    @{
        status = "stopped_at_coverage_gate"
        previous_outputs_modified = $false
        search_frames = @($StartFrame, $EndFrame)
        selected_rank = $SelectedRank
        selection_status = $Selection.status
        candidate_options_csv = $Selection.candidate_options_csv
        selection_json = $SelectionJson
        scan_json = $ScanJson
    } | ConvertTo-Json -Depth 6 | Set-Content `
        -LiteralPath (Join-Path $OutputRoot "RUN_STATUS.json") -Encoding UTF8
    throw (
        "Selected option did not meet the coverage gate. Review " +
        "$($Selection.candidate_options_csv); no curve result was claimed."
    )
}

$SharedModelArgs = @(
    "--selection-json", $SelectionJson,
    "--poses", $Kitti.Poses,
    "--calib", $Kitti.Calib,
    "--maximum-cv-folds", "20",
    "--curve-samples", "600"
)
$PolynomialOutput = $null
$BsplineOutput = $null

if ($Model -eq "both" -or $Model -eq "polynomial") {
    $PolynomialOutput = Join-Path $OutputRoot "02_polynomial_only"
    $PolynomialArgs = @(
        "scripts\fit_extended_polynomial.py",
        "--output-dir", $PolynomialOutput
    ) + $SharedModelArgs
    Invoke-CondaPython -Label "4/5 Separate robust polynomial experiment" `
        -PythonArgs $PolynomialArgs
}

if ($Model -eq "both" -or $Model -eq "bspline") {
    $BsplineOutput = Join-Path $OutputRoot "03_bspline_only"
    $BsplineArgs = @(
        "scripts\fit_extended_bspline.py",
        "--output-dir", $BsplineOutput
    ) + $SharedModelArgs
    Invoke-CondaPython -Label "5/5 Separate robust cubic B-spline experiment" `
        -PythonArgs $BsplineArgs
}

@{
    status = "complete"
    previous_outputs_modified = $false
    search_frame_start = $StartFrame
    search_frame_end_inclusive = $EndFrame
    searched_frame_count = $EndFrame - $StartFrame + 1
    selected_rank = $SelectedRank
    selected_frames = @(
        [int]$Selection.selected.start_frame,
        [int]$Selection.selected.end_frame
    )
    available_candidate_count = [int]$Selection.available_candidate_count
    exported_candidate_count = [int]$Selection.exported_candidate_count
    candidate_options_csv = [string]$Selection.candidate_options_csv
    scan_json = $ScanJson
    selection_json = $SelectionJson
    polynomial_output = $PolynomialOutput
    bspline_output = $BsplineOutput
    coordinate_changes = "none"
} | ConvertTo-Json -Depth 7 | Set-Content `
    -LiteralPath (Join-Path $OutputRoot "RUN_STATUS.json") -Encoding UTF8

Write-Host ""
Write-Host "EXTENDED CURVE EXPERIMENT FINISHED"
Write-Host "Output root: $OutputRoot"
Write-Host "Scanned frames: $StartFrame-$EndFrame (inclusive)"
Write-Host "Candidate options: $($Selection.candidate_options_csv)"
Write-Host "Selected rank: $SelectedRank"
Write-Host "Selected frames: $($Selection.selected.start_frame)-$($Selection.selected.end_frame)"
if ($PolynomialOutput) { Write-Host "Polynomial: $PolynomialOutput" }
if ($BsplineOutput) { Write-Host "B-spline: $BsplineOutput" }
Write-Host "Run audit: $(Join-Path $OutputRoot 'RUN_STATUS.json')"
