[CmdletBinding()]
param(
    [string]$EnvName = "surf2026-win",
    [string]$DatasetRoot = "F:\BaiduNetdiskDownload\kitti\odometry",
    [string]$SemanticKittiRoot = "F:\BaiduNetdiskDownload\SemanticKITTI",
    [string]$SelectedWindowsCsv = "",
    [string]$ClrnetRoot = "F:\2026_surf\CLRNet",
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [double]$TemporalMaximumMatchCostM = 1.50,
    [int]$TemporalMaximumGapFrames = 3,
    [int]$MinimumValidFrames = 10,
    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"
$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RootDir
if ([string]::IsNullOrWhiteSpace($SelectedWindowsCsv)) {
    $SelectedWindowsCsv = Join-Path $RootDir "configs\selected_windows_qly_20260814.csv"
}

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found. Open Anaconda PowerShell Prompt and retry."
}
$RequiredFiles = @(
    "scripts\kitti_odometry_workstation_input.ps1",
    "scripts\scan_clrnet_lane_counts.py",
    "scripts\analyze_lane_identity_modes.py",
    "scripts\create_fixed_window_selection.py",
    "scripts\fit_extended_polynomial.py",
    "scripts\fit_extended_bspline.py",
    "scripts\evaluate_semantickitti_curve_reference.py"
)
foreach ($Relative in $RequiredFiles) {
    if (-not (Test-Path -LiteralPath (Join-Path $RootDir $Relative) -PathType Leaf)) {
        throw "Required file is missing: $Relative"
    }
}
foreach ($Path in @($SelectedWindowsCsv, $ClrnetRoot, $DatasetRoot, $SemanticKittiRoot)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required input does not exist: $Path"
    }
}

. (Join-Path $PSScriptRoot "kitti_odometry_workstation_input.ps1")

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputRoot = Join-Path $RootDir "workstation_outputs\zcy_cxy_selected_windows_$Stamp"
}
if (Test-Path -LiteralPath $OutputRoot) {
    if (Get-ChildItem -LiteralPath $OutputRoot -Force) {
        throw "OutputRoot must be new or empty: $OutputRoot"
    }
} else {
    New-Item -ItemType Directory -Path $OutputRoot | Out-Null
}
$OutputRoot = (Resolve-Path -LiteralPath $OutputRoot).Path
$ClrnetRoot = (Resolve-Path -LiteralPath $ClrnetRoot).Path
$Windows = @(Import-Csv -LiteralPath $SelectedWindowsCsv | Sort-Object { [int]$_.priority })
if ($Windows.Count -eq 0) {
    throw "SelectedWindowsCsv contains no rows: $SelectedWindowsCsv"
}

function Invoke-CondaPython {
    param([string]$Label, [string[]]$PythonArgs)
    Write-Host ""
    Write-Host "[$Label]"
    & conda run --no-capture-output -n $EnvName python @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

function Invoke-SelectedWindow {
    param([PSCustomObject]$Window)

    $SequenceId = ([string]$Window.sequence).PadLeft(2, "0")
    $StartFrame = [int]$Window.start_frame
    $EndFrame = [int]$Window.end_frame
    $Purpose = [string]$Window.purpose
    $WindowDir = Join-Path $OutputRoot (
        "sequence_{0}_{1:000000}_{2:000000}_{3}" -f $SequenceId, $StartFrame, $EndFrame, $Purpose
    )
    New-Item -ItemType Directory -Path $WindowDir | Out-Null

    $Resolved = Resolve-KittiOdometryWorkstationInput `
        -DatasetRoot $DatasetRoot -SequenceId $SequenceId
    $Kitti = Test-KittiOdometrySequenceInput -InputPaths $Resolved `
        -MinimumFrameCount ($EndFrame + 1)
    $VelodyneDir = Join-Path $DatasetRoot (
        "data_odometry_velodyne\dataset\sequences\$SequenceId\velodyne"
    )
    $LabelDir = Join-Path $SemanticKittiRoot "dataset\sequences\$SequenceId\labels"
    foreach ($Path in @($VelodyneDir, $LabelDir)) {
        if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
            throw "Sequence $SequenceId reference input is missing: $Path"
        }
    }
    foreach ($Frame in $StartFrame..$EndFrame) {
        foreach ($Path in @(
            (Join-Path $VelodyneDir ("{0:000000}.bin" -f $Frame)),
            (Join-Path $LabelDir ("{0:000000}.label" -f $Frame))
        )) {
            if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
                throw "Sequence $SequenceId frame $Frame is missing: $Path"
            }
        }
    }

    $EgoDir = Join-Path $WindowDir "01_zcy_ego_adjacent"
    Invoke-CondaPython -Label "$SequenceId $StartFrame-${EndFrame}: ego_adjacent scan" `
        -PythonArgs @(
            "scripts\scan_clrnet_lane_counts.py",
            "--dataset-name", "KITTI Odometry Sequence $SequenceId",
            "--image-dir", $Kitti.ImageDir,
            "--frame-start", [string]$StartFrame,
            "--frame-end", [string]$EndFrame,
            "--calib", $Kitti.Calib,
            "--poses", $Kitti.Poses,
            "--candidate-selection-mode", "ego_adjacent",
            "--minimum-candidates", "2",
            "--minimum-bev-points-per-side", "4",
            "--local-z-range=3,50",
            "--fusion-x-range=-20,20",
            "--fusion-z-range=-20,50",
            "--clrnet-root", $ClrnetRoot,
            "--device", $Device,
            "--skip-recommendation",
            "--output-dir", $EgoDir
        )

    $TemporalDir = Join-Path $WindowDir "02_zcy_temporal_ego"
    Invoke-CondaPython -Label "$SequenceId $StartFrame-${EndFrame}: temporal_ego scan" `
        -PythonArgs @(
            "scripts\scan_clrnet_lane_counts.py",
            "--dataset-name", "KITTI Odometry Sequence $SequenceId",
            "--image-dir", $Kitti.ImageDir,
            "--frame-start", [string]$StartFrame,
            "--frame-end", [string]$EndFrame,
            "--calib", $Kitti.Calib,
            "--poses", $Kitti.Poses,
            "--candidate-selection-mode", "temporal_ego",
            "--temporal-maximum-match-cost-m", [string]$TemporalMaximumMatchCostM,
            "--temporal-maximum-gap-frames", [string]$TemporalMaximumGapFrames,
            "--minimum-candidates", "2",
            "--minimum-bev-points-per-side", "4",
            "--local-z-range=3,50",
            "--fusion-x-range=-20,20",
            "--fusion-z-range=-20,50",
            "--clrnet-root", $ClrnetRoot,
            "--device", $Device,
            "--skip-recommendation",
            "--output-dir", $TemporalDir
        )

    $IdentityDir = Join-Path $WindowDir "03_zcy_identity_comparison"
    Invoke-CondaPython -Label "$SequenceId $StartFrame-${EndFrame}: identity comparison" `
        -PythonArgs @(
            "scripts\analyze_lane_identity_modes.py",
            "--ego-json", (Join-Path $EgoDir "selected_lane_points.json"),
            "--temporal-json", (Join-Path $TemporalDir "selected_lane_points.json"),
            "--poses", $Kitti.Poses,
            "--calib", $Kitti.Calib,
            "--continuity-threshold-m", [string]$TemporalMaximumMatchCostM,
            "--output-dir", $IdentityDir
        )

    $SelectionJson = Join-Path $WindowDir "04_fixed_temporal_selection\selection.json"
    Invoke-CondaPython -Label "$SequenceId $StartFrame-${EndFrame}: fixed selection" `
        -PythonArgs @(
            "scripts\create_fixed_window_selection.py",
            "--scan-json", (Join-Path $TemporalDir "scan.json"),
            "--start-frame", [string]$StartFrame,
            "--end-frame", [string]$EndFrame,
            "--dataset-name", "KITTI Odometry Sequence $SequenceId",
            "--minimum-valid-frames", [string]$MinimumValidFrames,
            "--output", $SelectionJson
        )

    $PolynomialDir = Join-Path $WindowDir "05_polynomial"
    Invoke-CondaPython -Label "$SequenceId $StartFrame-${EndFrame}: polynomial fit" `
        -PythonArgs @(
            "scripts\fit_extended_polynomial.py",
            "--selection-json", $SelectionJson,
            "--poses", $Kitti.Poses,
            "--calib", $Kitti.Calib,
            "--output-dir", $PolynomialDir
        )

    $BsplineDir = Join-Path $WindowDir "06_bspline"
    Invoke-CondaPython -Label "$SequenceId $StartFrame-${EndFrame}: B-spline fit" `
        -PythonArgs @(
            "scripts\fit_extended_bspline.py",
            "--selection-json", $SelectionJson,
            "--poses", $Kitti.Poses,
            "--calib", $Kitti.Calib,
            "--output-dir", $BsplineDir
        )

    $SemanticDir = Join-Path $WindowDir "07_cxy_semantic_evaluation"
    $PolynomialMethod = "polynomial,{0},{1}" -f `
        (Join-Path $PolynomialDir "02_curves\left_polynomial_curve.csv"), `
        (Join-Path $PolynomialDir "02_curves\right_polynomial_curve.csv")
    $BsplineMethod = "bspline,{0},{1}" -f `
        (Join-Path $BsplineDir "02_curves\left_curve.csv"), `
        (Join-Path $BsplineDir "02_curves\right_curve.csv")
    Invoke-CondaPython -Label "$SequenceId $StartFrame-${EndFrame}: SemanticKITTI evaluation" `
        -PythonArgs @(
            "scripts\evaluate_semantickitti_curve_reference.py",
            "--selected-lane-points", (Join-Path $TemporalDir "selected_lane_points.json"),
            "--velodyne-dir", $VelodyneDir,
            "--label-dir", $LabelDir,
            "--poses", $Kitti.Poses,
            "--calib", $Kitti.Calib,
            "--frame-start", [string]$StartFrame,
            "--frame-end", [string]$EndFrame,
            "--reference-id", [string]$EndFrame,
            "--method", $PolynomialMethod,
            "--method", $BsplineMethod,
            "--method-result", ("polynomial,{0}" -f (Join-Path $PolynomialDir "RESULT.json")),
            "--method-result", ("bspline,{0}" -f (Join-Path $BsplineDir "RESULT.json")),
            "--output-dir", $SemanticDir
        )

    $Identity = Get-Content -LiteralPath (Join-Path $IdentityDir "RESULT.json") -Raw | ConvertFrom-Json
    $Semantic = Get-Content -LiteralPath (Join-Path $SemanticDir "RESULT.json") -Raw | ConvertFrom-Json
    return [PSCustomObject]@{
        priority = [int]$Window.priority
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
        output = $WindowDir
    }
}

$Rows = @()
foreach ($Window in $Windows) {
    try {
        $Rows += Invoke-SelectedWindow -Window $Window
    } catch {
        $Rows += [PSCustomObject]@{
            priority = [int]$Window.priority
            purpose = [string]$Window.purpose
            sequence = ([string]$Window.sequence).PadLeft(2, "0")
            start_frame = [int]$Window.start_frame
            end_frame = [int]$Window.end_frame
            status = "failed"
            error = $_.Exception.Message
        }
        Write-Warning $_.Exception.Message
    }
}

$Rows | Export-Csv -LiteralPath (Join-Path $OutputRoot "BATCH_SUMMARY.csv") `
    -NoTypeInformation -Encoding UTF8
$Complete = @($Rows | Where-Object status -eq "complete").Count
$Failed = @($Rows | Where-Object status -eq "failed").Count
$Status = [ordered]@{
    status = if ($Failed -eq 0) { "complete" } elseif ($Complete -gt 0) { "partial" } else { "failed" }
    selected_windows_csv = (Resolve-Path -LiteralPath $SelectedWindowsCsv).Path
    completed_windows = $Complete
    failed_windows = $Failed
    results = $Rows
    previous_outputs_modified = $false
}
$Status | ConvertTo-Json -Depth 8 | Set-Content `
    -LiteralPath (Join-Path $OutputRoot "STATUS.json") -Encoding UTF8

$BundleDir = Join-Path $OutputRoot "MEETING_BUNDLE"
New-Item -ItemType Directory -Path $BundleDir | Out-Null
Copy-Item -LiteralPath (Join-Path $OutputRoot "BATCH_SUMMARY.csv") -Destination $BundleDir
Copy-Item -LiteralPath (Join-Path $OutputRoot "STATUS.json") -Destination $BundleDir
foreach ($Row in $Rows | Where-Object status -eq "complete") {
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
$BundleZip = "$OutputRoot\zcy_cxy_meeting_bundle.zip"
Compress-Archive -LiteralPath $BundleDir -DestinationPath $BundleZip

Write-Host ""
Write-Host "ZCY + CXY AUTOMATED RUN FINISHED"
Write-Host "Output root: $OutputRoot"
Write-Host "Completed windows: $Complete; failed windows: $Failed"
Write-Host "Upload this small result bundle: $BundleZip"
if ($Complete -eq 0) {
    throw "Both selected windows failed. Review STATUS.json."
}
