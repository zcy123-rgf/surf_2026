[CmdletBinding()]
param(
    [string]$EnvName = "surf2026-win",
    [string]$DatasetRoot = "F:\BaiduNetdiskDownload\kitti\odometry",
    [string]$SelectedWindowsCsv = "",
    [string]$ClrnetRoot = "F:\2026_surf\CLRNet",
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [ValidateSet("all", "primary", "backup")]
    [string]$Purpose = "primary",
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
    "scripts\fit_coupled_bspline.py",
    "scripts\summarize_curved_improvement.py"
)
foreach ($Relative in $RequiredFiles) {
    if (-not (Test-Path -LiteralPath (Join-Path $RootDir $Relative) -PathType Leaf)) {
        throw "Required file is missing: $Relative"
    }
}
foreach ($Path in @($SelectedWindowsCsv, $ClrnetRoot, $DatasetRoot)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required input does not exist: $Path"
    }
}

. (Join-Path $PSScriptRoot "kitti_odometry_workstation_input.ps1")

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputRoot = Join-Path $RootDir "workstation_outputs\curved_improvement_$Stamp"
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
if ($Purpose -ne "all") {
    $Windows = @($Windows | Where-Object { [string]$_.purpose -eq $Purpose })
}
if ($Windows.Count -eq 0) {
    throw "No selected windows match Purpose=$Purpose."
}

function Invoke-CondaPython {
    param([string]$Label, [string[]]$PythonArgs)
    Write-Host ""
    Write-Host "[$Label]"
    & conda run --no-capture-output -n $EnvName python @PythonArgs | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

function Invoke-Scan {
    param(
        [string]$SequenceId,
        [int]$StartFrame,
        [int]$EndFrame,
        [object]$Kitti,
        [string]$Mode,
        [string]$Destination
    )
    $Arguments = @(
        "scripts\scan_clrnet_lane_counts.py",
        "--dataset-name", "KITTI Odometry Sequence $SequenceId",
        "--image-dir", $Kitti.ImageDir,
        "--frame-start", [string]$StartFrame,
        "--frame-end", [string]$EndFrame,
        "--calib", $Kitti.Calib,
        "--poses", $Kitti.Poses,
        "--candidate-selection-mode", $Mode,
        "--minimum-candidates", "2",
        "--minimum-bev-points-per-side", "4",
        "--local-z-range=3,50",
        "--fusion-x-range=-20,20",
        "--fusion-z-range=-20,50",
        "--clrnet-root", $ClrnetRoot,
        "--device", $Device,
        "--skip-recommendation",
        "--output-dir", $Destination
    )
    if ($Mode -in @("temporal_ego", "temporal_joint")) {
        $Arguments += @(
            "--temporal-maximum-match-cost-m", [string]$TemporalMaximumMatchCostM,
            "--temporal-maximum-gap-frames", [string]$TemporalMaximumGapFrames
        )
    }
    Invoke-CondaPython -Label "$SequenceId $StartFrame-${EndFrame}: $Mode scan" `
        -PythonArgs $Arguments
}

function Invoke-Window {
    param([PSCustomObject]$Window)
    $SequenceId = ([string]$Window.sequence).PadLeft(2, "0")
    $StartFrame = [int]$Window.start_frame
    $EndFrame = [int]$Window.end_frame
    $WindowPurpose = [string]$Window.purpose
    $WindowDir = Join-Path $OutputRoot (
        "sequence_{0}_{1:000000}_{2:000000}_{3}" -f `
            $SequenceId, $StartFrame, $EndFrame, $WindowPurpose
    )
    New-Item -ItemType Directory -Path $WindowDir | Out-Null
    $Resolved = Resolve-KittiOdometryWorkstationInput `
        -DatasetRoot $DatasetRoot -SequenceId $SequenceId
    $Kitti = Test-KittiOdometrySequenceInput -InputPaths $Resolved `
        -MinimumFrameCount ($EndFrame + 1)

    $EgoDir = Join-Path $WindowDir "01_ego_adjacent"
    $OldDir = Join-Path $WindowDir "02_temporal_ego"
    $JointDir = Join-Path $WindowDir "03_temporal_joint"
    Invoke-Scan $SequenceId $StartFrame $EndFrame $Kitti "ego_adjacent" $EgoDir
    Invoke-Scan $SequenceId $StartFrame $EndFrame $Kitti "temporal_ego" $OldDir
    Invoke-Scan $SequenceId $StartFrame $EndFrame $Kitti "temporal_joint" $JointDir

    $OldIdentityDir = Join-Path $WindowDir "04_identity_old"
    Invoke-CondaPython -Label "${SequenceId}: old identity comparison" -PythonArgs @(
        "scripts\analyze_lane_identity_modes.py",
        "--ego-json", (Join-Path $EgoDir "selected_lane_points.json"),
        "--temporal-json", (Join-Path $OldDir "selected_lane_points.json"),
        "--comparison-name", "temporal_ego",
        "--poses", $Kitti.Poses,
        "--calib", $Kitti.Calib,
        "--continuity-threshold-m", [string]$TemporalMaximumMatchCostM,
        "--output-dir", $OldIdentityDir
    )
    $JointIdentityDir = Join-Path $WindowDir "05_identity_joint"
    Invoke-CondaPython -Label "${SequenceId}: joint identity comparison" -PythonArgs @(
        "scripts\analyze_lane_identity_modes.py",
        "--ego-json", (Join-Path $EgoDir "selected_lane_points.json"),
        "--temporal-json", (Join-Path $JointDir "selected_lane_points.json"),
        "--comparison-name", "temporal_joint",
        "--poses", $Kitti.Poses,
        "--calib", $Kitti.Calib,
        "--continuity-threshold-m", [string]$TemporalMaximumMatchCostM,
        "--output-dir", $JointIdentityDir
    )

    $SelectionJson = Join-Path $WindowDir "06_fixed_joint_selection\selection.json"
    Invoke-CondaPython -Label "${SequenceId}: fixed joint selection" -PythonArgs @(
        "scripts\create_fixed_window_selection.py",
        "--scan-json", (Join-Path $JointDir "scan.json"),
        "--start-frame", [string]$StartFrame,
        "--end-frame", [string]$EndFrame,
        "--dataset-name", "KITTI Odometry Sequence $SequenceId",
        "--minimum-valid-frames", [string]$MinimumValidFrames,
        "--output", $SelectionJson
    )

    $PolynomialDir = Join-Path $WindowDir "07_polynomial"
    Invoke-CondaPython -Label "${SequenceId}: polynomial" -PythonArgs @(
        "scripts\fit_extended_polynomial.py",
        "--selection-json", $SelectionJson,
        "--poses", $Kitti.Poses,
        "--calib", $Kitti.Calib,
        "--output-dir", $PolynomialDir
    )
    $BsplineDir = Join-Path $WindowDir "08_independent_bspline"
    Invoke-CondaPython -Label "${SequenceId}: independent B-spline" -PythonArgs @(
        "scripts\fit_extended_bspline.py",
        "--selection-json", $SelectionJson,
        "--poses", $Kitti.Poses,
        "--calib", $Kitti.Calib,
        "--output-dir", $BsplineDir
    )
    $CoupledDir = Join-Path $WindowDir "09_coupled_bspline"
    Invoke-CondaPython -Label "${SequenceId}: coupled B-spline" -PythonArgs @(
        "scripts\fit_coupled_bspline.py",
        "--selection-json", $SelectionJson,
        "--poses", $Kitti.Poses,
        "--calib", $Kitti.Calib,
        "--output-dir", $CoupledDir
    )

    $SummaryDir = Join-Path $WindowDir "10_improvement_summary"
    Invoke-CondaPython -Label "${SequenceId}: final improvement summary" -PythonArgs @(
        "scripts\summarize_curved_improvement.py",
        "--old-identity-result", (Join-Path $OldIdentityDir "RESULT.json"),
        "--joint-identity-result", (Join-Path $JointIdentityDir "RESULT.json"),
        "--polynomial-result", (Join-Path $PolynomialDir "RESULT.json"),
        "--bspline-result", (Join-Path $BsplineDir "RESULT.json"),
        "--coupled-result", (Join-Path $CoupledDir "RESULT.json"),
        "--output-dir", $SummaryDir
    )
    $Summary = Get-Content -LiteralPath (Join-Path $SummaryDir "RESULT.json") `
        -Raw | ConvertFrom-Json
    $JointSummary = Get-Content -LiteralPath (Join-Path $JointIdentityDir "RESULT.json") `
        -Raw | ConvertFrom-Json
    $JointMode = @($JointSummary.mode_summaries | Where-Object {
        [string]$_.method -eq "temporal_joint"
    })[0]
    return [PSCustomObject]@{
        priority = [int]$Window.priority
        purpose = $WindowPurpose
        sequence = $SequenceId
        start_frame = $StartFrame
        end_frame = $EndFrame
        status = "complete"
        temporal_joint_valid_frames = [int]$JointMode.valid_two_lane_frames
        temporal_joint_valid_rate = [double]$JointMode.valid_two_lane_rate
        joint_valid_frame_change = [int]$Summary.registered_decisions.joint_valid_frame_change
        coupled_relative_rmse_change_percent = $Summary.registered_decisions.coupled_relative_rmse_change_percent
        output = $WindowDir
    }
}

$Rows = @()
foreach ($Window in $Windows) {
    try {
        $Rows += Invoke-Window -Window $Window
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
$Rows = @($Rows | Where-Object {
    $null -ne $_ -and $null -ne $_.PSObject.Properties["status"]
})
$Rows | Export-Csv -LiteralPath (Join-Path $OutputRoot "BATCH_SUMMARY.csv") `
    -NoTypeInformation -Encoding UTF8
$Complete = @($Rows | Where-Object { [string]$_.status -eq "complete" }).Count
$Failed = @($Rows | Where-Object { [string]$_.status -eq "failed" }).Count
$Status = [ordered]@{
    status = if ($Failed -eq 0) { "complete" } elseif ($Complete -gt 0) { "partial" } else { "failed" }
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
foreach ($Row in $Rows | Where-Object { [string]$_.status -eq "complete" }) {
    $Name = "sequence_{0}_{1:000000}_{2:000000}" -f `
        $Row.sequence, $Row.start_frame, $Row.end_frame
    $Target = Join-Path $BundleDir $Name
    New-Item -ItemType Directory -Path $Target | Out-Null
    $Source = [string]$Row.output
    Copy-Item -LiteralPath (Join-Path $Source "10_improvement_summary") `
        -Destination (Join-Path $Target "summary") -Recurse
    Copy-Item -LiteralPath (Join-Path $Source "04_identity_old\RESULT.json") `
        -Destination (Join-Path $Target "identity_old_RESULT.json")
    Copy-Item -LiteralPath (Join-Path $Source "05_identity_joint\RESULT.json") `
        -Destination (Join-Path $Target "identity_joint_RESULT.json")
    Copy-Item -LiteralPath (Join-Path $Source "07_polynomial\03_figures\polynomial_left_right.png") `
        -Destination (Join-Path $Target "polynomial.png")
    Copy-Item -LiteralPath (Join-Path $Source "08_independent_bspline\03_figures\bspline_left_right.png") `
        -Destination (Join-Path $Target "independent_bspline.png")
    Copy-Item -LiteralPath (Join-Path $Source "09_coupled_bspline\03_figures\coupled_bspline_left_right.png") `
        -Destination (Join-Path $Target "coupled_bspline.png")
    Copy-Item -LiteralPath (Join-Path $Source "09_coupled_bspline\RESULT.json") `
        -Destination (Join-Path $Target "coupled_RESULT.json")
}
$BundleZip = Join-Path $OutputRoot "curved_improvement_bundle.zip"
Compress-Archive -LiteralPath $BundleDir -DestinationPath $BundleZip

Write-Host ""
Write-Host "CURVED IMPROVEMENT RUN FINISHED"
Write-Host "Output root: $OutputRoot"
Write-Host "Completed windows: $Complete; failed windows: $Failed"
Write-Host "Upload this result bundle: $BundleZip"
if ($Complete -eq 0) {
    throw "No selected window completed. Review STATUS.json."
}
