[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ExistingRunRoot,
    [string]$EnvName = "surf2026-win",
    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"
$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RootDir
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found. Open Anaconda PowerShell Prompt and retry."
}
foreach ($Relative in @(
    "scripts\diagnose_curved_scan_failures.py",
    "scripts\fit_piecewise_bspline.py"
)) {
    if (-not (Test-Path -LiteralPath (Join-Path $RootDir $Relative) -PathType Leaf)) {
        throw "Required file is missing: $Relative"
    }
}
if (-not (Test-Path -LiteralPath $ExistingRunRoot -PathType Container)) {
    throw "ExistingRunRoot does not exist: $ExistingRunRoot"
}
$ExistingRunRoot = (Resolve-Path -LiteralPath $ExistingRunRoot).Path
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputRoot = Join-Path $RootDir "workstation_outputs\curved_piecewise_retry_$Stamp"
}
if (Test-Path -LiteralPath $OutputRoot) {
    if (Get-ChildItem -LiteralPath $OutputRoot -Force) {
        throw "OutputRoot must be new or empty: $OutputRoot"
    }
} else {
    New-Item -ItemType Directory -Path $OutputRoot | Out-Null
}
$OutputRoot = (Resolve-Path -LiteralPath $OutputRoot).Path

function Invoke-CondaPython {
    param([string]$Label, [string[]]$PythonArgs)
    Write-Host ""
    Write-Host "[$Label]"
    & conda run --no-capture-output -n $EnvName python @PythonArgs | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

$Rows = @()
$SequenceDirs = @(Get-ChildItem -LiteralPath $ExistingRunRoot -Directory | Where-Object {
    $_.Name -like "sequence_*"
})
if ($SequenceDirs.Count -eq 0) {
    throw "No sequence_* result directories were found in $ExistingRunRoot"
}
foreach ($SequenceDir in $SequenceDirs) {
    try {
        $OldScan = Join-Path $SequenceDir.FullName "02_temporal_ego\scan.json"
        $JointScan = Join-Path $SequenceDir.FullName "03_temporal_joint\scan.json"
        $Selection = Join-Path $SequenceDir.FullName "06_fixed_joint_selection\selection.json"
        $IndependentResult = Join-Path $SequenceDir.FullName "08_independent_bspline\RESULT.json"
        foreach ($Path in @($OldScan, $JointScan, $Selection, $IndependentResult)) {
            if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
                throw "Existing result input is missing: $Path"
            }
        }
        $Independent = Get-Content -LiteralPath $IndependentResult -Raw | ConvertFrom-Json
        $Poses = [string]$Independent.poses
        $Calib = [string]$Independent.calib
        foreach ($Path in @($Poses, $Calib)) {
            if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
                throw "KITTI input recorded by the previous run is missing: $Path"
            }
        }

        $Destination = Join-Path $OutputRoot $SequenceDir.Name
        New-Item -ItemType Directory -Path $Destination | Out-Null
        $DiagnosisDir = Join-Path $Destination "01_failure_diagnosis"
        Invoke-CondaPython -Label "$($SequenceDir.Name): failure diagnosis" -PythonArgs @(
            "scripts\diagnose_curved_scan_failures.py",
            "--scan", ("temporal_ego,{0}" -f $OldScan),
            "--scan", ("temporal_joint,{0}" -f $JointScan),
            "--output-dir", $DiagnosisDir
        )
        $PiecewiseDir = Join-Path $Destination "02_piecewise_bspline"
        Invoke-CondaPython -Label "$($SequenceDir.Name): piecewise B-spline" -PythonArgs @(
            "scripts\fit_piecewise_bspline.py",
            "--selection-json", $Selection,
            "--poses", $Poses,
            "--calib", $Calib,
            "--output-dir", $PiecewiseDir
        )

        $Diagnosis = Get-Content -LiteralPath (Join-Path $DiagnosisDir "RESULT.json") `
            -Raw | ConvertFrom-Json
        $Piecewise = Get-Content -LiteralPath (Join-Path $PiecewiseDir "RESULT.json") `
            -Raw | ConvertFrom-Json
        $IndependentSelected = @($Independent.cross_validation | Where-Object {
            [bool]$_.selected_by_one_standard_error_rule
        })[0]
        $IndependentRmse = [double]$IndependentSelected.mean_fold_rmse_m
        $PiecewiseRmse = [double]$Piecewise.selected_model.mean_fold_rmse_m
        $OldDiagnosis = @($Diagnosis.summary | Where-Object {
            [string]$_.method -eq "temporal_ego"
        })[0]
        $Summary = [ordered]@{
            status = "complete"
            sequence_result = $SequenceDir.Name
            valid_two_lane_metric_frames = [int]$OldDiagnosis.valid_two_lane_metric_frames
            fewer_than_two_clrnet_candidates = [int]$OldDiagnosis.fewer_than_two_clrnet_candidates
            image_side_pairing_failed = [int]$OldDiagnosis.image_side_pairing_failed
            temporal_distance_gate_failed = [int]$OldDiagnosis.temporal_distance_gate_failed
            other_pair_selection_failure = [int]$OldDiagnosis.other_pair_selection_failure
            selected_pair_has_too_few_ipm_points = [int]$OldDiagnosis.selected_pair_has_too_few_ipm_points
            independent_bspline_mean_fold_rmse_m = $IndependentRmse
            piecewise_bspline_mean_fold_rmse_m = $PiecewiseRmse
            piecewise_relative_rmse_change_percent = 100.0 * ($PiecewiseRmse - $IndependentRmse) / $IndependentRmse
            piecewise_selection_gate_passed = [bool]$Piecewise.selection_gate_passed
            piecewise_curves_cross = $Piecewise.width_check.curves_cross
            previous_outputs_modified = $false
        }
        $Summary | ConvertTo-Json -Depth 8 | Set-Content `
            -LiteralPath (Join-Path $Destination "RESULT.json") -Encoding UTF8
        $Rows += [PSCustomObject]$Summary
    } catch {
        $Rows += [PSCustomObject]@{
            status = "failed"
            sequence_result = $SequenceDir.Name
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
@{
    status = if ($Failed -eq 0) { "complete" } elseif ($Complete -gt 0) { "partial" } else { "failed" }
    completed = $Complete
    failed = $Failed
    input_run_root = $ExistingRunRoot
    output_root = $OutputRoot
    previous_outputs_modified = $false
} | ConvertTo-Json -Depth 6 | Set-Content `
    -LiteralPath (Join-Path $OutputRoot "STATUS.json") -Encoding UTF8

$BundleDir = Join-Path $OutputRoot "MEETING_BUNDLE"
New-Item -ItemType Directory -Path $BundleDir | Out-Null
Copy-Item -LiteralPath (Join-Path $OutputRoot "BATCH_SUMMARY.csv") -Destination $BundleDir
Copy-Item -LiteralPath (Join-Path $OutputRoot "STATUS.json") -Destination $BundleDir
foreach ($Row in $Rows | Where-Object { [string]$_.status -eq "complete" }) {
    $Source = Join-Path $OutputRoot ([string]$Row.sequence_result)
    $Target = Join-Path $BundleDir ([string]$Row.sequence_result)
    New-Item -ItemType Directory -Path $Target | Out-Null
    Copy-Item -LiteralPath (Join-Path $Source "RESULT.json") -Destination $Target
    Copy-Item -LiteralPath (Join-Path $Source "01_failure_diagnosis") `
        -Destination (Join-Path $Target "failure_diagnosis") -Recurse
    Copy-Item -LiteralPath (Join-Path $Source "02_piecewise_bspline\RESULT.json") `
        -Destination (Join-Path $Target "piecewise_RESULT.json")
    Copy-Item -LiteralPath (Join-Path $Source "02_piecewise_bspline\01_evaluation") `
        -Destination (Join-Path $Target "piecewise_evaluation") -Recurse
    Copy-Item -LiteralPath (Join-Path $Source "02_piecewise_bspline\03_figures\piecewise_bspline_left_right.png") `
        -Destination (Join-Path $Target "piecewise_bspline.png")
    Copy-Item -LiteralPath (Join-Path $ExistingRunRoot "$($Row.sequence_result)\08_independent_bspline\RESULT.json") `
        -Destination (Join-Path $Target "independent_RESULT.json")
    Copy-Item -LiteralPath (Join-Path $ExistingRunRoot "$($Row.sequence_result)\08_independent_bspline\03_figures\bspline_left_right.png") `
        -Destination (Join-Path $Target "independent_bspline.png")
}
$BundleZip = Join-Path $OutputRoot "piecewise_retry_bundle.zip"
Compress-Archive -LiteralPath $BundleDir -DestinationPath $BundleZip

Write-Host ""
Write-Host "PIECEWISE RETRY FINISHED"
Write-Host "Output root: $OutputRoot"
Write-Host "Completed: $Complete; failed: $Failed"
Write-Host "Upload this result bundle: $BundleZip"
if ($Complete -eq 0) {
    throw "No piecewise retry completed. Review STATUS.json."
}
