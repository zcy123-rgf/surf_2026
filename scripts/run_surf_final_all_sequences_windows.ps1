<#
.SYNOPSIS
Runs the frozen SURF lane pipeline over complete KITTI Odometry Sequences.

.DESCRIPTION
The default batch covers Sequences 00-10.  Each Sequence is isolated in its
own output directory.  A failed Sequence is recorded and does not stop the
remaining batch.  Exact completed full-Sequence results are reused unless
-ForceRerun is supplied, so the verified Sequence 01 run is not repeated.

Only Sequence 01 currently has reviewed straight seed ranges.  Other
Sequences use the analyzer's lowest-curvature-quartile baseline and are
explicitly labelled automatic_baseline in the batch summary.  These runs
measure coverage, consistency and scalability; they are not official
lane-position accuracy benchmarks.
#>

[CmdletBinding()]
param(
    [string]$EnvName = "surf2026-win",
    [string]$DatasetRoot = "F:\BaiduNetdiskDownload\kitti\odometry",
    [string]$ClrnetRoot = "",
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cuda",
    [string]$SequenceIds = "00,01,02,03,04,05,06,07,08,09,10",
    [int]$WindowLength = 15,
    [int]$WindowStride = 10,
    [string]$OutputRoot = "",
    [switch]$ForceRerun,
    [switch]$StopOnFailure
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
if ([string]::IsNullOrWhiteSpace($ClrnetRoot)) {
    $ClrnetRoot = Join-Path $ProjectRoot "CLRNet"
}
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found. Open Anaconda PowerShell Prompt and retry."
}

$Runner = Join-Path $PSScriptRoot "run_surf_final_window_windows.ps1"
$InputResolver = Join-Path $PSScriptRoot "kitti_odometry_workstation_input.ps1"
$Summarizer = Join-Path $PSScriptRoot "summarize_surf_final_all_sequences.py"
foreach ($Required in @($Runner, $InputResolver, $Summarizer)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required script is missing: $Required"
    }
}
. $InputResolver

$Ids = @(
    $SequenceIds.Split(",") |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ }
)
if (-not $Ids) { throw "SequenceIds is empty." }
foreach ($Id in $Ids) {
    if ($Id -notmatch "^\d{2}$") { throw "Bad sequence id: $Id" }
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputRoot = Join-Path $ProjectRoot (
        "workstation_outputs\surf_final_all_sequences_$Stamp"
    )
}
if (-not (Test-Path -LiteralPath $OutputRoot)) {
    New-Item -ItemType Directory -Path $OutputRoot | Out-Null
}
$OutputRoot = (Resolve-Path -LiteralPath $OutputRoot).Path

$ReviewedStraightSeeds = @{
    "01" = "851-875,991-1005"
}

function Get-MetricPropertyCount {
    param([object]$MetricObject, [string]$Name)
    if ($null -eq $MetricObject) { return 0 }
    $Property = $MetricObject.PSObject.Properties[$Name]
    if ($null -eq $Property) { return 0 }
    return [int]$Property.Value
}

function Read-CompatibleMetrics {
    param(
        [string]$Path,
        [string]$SequenceId,
        [int]$StartFrame,
        [int]$EndFrame
    )
    try {
        $Metrics = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        if (
            [string]$Metrics.sequence_id -eq $SequenceId -and
            [int]$Metrics.frames[0] -eq $StartFrame -and
            [int]$Metrics.frames[1] -eq $EndFrame -and
            [string]$Metrics.status -like "complete*"
        ) {
            return $Metrics
        }
    } catch {
        return $null
    }
    return $null
}

function Write-BatchState {
    param([hashtable]$RowsById, [string]$Root, [string[]]$RequestedIds)
    $Rows = @($RowsById.Values | Sort-Object sequence_id)
    $Rows | Export-Csv -LiteralPath (Join-Path $Root "BATCH_SUMMARY.csv") `
        -NoTypeInformation -Encoding UTF8
    $Complete = @($Rows | Where-Object { $_.status -like "complete*" }).Count
    $Failed = @($Rows | Where-Object { $_.status -eq "failed" }).Count
    $State = [ordered]@{
        status = if ($Rows.Count -lt $RequestedIds.Count) {
            "running"
        } elseif ($Failed -gt 0) {
            "complete_with_failures"
        } else {
            "complete"
        }
        requested_sequences = $RequestedIds
        recorded_sequences = $Rows.Count
        completed_sequences = $Complete
        failed_sequences = $Failed
        total_requested_frames = [int](
            ($Rows | Measure-Object -Property frame_count -Sum).Sum
        )
        output_root = $Root
        previous_outputs_modified = $false
        interpretation = @(
            "Sequence 01 uses reviewed straight seeds when executed by this batch",
            "other Sequences use an automatic low-curvature baseline unless a compatible result is reused",
            "held-out errors measure CLRNet/IPM consistency, not official lane-position accuracy",
            "accepted occlusion bridges are low-confidence hypotheses, never observations"
        )
    }
    $State | ConvertTo-Json -Depth 8 |
        Set-Content -LiteralPath (Join-Path $Root "BATCH_STATUS.json") `
            -Encoding UTF8
}

# Index exact completed full-Sequence runs before starting the batch.
$ExistingByKey = @{}
if (-not $ForceRerun) {
    $SearchRoots = @(
        (Join-Path $ProjectRoot "results"),
        (Join-Path $ProjectRoot "workstation_outputs")
    ) | Where-Object { Test-Path -LiteralPath $_ -PathType Container }
    foreach ($SearchRoot in $SearchRoots) {
        foreach ($File in @(
            Get-ChildItem -LiteralPath $SearchRoot -Recurse -File `
                -Filter "FINAL_METRICS.json" -ErrorAction SilentlyContinue
        )) {
            try {
                $Metrics = Get-Content -LiteralPath $File.FullName -Raw |
                    ConvertFrom-Json
                if ([string]$Metrics.status -like "complete*") {
                    $Key = "{0}:{1}:{2}" -f (
                        [string]$Metrics.sequence_id,
                        [int]$Metrics.frames[0],
                        [int]$Metrics.frames[1]
                    )
                    $CandidateRoot = Split-Path -Parent $File.FullName
                    $CandidateHasStages = Test-Path -LiteralPath (
                        Join-Path $CandidateRoot "01_lane_detection_and_tracking"
                    ) -PathType Container
                    if (-not $ExistingByKey.ContainsKey($Key)) {
                        $ExistingByKey[$Key] = $File.FullName
                    } elseif ($CandidateHasStages) {
                        # Prefer the complete result root over the flat review copy.
                        $ExistingByKey[$Key] = $File.FullName
                    }
                }
            } catch {
                Write-Warning "Ignored unreadable metrics file: $($File.FullName)"
            }
        }
    }
}

$RowsById = @{}
$PreviousSummary = Join-Path $OutputRoot "BATCH_SUMMARY.csv"
if (Test-Path -LiteralPath $PreviousSummary -PathType Leaf) {
    foreach ($Row in @(Import-Csv -LiteralPath $PreviousSummary)) {
        $RowsById[[string]$Row.sequence_id] = $Row
    }
}

foreach ($Id in $Ids) {
    $Input = $null
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "SURF full batch: KITTI Odometry Sequence $Id"
    Write-Host "============================================================"
    try {
        $Paths = Resolve-KittiOdometryWorkstationInput `
            -DatasetRoot $DatasetRoot -SequenceId $Id
        $Input = Test-KittiOdometrySequenceInput -InputPaths $Paths
        $StartFrame = 0
        $EndFrame = [int]$Input.MaximumFrame
        $FrameCount = [int]$Input.FrameCount
        $Key = "${Id}:${StartFrame}:${EndFrame}"
        $MetricsPath = $null
        $Source = "executed"

        if (-not $ForceRerun -and $ExistingByKey.ContainsKey($Key)) {
            $MetricsPath = [string]$ExistingByKey[$Key]
            $Source = "reused_exact_completed_result"
            Write-Host "Reusing exact completed result: $MetricsPath"
        } else {
            $SequenceRoot = Join-Path $OutputRoot "sequence_$Id"
            if (-not (Test-Path -LiteralPath $SequenceRoot)) {
                New-Item -ItemType Directory -Path $SequenceRoot | Out-Null
            }
            $AttemptStamp = Get-Date -Format "yyyyMMdd_HHmmss"
            $AttemptRoot = Join-Path $SequenceRoot "attempt_$AttemptStamp"
            $RunnerParameters = @{
                EnvName = $EnvName
                DatasetRoot = $DatasetRoot
                ClrnetRoot = $ClrnetRoot
                Device = $Device
                SequenceId = $Id
                StartFrame = $StartFrame
                EndFrame = $EndFrame
                WindowLength = $WindowLength
                WindowStride = $WindowStride
                OutputRoot = $AttemptRoot
            }
            if ($ReviewedStraightSeeds.ContainsKey($Id)) {
                $RunnerParameters.StraightSeedRanges = $ReviewedStraightSeeds[$Id]
            }
            & $Runner @RunnerParameters
            $MetricsPath = Join-Path $AttemptRoot "FINAL_METRICS.json"
        }

        $Metrics = Read-CompatibleMetrics -Path $MetricsPath `
            -SequenceId $Id -StartFrame $StartFrame -EndFrame $EndFrame
        if ($null -eq $Metrics) {
            throw "Compatible completed metrics were not produced: $MetricsPath"
        }
        $MetricsDir = Split-Path -Parent $MetricsPath
        $Models = $Metrics.selected_model_counts
        $RowsById[$Id] = [PSCustomObject][ordered]@{
            sequence_id = $Id
            status = [string]$Metrics.status
            source = $Source
            frame_count = $FrameCount
            threshold_baseline = [string]$Metrics.curvature_thresholds.baseline_source
            straight_threshold_1pm = [double]$Metrics.curvature_thresholds.straight_threshold_abs_curvature_1pm
            curve_threshold_1pm = [double]$Metrics.curvature_thresholds.curve_threshold_abs_curvature_1pm
            left_observation_fraction = [double]$Metrics.observation_coverage.left_fraction
            right_observation_fraction = [double]$Metrics.observation_coverage.right_fraction
            both_observation_fraction = [double]$Metrics.observation_coverage.both_fraction
            completed_fit_fraction = [double]$Metrics.fitting_coverage.completed_fraction
            continuity_pass_fraction = [double]$Metrics.continuity.passed_fraction
            straight_polynomial_fits = Get-MetricPropertyCount $Models "straight|parametric_polynomial"
            transition_bspline_fits = Get-MetricPropertyCount $Models "transition|parametric_cubic_bspline"
            curve_polynomial_fits = Get-MetricPropertyCount $Models "curve|parametric_polynomial"
            curve_bspline_fits = Get-MetricPropertyCount $Models "curve|parametric_cubic_bspline"
            accepted_low_confidence_bridges = [int]$Metrics.bridge_audit.accepted_low_confidence_hypotheses
            rejected_gaps = [int]$Metrics.bridge_audit.rejected_gaps
            result_directory = $MetricsDir
            metrics_path = $MetricsPath
            error = ""
        }
    } catch {
        $Message = $_.Exception.Message
        Write-Warning "Sequence $Id failed: $Message"
        $FrameCountValue = 0
        try {
            if ($null -ne $Input) { $FrameCountValue = [int]$Input.FrameCount }
        } catch {}
        $RowsById[$Id] = [PSCustomObject][ordered]@{
            sequence_id = $Id
            status = "failed"
            source = "attempted"
            frame_count = $FrameCountValue
            threshold_baseline = ""
            straight_threshold_1pm = ""
            curve_threshold_1pm = ""
            left_observation_fraction = ""
            right_observation_fraction = ""
            both_observation_fraction = ""
            completed_fit_fraction = ""
            continuity_pass_fraction = ""
            straight_polynomial_fits = ""
            transition_bspline_fits = ""
            curve_polynomial_fits = ""
            curve_bspline_fits = ""
            accepted_low_confidence_bridges = ""
            rejected_gaps = ""
            result_directory = ""
            metrics_path = ""
            error = $Message
        }
        Write-BatchState -RowsById $RowsById -Root $OutputRoot `
            -RequestedIds $Ids
        if ($StopOnFailure) { throw }
        continue
    }
    Write-BatchState -RowsById $RowsById -Root $OutputRoot `
        -RequestedIds $Ids
}

$FinalRows = @($RowsById.Values | Sort-Object sequence_id)
$ReviewStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ReviewDir = Join-Path $OutputRoot "review_bundle_$ReviewStamp"
New-Item -ItemType Directory -Path $ReviewDir | Out-Null
Copy-Item -LiteralPath (Join-Path $OutputRoot "BATCH_SUMMARY.csv") `
    -Destination $ReviewDir
Copy-Item -LiteralPath (Join-Path $OutputRoot "BATCH_STATUS.json") `
    -Destination $ReviewDir

foreach ($Row in $FinalRows) {
    if ($Row.status -notlike "complete*" -or -not $Row.result_directory) {
        continue
    }
    $SourceDir = [string]$Row.result_directory
    $TargetDir = Join-Path $ReviewDir "sequence_$($Row.sequence_id)"
    New-Item -ItemType Directory -Path $TargetDir | Out-Null
    $ReviewFiles = @(
        (Join-Path $SourceDir "FINAL_STATUS.json"),
        (Join-Path $SourceDir "FINAL_METRICS.json"),
        (Join-Path $SourceDir "02_pose_curvature\pose_curvature_overview.png"),
        (Join-Path $SourceDir "03_curve_models_and_fusion\adaptive_piecewise_xz_overview.png"),
        (Join-Path $SourceDir "04_occlusion_hypotheses\occlusion_bridge_overview.png")
    )
    foreach ($File in $ReviewFiles) {
        if (Test-Path -LiteralPath $File -PathType Leaf) {
            Copy-Item -LiteralPath $File -Destination $TargetDir
        }
    }
}

$CompleteCount = @($FinalRows | Where-Object { $_.status -like "complete*" }).Count
$FailedCount = @($FinalRows | Where-Object { $_.status -eq "failed" }).Count
$SummaryLines = @(
    "# SURF full KITTI Odometry batch",
    "",
    "- Requested Sequences: $($Ids -join ', ')",
    "- Completed: $CompleteCount",
    "- Failed: $FailedCount",
    "- Total dataset frames recorded: $((($FinalRows | Measure-Object frame_count -Sum).Sum))",
    "- Coordinate system: metric Cartesian common-reference X/Z",
    "- Accuracy scope: internal CLRNet/IPM consistency, not official lane ground truth",
    "",
    "See BATCH_SUMMARY.csv for per-Sequence coverage, model and bridge results."
)
$SummaryPath = Join-Path $ReviewDir "BATCH_README.md"
$SummaryLines | Set-Content -LiteralPath $SummaryPath -Encoding UTF8
& conda run --no-capture-output -n $EnvName python $Summarizer `
    --batch-summary (Join-Path $OutputRoot "BATCH_SUMMARY.csv") `
    --output-dir $ReviewDir
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Cross-Sequence visualization failed; raw batch CSV is intact."
}
$ReviewZip = Join-Path $OutputRoot "surf_final_all_sequences_review_$ReviewStamp.zip"
Compress-Archive -Path (Join-Path $ReviewDir "*") `
    -DestinationPath $ReviewZip

Write-Host ""
Write-Host "SURF ALL-SEQUENCES BATCH FINISHED"
Write-Host "Completed: $CompleteCount; failed: $FailedCount"
Write-Host "Output root: $OutputRoot"
Write-Host "Batch summary: $(Join-Path $OutputRoot 'BATCH_SUMMARY.csv')"
Write-Host "Review ZIP: $ReviewZip"
