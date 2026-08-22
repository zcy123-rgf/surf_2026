[CmdletBinding()]
param(
    [string]$EnvName = "surf2026-win",
    [string]$DatasetRoot = "F:\BaiduNetdiskDownload\kitti\odometry",
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [ValidateSet("00", "01", "02", "03", "04", "05", "06", "07", "08", "09")]
    [string[]]$SequenceIds = @("00", "01", "02", "03", "04", "05", "06", "07", "08", "09"),
    [int]$TopCandidateCount = 100,
    [int]$SelectedRank = 1,
    [int]$MinimumValidFramesPerBlock = 10,
    [ValidateSet("coverage", "sustained_curve")]
    [string]$RankingMode = "sustained_curve",
    [double]$MinimumTrajectoryTurnDegPerBlock = 1.0,
    [ValidateSet("outermost", "ego_adjacent")]
    [string]$CandidateSelectionMode = "ego_adjacent",
    [ValidateSet("both", "polynomial", "bspline")]
    [string]$Model = "both",
    [string]$BatchOutputRoot = "",
    [switch]$StopOnFailure
)

$ErrorActionPreference = "Stop"
$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RootDir

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found. Open Anaconda PowerShell Prompt and retry."
}
if ($SequenceIds.Count -eq 0) {
    throw "At least one sequence must be selected."
}
if ($SequenceIds.Count -ne (@($SequenceIds | Sort-Object -Unique)).Count) {
    throw "SequenceIds must not contain duplicates."
}

if ([string]::IsNullOrWhiteSpace($BatchOutputRoot)) {
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $BatchOutputRoot = Join-Path $RootDir (
        "workstation_outputs\sequences00_09_independent_$Stamp"
    )
}
if (Test-Path -LiteralPath $BatchOutputRoot) {
    if (Get-ChildItem -LiteralPath $BatchOutputRoot -Force) {
        throw "Batch output directory must be new or empty: $BatchOutputRoot"
    }
} else {
    New-Item -ItemType Directory -Path $BatchOutputRoot | Out-Null
}
$BatchOutputRoot = (Resolve-Path -LiteralPath $BatchOutputRoot).Path

. (Join-Path $PSScriptRoot "kitti_odometry_workstation_input.ps1")
$InventoryRows = @()
$InventoryFailures = @()
foreach ($SequenceId in $SequenceIds) {
    try {
        $Paths = Resolve-KittiOdometryWorkstationInput `
            -DatasetRoot $DatasetRoot `
            -SequenceId $SequenceId
        $Check = Test-KittiOdometrySequenceInput -InputPaths $Paths
        $InventoryRows += [PSCustomObject]@{
            sequence_id = $SequenceId
            status = "ready"
            frame_count = [int]$Check.FrameCount
            maximum_frame = [int]$Check.MaximumFrame
            image_dir = $Check.ImageDir
            poses = $Check.Poses
            error = ""
        }
    } catch {
        $InventoryFailures += $SequenceId
        $InventoryRows += [PSCustomObject]@{
            sequence_id = $SequenceId
            status = "failed"
            frame_count = $null
            maximum_frame = $null
            image_dir = ""
            poses = ""
            error = $_.Exception.Message
        }
    }
}
$InventoryPath = Join-Path $BatchOutputRoot "DATASET_INVENTORY.csv"
$InventoryRows | Export-Csv -LiteralPath $InventoryPath `
    -NoTypeInformation -Encoding UTF8
$InventoryRows | Format-Table sequence_id,status,frame_count,maximum_frame
if ($InventoryFailures.Count -gt 0) {
    throw (
        "Dataset preflight failed for sequences: " +
        ($InventoryFailures -join ",") + ". See $InventoryPath"
    )
}
$TotalFrameCount = [int](
    ($InventoryRows | Measure-Object frame_count -Sum).Sum
)
Write-Host "Dataset preflight passed: $TotalFrameCount frames across $($SequenceIds.Count) sequences."

Write-Host "[PREFLIGHT] CUDA and CLRNet"
& conda run --no-capture-output -n $EnvName python `
    scripts\check_windows_env.py --device $Device --run-clrnet
if ($LASTEXITCODE -ne 0) {
    throw "Windows CUDA/CLRNet preflight failed with exit code $LASTEXITCODE."
}

$Runner = Join-Path $PSScriptRoot "run_single_odometry_sequence_curve_windows.ps1"
if (-not (Test-Path -LiteralPath $Runner -PathType Leaf)) {
    throw "Single-sequence runner is missing: $Runner"
}

$Rows = @()
foreach ($SequenceId in $SequenceIds) {
    $SequenceOutput = Join-Path $BatchOutputRoot "sequence_$SequenceId"
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "START SEQUENCE $SequenceId"
    Write-Host "Output: $SequenceOutput"
    Write-Host "============================================================"

    try {
        & $Runner `
            -EnvName $EnvName `
            -DatasetRoot $DatasetRoot `
            -SequenceId $SequenceId `
            -Device $Device `
            -TopCandidateCount $TopCandidateCount `
            -SelectedRank $SelectedRank `
            -MinimumValidFramesPerBlock $MinimumValidFramesPerBlock `
            -RankingMode $RankingMode `
            -MinimumTrajectoryTurnDegPerBlock `
                $MinimumTrajectoryTurnDegPerBlock `
            -CandidateSelectionMode $CandidateSelectionMode `
            -Model $Model `
            -OutputDir $SequenceOutput `
            -SkipPreflight | Out-Host

        $Result = Get-Content `
            -LiteralPath (Join-Path $SequenceOutput "RUN_STATUS.json") `
            -Raw | ConvertFrom-Json

        $Rows += [PSCustomObject]@{
            sequence_id = $SequenceId
            status = "complete"
            frame_count = [int]$Result.sequence_frame_count
            selected_frames = $Result.selected_frames -join "-"
            output_root = $Result.output_root
            error = ""
        }
    } catch {
        $Message = $_.Exception.Message
        $Rows += [PSCustomObject]@{
            sequence_id = $SequenceId
            status = "failed"
            frame_count = $null
            selected_frames = ""
            output_root = $SequenceOutput
            error = $Message
        }
        if (-not (Test-Path -LiteralPath $SequenceOutput)) {
            New-Item -ItemType Directory -Path $SequenceOutput | Out-Null
        }
        @{
            status = "failed"
            sequence_id = $SequenceId
            previous_outputs_modified = $false
            error = $Message
        } | ConvertTo-Json -Depth 5 | Set-Content `
            -LiteralPath (Join-Path $SequenceOutput "SEQUENCE_FAILURE.json") `
            -Encoding UTF8
        Write-Warning "Sequence $SequenceId failed: $Message"
        if ($StopOnFailure) {
            throw
        }
    }

    $Rows | Export-Csv `
        -LiteralPath (Join-Path $BatchOutputRoot "BATCH_SUMMARY.csv") `
        -NoTypeInformation -Encoding UTF8
    @{
        status = "running"
        dataset = "KITTI Odometry Sequences 00-09"
        sequences_requested = $SequenceIds
        sequences_finished = $Rows.Count
        total_dataset_frames = $TotalFrameCount
        ranking_mode = $RankingMode
        candidate_selection_mode = $CandidateSelectionMode
        previous_outputs_modified = $false
        results = $Rows
    } | ConvertTo-Json -Depth 7 | Set-Content `
        -LiteralPath (Join-Path $BatchOutputRoot "BATCH_STATUS.json") `
        -Encoding UTF8
}

$Completed = @($Rows | Where-Object status -eq "complete").Count
$Failed = @($Rows | Where-Object status -eq "failed").Count
$FinalStatus = if ($Failed -eq 0) { "complete" } else { "complete_with_failures" }
@{
    status = $FinalStatus
    dataset = "KITTI Odometry Sequences 00-09"
    sequences_requested = $SequenceIds
    completed_sequence_count = $Completed
    failed_sequence_count = $Failed
    total_dataset_frames = $TotalFrameCount
    ranking_mode = $RankingMode
    candidate_selection_mode = $CandidateSelectionMode
    previous_outputs_modified = $false
    results = $Rows
} | ConvertTo-Json -Depth 7 | Set-Content `
    -LiteralPath (Join-Path $BatchOutputRoot "BATCH_STATUS.json") `
    -Encoding UTF8

Write-Host ""
Write-Host "SEQUENCES 00-09 BATCH FINISHED"
Write-Host "Completed: $Completed"
Write-Host "Failed: $Failed"
Write-Host "Output root: $BatchOutputRoot"
Write-Host "Dataset inventory: $InventoryPath"
Write-Host "Summary: $(Join-Path $BatchOutputRoot 'BATCH_SUMMARY.csv')"
