[CmdletBinding()]
param(
    [string]$EnvName = "surf2026-win",
    [string]$DatasetRoot = "F:\BaiduNetdiskDownload\kitti\odometry",
    [string]$First20OutputRoot = "",
    [string]$OutputRoot = "",
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [string[]]$CurvedRanges = @("95-114", "1549-1568"),
    [switch]$SkipCurvedCandidates
)

$ErrorActionPreference = "Stop"
$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RootDir

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found. Open Anaconda PowerShell Prompt and retry."
}

. (Join-Path $PSScriptRoot "kitti00_workstation_input.ps1")
$Kitti = Resolve-Kitti00WorkstationInput -DatasetRoot $DatasetRoot
$Verification = Test-Kitti00FirstFiveInput `
    -InputPaths $Kitti `
    -RequireCompleteSequence

if ([string]::IsNullOrWhiteSpace($First20OutputRoot)) {
    $Latest = Get-ChildItem (Join-Path $RootDir "workstation_outputs") -Directory |
        Where-Object Name -Like "first20_two_curves_*" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $Latest) {
        throw "No first20_two_curves_* output was found. Run run_first20_two_curve_windows.ps1 first."
    }
    $First20OutputRoot = $Latest.FullName
}
$First20OutputRoot = (Resolve-Path -LiteralPath $First20OutputRoot).Path
$First20Aligned = Join-Path $First20OutputRoot `
    "01_from_scratch_pipeline\00_metadata\aligned_lane_points.json"
$First20Detected = Join-Path $First20OutputRoot `
    "01_from_scratch_pipeline\00_metadata\detected_lane_points.json"

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputRoot = Join-Path $RootDir "workstation_outputs\meeting_four_questions_$Stamp"
}
if (Test-Path -LiteralPath $OutputRoot) {
    if (Get-ChildItem -LiteralPath $OutputRoot -Force) {
        throw "OutputRoot must be new or empty: $OutputRoot"
    }
} else {
    New-Item -ItemType Directory -Path $OutputRoot | Out-Null
}

$ManualJson = Join-Path $RootDir "annotations\kitti00_first5_manual_annotations.json"
$Required = @($First20Aligned, $First20Detected, $Kitti.ImageDir, $Kitti.Calib, $Kitti.Poses, $ManualJson)
$Missing = $Required | Where-Object { -not (Test-Path -LiteralPath $_) }
if ($Missing) {
    throw "Required inputs are missing:`n$($Missing -join [Environment]::NewLine)"
}

function Invoke-CondaPython {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [string[]]$PythonArgs
    )
    Write-Host ""
    Write-Host "[$Label]"
    & conda run --no-capture-output -n $EnvName python @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

Write-Host "Verified Sequence 00: $($Verification.ImageCount) images and $($Verification.PoseRows) poses"
Write-Host "Read-only first-20 source: $First20OutputRoot"
Write-Host "New isolated output: $OutputRoot"

$First20Pipeline = Join-Path $First20OutputRoot "01_from_scratch_pipeline"
$First20VisualGate = Join-Path $OutputRoot "00_first20_visual_identity_gate"
Invoke-CondaPython -Label "First-20 visual gate: original frames" `
    -PythonArgs @(
        "scripts\make_image_contact_sheet.py",
        "--input-dir", (Join-Path $First20Pipeline "01_original_frames"),
        "--output", (Join-Path $First20VisualGate "01_original_frames.png")
    )
Invoke-CondaPython -Label "First-20 visual gate: all CLRNet candidates" `
    -PythonArgs @(
        "scripts\make_image_contact_sheet.py",
        "--input-dir", (Join-Path $First20Pipeline "02_clrnet_points\all_candidates"),
        "--output", (Join-Path $First20VisualGate "02_all_candidates.png")
    )
Invoke-CondaPython -Label "First-20 visual gate: selected pair" `
    -PythonArgs @(
        "scripts\make_image_contact_sheet.py",
        "--input-dir", (Join-Path $First20Pipeline "02_clrnet_points\selected_two"),
        "--output", (Join-Path $First20VisualGate "03_selected_two.png")
    )

$StraightAnalysis = Join-Path $OutputRoot "01_first20_straight_analysis"
Invoke-CondaPython -Label "Question 1/3/4: first-20 direct fit, segment compression and manual pseudo-GT metrics" `
    -PythonArgs @(
        "scripts\analyze_lane_curve_hierarchy.py",
        "--aligned-json", $First20Aligned,
        "--detected-json", $First20Detected,
        "--reference-id", "19",
        "--segment-size", "5",
        "--feature-points-per-segment", "8",
        "--manual-json", $ManualJson,
        "--calib", $Kitti.Calib,
        "--poses", $Kitti.Poses,
        "--output-dir", $StraightAnalysis
    )

$CandidateStatuses = @()
if (-not $SkipCurvedCandidates) {
    foreach ($RangeText in $CurvedRanges) {
        if ($RangeText -notmatch '^(\d+)-(\d+)$') {
            throw "Invalid curved range '$RangeText'. Use start-end, for example 95-114."
        }
        $StartFrame = [int]$Matches[1]
        $EndFrame = [int]$Matches[2]
        if ($EndFrame -lt $StartFrame) {
            throw "Invalid decreasing curved range '$RangeText'."
        }
        $FrameArray = $StartFrame..$EndFrame
        if ($FrameArray.Count -lt 5 -or ($FrameArray.Count % 5) -ne 0) {
            throw "Curved range '$RangeText' must contain at least 5 frames and be divisible by 5."
        }
        $RequestedFrameIds = $FrameArray -join ","
        $RequestedStartFrame = $StartFrame
        $RequestedEndFrame = $EndFrame
        $RangeName = "frames_{0:000000}_{1:000000}" -f $StartFrame, $EndFrame
        $CandidateRoot = Join-Path $OutputRoot "02_curved_candidates\$RangeName"
        $ScanDir = Join-Path $CandidateRoot "00_detection_scan"
        $PipelineDir = Join-Path $CandidateRoot "01_from_scratch_pipeline"
        $CurveDir = Join-Path $CandidateRoot "02_direct_two_curves"
        $AnalysisDir = Join-Path $CandidateRoot "03_model_and_hierarchy_analysis"
        try {
            foreach ($FrameId in $FrameArray) {
                $Image = Join-Path $Kitti.ImageDir ($FrameId.ToString("000000") + ".png")
                if (-not (Test-Path -LiteralPath $Image)) {
                    throw "Missing image: $Image"
                }
            }
            Invoke-CondaPython -Label "Curved candidate ${RangeText}: preflight CLRNet candidate-count scan" `
                -PythonArgs @(
                    "scripts\scan_clrnet_lane_counts.py",
                    "--image-dir", $Kitti.ImageDir,
                    "--image-pattern", "{frame_id:06d}.png",
                    "--frame-ids", $RequestedFrameIds,
                    "--poses", $Kitti.Poses,
                    "--segment-size", "5",
                    "--minimum-candidates", "2",
                    "--device", $Device,
                    "--output-dir", $ScanDir
                )
            $VisualGate = Join-Path $CandidateRoot "00_visual_identity_gate"
            Invoke-CondaPython -Label "Curved candidate ${RangeText}: scan original contact sheet" `
                -PythonArgs @(
                    "scripts\make_image_contact_sheet.py",
                    "--input-dir", (Join-Path $ScanDir "original_frames"),
                    "--output", (Join-Path $VisualGate "01_scan_original_frames.png")
                )
            Invoke-CondaPython -Label "Curved candidate ${RangeText}: scan all-candidate contact sheet" `
                -PythonArgs @(
                    "scripts\make_image_contact_sheet.py",
                    "--input-dir", (Join-Path $ScanDir "all_candidates"),
                    "--output", (Join-Path $VisualGate "02_scan_all_candidates.png")
                )

            $RecommendationPath = Join-Path $ScanDir "recommendation.json"
            $Recommendation = Get-Content -LiteralPath $RecommendationPath -Raw |
                ConvertFrom-Json
            if ($Recommendation.status -ne "selected") {
                $CandidateStatuses += [PSCustomObject]@{
                    requested_range = $RangeText
                    selected_range = $null
                    status = "skipped_no_valid_two_candidate_run"
                    output = $CandidateRoot
                    detection_scan = $RecommendationPath
                    reason = $Recommendation.reason
                }
                Write-Warning "Curved candidate $RangeText has no valid consecutive subrange; no curve was fitted."
                continue
            }

            $FrameArray = @($Recommendation.frame_ids | ForEach-Object { [int]$_ })
            $StartFrame = [int]$Recommendation.selected_start
            $EndFrame = [int]$Recommendation.selected_end
            $FrameIds = $FrameArray -join ","
            $Weights = (($FrameArray | ForEach-Object { "1.0" }) -join ",")
            $SelectedRange = "$StartFrame-$EndFrame"

            Invoke-CondaPython -Label "Curved candidate ${RangeText}, selected ${SelectedRange}: CLRNet, metric IPM and pose alignment" `
                -PythonArgs @(
                    "scripts\run_full_point_pipeline.py",
                    "--image-dir", $Kitti.ImageDir,
                    "--image-pattern", "{frame_id:06d}.png",
                    "--calib", $Kitti.Calib,
                    "--poses", $Kitti.Poses,
                    "--frame-ids", $FrameIds,
                    "--reference-id", "$EndFrame",
                    "--local-x-range=-10,10",
                    "--local-z-range=3,50",
                    "--fusion-x-range=-20,20",
                    "--fusion-z-range=-20,50",
                    "--weights", $Weights,
                    "--device", $Device,
                    "--output-dir", $PipelineDir
                )
            Invoke-CondaPython -Label "Curved candidate ${RangeText}: selected-subrange pair contact sheet" `
                -PythonArgs @(
                    "scripts\make_image_contact_sheet.py",
                    "--input-dir", (Join-Path $PipelineDir "02_clrnet_points\selected_two"),
                    "--output", (Join-Path $VisualGate "03_selected_subrange_two.png")
                )
            $Aligned = Join-Path $PipelineDir "00_metadata\aligned_lane_points.json"
            $Detected = Join-Path $PipelineDir "00_metadata\detected_lane_points.json"
            Invoke-CondaPython -Label "Curved candidate ${RangeText}: direct two-curve fit" `
                -PythonArgs @(
                    "scripts\fit_first5_two_curves.py",
                    "--aligned-json", $Aligned,
                    "--frame-ids", $FrameIds,
                    "--reference-id", "$EndFrame",
                    "--x-range=-20,20",
                    "--z-range=-20,50",
                    "--output-dir", $CurveDir
                )
            Invoke-CondaPython -Label "Curved candidate ${RangeText}: polynomial/spline and segment-feature audit" `
                -PythonArgs @(
                    "scripts\analyze_lane_curve_hierarchy.py",
                    "--aligned-json", $Aligned,
                    "--detected-json", $Detected,
                    "--reference-id", "$EndFrame",
                    "--segment-size", "5",
                    "--feature-points-per-segment", "8",
                    "--output-dir", $AnalysisDir
                )
            $CandidateStatuses += [PSCustomObject]@{
                requested_range = $RangeText
                selected_range = $SelectedRange
                selected_frame_count = $FrameArray.Count
                status = "complete"
                output = $CandidateRoot
                detection_scan = $RecommendationPath
                validity_gate = "inspect originals, all_candidates and selected_two before claiming a curved lane result"
            }
        } catch {
            $CandidateStatuses += [PSCustomObject]@{
                requested_range = $RangeText
                status = "failed"
                output = $CandidateRoot
                error = $_.Exception.Message
            }
            Write-Warning "Curved candidate $RangeText failed: $($_.Exception.Message)"
        }
    }
}

$StatusPath = Join-Path $OutputRoot "STATUS.json"
@{
    status = "complete_with_candidate_statuses"
    dataset = "KITTI Odometry Sequence 00"
    first20_source = $First20OutputRoot
    previous_outputs_modified = $false
    first20_analysis = $StraightAnalysis
    curved_candidates = $CandidateStatuses
    warnings = @(
        "SemanticKITTI class-60 coverage helped choose candidates but is not lane-polyline ground truth.",
        "A curved candidate is valid only after image topology and selected lane identities are inspected.",
        "Manual labels cover frames 0-4 only and are pseudo-ground-truth, not official KITTI ground truth."
    )
} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $StatusPath -Encoding UTF8

Write-Host ""
Write-Host "Completed without modifying previous outputs."
Write-Host "Output root: $OutputRoot"
Write-Host "First open:"
Write-Host "  00_first20_visual_identity_gate\03_selected_two.png"
Write-Host "  01_first20_straight_analysis\00_audit\audit.json"
Write-Host "  01_first20_straight_analysis\03_hierarchical_refusion\direct_vs_refused.png"
Write-Host "  01_first20_straight_analysis\04_manual_pseudo_gt\direct_curve_metrics.csv"
Write-Host "For every curved candidate, inspect before using metrics:"
Write-Host "  00_detection_scan\lane_counts.csv"
Write-Host "  00_detection_scan\recommendation.json"
Write-Host "  00_visual_identity_gate\01_scan_original_frames.png"
Write-Host "  00_visual_identity_gate\02_scan_all_candidates.png"
Write-Host "  00_visual_identity_gate\03_selected_subrange_two.png (only when selected)"
