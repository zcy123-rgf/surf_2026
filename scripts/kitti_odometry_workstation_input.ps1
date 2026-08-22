Set-StrictMode -Version Latest

function Resolve-KittiOdometryWorkstationInput {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$DatasetRoot,
        [Parameter(Mandatory = $true)]
        [ValidatePattern('^\d{2}$')]
        [string]$SequenceId
    )

    if (-not (Test-Path -LiteralPath $DatasetRoot -PathType Container)) {
        throw "KITTI odometry root does not exist: $DatasetRoot"
    }
    $Root = (Resolve-Path -LiteralPath $DatasetRoot).Path

    $Candidates = @(
        [PSCustomObject]@{
            Layout = "official-separated-downloads"
            Root = $Root
            SequenceId = $SequenceId
            ImageDir = Join-Path $Root (
                "data_odometry_color\dataset\sequences\$SequenceId\image_2"
            )
            Calib = Join-Path $Root (
                "data_odometry_calib\dataset\sequences\$SequenceId\calib.txt"
            )
            Poses = Join-Path $Root (
                "data_odometry_poses\dataset\poses\$SequenceId.txt"
            )
            Times = Join-Path $Root (
                "data_odometry_color\dataset\sequences\$SequenceId\times.txt"
            )
        }
    )

    $MergedRoots = @()
    if (Test-Path -LiteralPath (Join-Path $Root "dataset\sequences") -PathType Container) {
        $MergedRoots += (Join-Path $Root "dataset")
    }
    if (Test-Path -LiteralPath (Join-Path $Root "sequences") -PathType Container) {
        $MergedRoots += $Root
    }
    foreach ($MergedRoot in $MergedRoots) {
        $Candidates += [PSCustomObject]@{
            Layout = "merged-dataset"
            Root = $MergedRoot
            SequenceId = $SequenceId
            ImageDir = Join-Path $MergedRoot "sequences\$SequenceId\image_2"
            Calib = Join-Path $MergedRoot "sequences\$SequenceId\calib.txt"
            Poses = Join-Path $MergedRoot "poses\$SequenceId.txt"
            Times = Join-Path $MergedRoot "sequences\$SequenceId\times.txt"
        }
    }

    foreach ($Candidate in $Candidates) {
        $Required = @(
            $Candidate.ImageDir,
            $Candidate.Calib,
            $Candidate.Poses,
            $Candidate.Times
        )
        if (-not ($Required | Where-Object { -not (Test-Path -LiteralPath $_) })) {
            return $Candidate
        }
    }

    throw @"
Could not resolve KITTI Odometry Sequence $SequenceId under:
  $Root
Expected the official separated downloads or a merged dataset containing:
  sequences\$SequenceId\image_2
  sequences\$SequenceId\calib.txt
  poses\$SequenceId.txt
  sequences\$SequenceId\times.txt
"@
}

function Test-KittiOdometrySequenceInput {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [PSCustomObject]$InputPaths,
        [int]$MinimumFrameCount = 105
    )

    $PoseLines = @(Get-Content -LiteralPath $InputPaths.Poses)
    $TimeLines = @(Get-Content -LiteralPath $InputPaths.Times)
    $ImageFiles = @(
        Get-ChildItem -LiteralPath $InputPaths.ImageDir -Filter "*.png" -File |
            Sort-Object Name
    )
    $ImageCount = $ImageFiles.Count
    $PoseCount = $PoseLines.Count
    $TimeCount = $TimeLines.Count

    if ($ImageCount -lt $MinimumFrameCount) {
        throw (
            "Sequence $($InputPaths.SequenceId) has only $ImageCount images; " +
            "at least $MinimumFrameCount are required."
        )
    }
    if ($ImageCount -ne $PoseCount -or $ImageCount -ne $TimeCount) {
        throw (
            "Sequence $($InputPaths.SequenceId) is not aligned: " +
            "images=$ImageCount, poses=$PoseCount, times=$TimeCount."
        )
    }
    for ($Index = 0; $Index -lt $PoseCount; $Index++) {
        if (($PoseLines[$Index].Trim() -split "\s+").Count -ne 12) {
            throw (
                "Sequence $($InputPaths.SequenceId) pose row $Index does not " +
                "contain 12 values."
            )
        }
    }

    $FirstExpected = "000000.png"
    $LastExpected = "{0:000000}.png" -f ($ImageCount - 1)
    if ($ImageFiles[0].Name -ne $FirstExpected -or $ImageFiles[-1].Name -ne $LastExpected) {
        throw (
            "Sequence $($InputPaths.SequenceId) image names are not a contiguous " +
            "zero-based range: first=$($ImageFiles[0].Name), " +
            "last=$($ImageFiles[-1].Name), expected last=$LastExpected."
        )
    }

    $CalibText = Get-Content -LiteralPath $InputPaths.Calib -Raw
    if ($CalibText -notmatch "(?m)^P2:\s+(?:[-+0-9.eE]+\s+){11}[-+0-9.eE]+\s*$") {
        throw (
            "Sequence $($InputPaths.SequenceId) calibration does not contain " +
            "a valid 3x4 P2 row: $($InputPaths.Calib)"
        )
    }

    return [PSCustomObject]@{
        SequenceId = $InputPaths.SequenceId
        Layout = $InputPaths.Layout
        FrameCount = $ImageCount
        MaximumFrame = $ImageCount - 1
        ImageCount = $ImageCount
        PoseRows = $PoseCount
        TimeRows = $TimeCount
        ImageDir = $InputPaths.ImageDir
        Calib = $InputPaths.Calib
        Poses = $InputPaths.Poses
        Times = $InputPaths.Times
    }
}
