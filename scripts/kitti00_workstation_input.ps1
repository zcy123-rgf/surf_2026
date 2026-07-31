Set-StrictMode -Version Latest

function Resolve-Kitti00WorkstationInput {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$DatasetRoot
    )

    if (-not (Test-Path -LiteralPath $DatasetRoot -PathType Container)) {
        throw "KITTI odometry root does not exist: $DatasetRoot"
    }
    $Root = (Resolve-Path -LiteralPath $DatasetRoot).Path

    $Split = [PSCustomObject]@{
        Layout   = "official-separated-downloads"
        Root     = $Root
        ImageDir = Join-Path $Root "data_odometry_color\dataset\sequences\00\image_2"
        Calib    = Join-Path $Root "data_odometry_calib\dataset\sequences\00\calib.txt"
        Poses    = Join-Path $Root "data_odometry_poses\dataset\poses\00.txt"
        Times    = Join-Path $Root "data_odometry_color\dataset\sequences\00\times.txt"
    }

    $MergedDataset = if (
        Test-Path -LiteralPath (Join-Path $Root "dataset\sequences\00") -PathType Container
    ) {
        Join-Path $Root "dataset"
    } elseif (
        Test-Path -LiteralPath (Join-Path $Root "sequences\00") -PathType Container
    ) {
        $Root
    } else {
        $null
    }

    $Candidates = @($Split)
    if ($null -ne $MergedDataset) {
        $Candidates += [PSCustomObject]@{
            Layout   = "merged-dataset"
            Root     = $MergedDataset
            ImageDir = Join-Path $MergedDataset "sequences\00\image_2"
            Calib    = Join-Path $MergedDataset "sequences\00\calib.txt"
            Poses    = Join-Path $MergedDataset "poses\00.txt"
            Times    = Join-Path $MergedDataset "sequences\00\times.txt"
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
Could not resolve a complete KITTI Odometry Sequence 00 input under:
  $Root
Expected either the official separated downloads:
  data_odometry_color\dataset\sequences\00\image_2
  data_odometry_calib\dataset\sequences\00\calib.txt
  data_odometry_poses\dataset\poses\00.txt
or a merged dataset containing sequences\00 and poses\00.txt.
"@
}

function Test-Kitti00FirstFiveInput {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [PSCustomObject]$InputPaths,
        [switch]$RequireCompleteSequence
    )

    $ExpectedHashes = [ordered]@{
        "000000.png" = "af34528f0edd14ddf8ee7ac9017f4b7f540e04e3d50cecce006245d950ea1d6c"
        "000001.png" = "cc8a7a3c3fa653b5edec3ac81dd7bda4e5391330c82070d8bc6c3e2831572263"
        "000002.png" = "18504a8bf4e71a2273cc31289f0c3fdd1a46942fdd59125a14bb14da58457522"
        "000003.png" = "0766814389f2650b556913b07f3b1e2344ad9f1e6e9defef7ccceddd22e2562a"
        "000004.png" = "b0d2c7364bdfb75b099d48303ff5e296b1004aa2b5969d81383b8c36c62b2c49"
    }

    $Verified = @()
    foreach ($Entry in $ExpectedHashes.GetEnumerator()) {
        $Path = Join-Path $InputPaths.ImageDir $Entry.Key
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
            throw "Required KITTI frame is missing: $Path"
        }
        $Actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($Actual -ne $Entry.Value) {
            throw "KITTI frame identity check failed for $Path. Expected $($Entry.Value), got $Actual."
        }
        $Verified += [PSCustomObject]@{
            Frame  = $Entry.Key
            SHA256 = $Actual
        }
    }

    $PoseLines = @(Get-Content -LiteralPath $InputPaths.Poses)
    if ($PoseLines.Count -lt 5) {
        throw "Pose file has fewer than five rows: $($InputPaths.Poses)"
    }
    for ($Index = 0; $Index -lt 5; $Index++) {
        if (($PoseLines[$Index].Trim() -split "\s+").Count -ne 12) {
            throw "Pose row $Index does not contain 12 values: $($InputPaths.Poses)"
        }
    }

    $CalibText = Get-Content -LiteralPath $InputPaths.Calib -Raw
    if ($CalibText -notmatch "(?m)^P2:\s+(?:[-+0-9.eE]+\s+){11}[-+0-9.eE]+\s*$") {
        throw "Calibration file does not contain a valid 3x4 P2 row: $($InputPaths.Calib)"
    }

    $ImageCount = (Get-ChildItem -LiteralPath $InputPaths.ImageDir -Filter *.png -File).Count
    $TimeCount = @(Get-Content -LiteralPath $InputPaths.Times).Count
    if ($RequireCompleteSequence) {
        if ($ImageCount -ne 4541) {
            throw "Sequence 00 image_2 should contain 4541 PNG files, found $ImageCount."
        }
        if ($PoseLines.Count -ne 4541) {
            throw "Sequence 00 pose file should contain 4541 rows, found $($PoseLines.Count)."
        }
        if ($TimeCount -ne 4541) {
            throw "Sequence 00 times.txt should contain 4541 rows, found $TimeCount."
        }
    }

    return [PSCustomObject]@{
        Layout         = $InputPaths.Layout
        ImageCount     = $ImageCount
        PoseRows       = $PoseLines.Count
        TimeRows       = $TimeCount
        VerifiedFrames = $Verified
    }
}
