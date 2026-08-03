[CmdletBinding()]
param(
    [string]$Destination = "F:\2026_surf\clean_workstation_release",
    [string]$ClrnetSource = "",
    [switch]$CopyClrnet
)

$ErrorActionPreference = "Stop"
$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if (Test-Path -LiteralPath $Destination) {
    if (Get-ChildItem -LiteralPath $Destination -Force) {
        throw "Destination must be new or empty: $Destination"
    }
} else {
    New-Item -ItemType Directory -Path $Destination | Out-Null
}

$Files = @(
    ".gitignore",
    "requirements.txt",
    "requirements-windows.txt",
    "run_demo.py"
)
$Directories = @(
    "annotations",
    "data",
    "surf_bev",
    "workstation_release"
)
$ScriptFiles = @(
    "check_windows_env.py",
    "kitti00_workstation_input.ps1",
    "run_full_point_pipeline.py",
    "evaluate_denoise_methods.py",
    "run_selected_ransac_reference.py",
    "fit_first5_two_curves.py",
    "analyze_lane_curve_hierarchy.py",
    "compare_polynomial_bspline.py",
    "setup_clrnet_windows.ps1"
)
$ScriptDirectories = @("clrnet_compat", "patches")

foreach ($Relative in $Files) {
    $Source = Join-Path $RootDir $Relative
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        throw "Required release file is missing: $Source"
    }
    $Target = Join-Path $Destination $Relative
    $Parent = Split-Path -Parent $Target
    New-Item -ItemType Directory -Path $Parent -Force | Out-Null
    Copy-Item -LiteralPath $Source -Destination $Target
}
foreach ($Relative in $Directories) {
    $Source = Join-Path $RootDir $Relative
    if (-not (Test-Path -LiteralPath $Source -PathType Container)) {
        throw "Required release directory is missing: $Source"
    }
    Copy-Item -LiteralPath $Source -Destination $Destination -Recurse
}

$TargetScripts = Join-Path $Destination "scripts"
New-Item -ItemType Directory -Path $TargetScripts -Force | Out-Null
foreach ($Name in $ScriptFiles) {
    $Source = Join-Path $RootDir "scripts\$Name"
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        throw "Required release script is missing: $Source"
    }
    Copy-Item -LiteralPath $Source -Destination (Join-Path $TargetScripts $Name)
}
foreach ($Name in $ScriptDirectories) {
    $Source = Join-Path $RootDir "scripts\$Name"
    Copy-Item -LiteralPath $Source -Destination $TargetScripts -Recurse
}

if ([string]::IsNullOrWhiteSpace($ClrnetSource)) {
    $ClrnetSource = Join-Path $RootDir "CLRNet"
} else {
    $ClrnetSource = (Resolve-Path -LiteralPath $ClrnetSource).Path
}
if (-not (Test-Path -LiteralPath (Join-Path $ClrnetSource "clrnet") -PathType Container)) {
    throw "The working CLRNet directory is incomplete: $ClrnetSource"
}
$ClrnetTarget = Join-Path $Destination "CLRNet"
if ($CopyClrnet) {
    Write-Host "Copying CLRNet, compatibility files and local weights..."
    Copy-Item -LiteralPath $ClrnetSource -Destination $ClrnetTarget -Recurse
} else {
    Write-Host "Creating a junction to the already working CLRNet directory..."
    New-Item -ItemType Junction -Path $ClrnetTarget -Target $ClrnetSource | Out-Null
}

@{
    status = "complete"
    source_root = $RootDir
    destination = (Resolve-Path -LiteralPath $Destination).Path
    clrnet_mode = $(if ($CopyClrnet) { "copied" } else { "junction" })
    default_method = "RANSAC reference"
    optional_method = "polynomial/B-spline comparison"
    excluded_from_default = @(
        "PPT generators",
        "SemanticKITTI audit experiments",
        "candidate segment search",
        "short-segment sparse hierarchical refusion",
        "old result directories"
    )
} | ConvertTo-Json -Depth 5 | Set-Content `
    -LiteralPath (Join-Path $Destination "RELEASE_STATUS.json") -Encoding UTF8

Write-Host ""
Write-Host "CLEAN WORKSTATION COPY CREATED"
Write-Host "Destination: $Destination"
Write-Host "Default: $Destination\workstation_release\01_ransac_reference\run_windows.ps1"
Write-Host "Optional: $Destination\workstation_release\02_curve_models\run_windows.ps1"
