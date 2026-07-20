[CmdletBinding()]
param(
    [string]$EnvName = "surf2026-win",
    [ValidateSet("cuda", "cpu")]
    [string]$Backend = "cuda",
    [switch]$SkipWeights,
    [switch]$SkipSmokeTest
)

$ErrorActionPreference = "Stop"
$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ClrnetDir = Join-Path $RootDir "CLRNet"
$PatchPath = Join-Path $RootDir "scripts\patches\clrnet_mac_compat.patch"
$CompatMmcv = Join-Path $RootDir "scripts\clrnet_compat\mmcv"
$WeightDir = Join-Path $ClrnetDir "weights"
$WeightZip = Join-Path $WeightDir "culane_r18.pth.zip"
$WeightPath = Join-Path $WeightDir "culane_r18.pth"
$WeightUrl = "https://github.com/Turoad/CLRNet/releases/download/models/culane_r18.pth.zip"

Set-Location $RootDir

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git was not found in PATH."
}
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found in PATH. Start Anaconda PowerShell Prompt and retry."
}

Write-Host "[1/7] Initializing CLRNet submodule..."
& git submodule update --init --recursive
if ($LASTEXITCODE -ne 0) { throw "CLRNet submodule initialization failed." }

Write-Host "[2/7] Preparing isolated Python 3.9 environment: $EnvName"
$CondaInfo = (& conda env list --json | ConvertFrom-Json)
$Existing = $CondaInfo.envs | Where-Object { (Split-Path $_ -Leaf) -eq $EnvName }
if (-not $Existing) {
    & conda create -y -n $EnvName python=3.9 pip
    if ($LASTEXITCODE -ne 0) { throw "Conda environment creation failed." }
}

Write-Host "[3/7] Installing PyTorch..."
& conda run -n $EnvName python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
if ($Backend -eq "cuda") {
    & conda run -n $EnvName python -m pip install `
        torch==2.5.1 torchvision==0.20.1 `
        --index-url https://download.pytorch.org/whl/cu121
} else {
    & conda run -n $EnvName python -m pip install `
        torch==2.5.1 torchvision==0.20.1 `
        --index-url https://download.pytorch.org/whl/cpu
}
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed." }

Write-Host "[4/7] Installing project dependencies..."
& conda run -n $EnvName python -m pip install -r requirements-windows.txt
if ($LASTEXITCODE -ne 0) { throw "Project dependency installation failed." }

Write-Host "[5/7] Applying CLRNet no-compiler compatibility layer..."
& git -C $ClrnetDir apply --reverse --check $PatchPath 2>$null
if ($LASTEXITCODE -ne 0) {
    & git -C $ClrnetDir apply --check $PatchPath
    if ($LASTEXITCODE -ne 0) { throw "CLRNet compatibility patch cannot be applied cleanly." }
    & git -C $ClrnetDir apply $PatchPath
    if ($LASTEXITCODE -ne 0) { throw "CLRNet compatibility patch failed." }
}

$MmcvTarget = Join-Path $ClrnetDir "mmcv"
New-Item -ItemType Directory -Force -Path $MmcvTarget | Out-Null
Copy-Item -Path (Join-Path $CompatMmcv "*") -Destination $MmcvTarget -Recurse -Force

Write-Host "[6/7] Preparing CLRNet CULane ResNet-18 weights..."
New-Item -ItemType Directory -Force -Path $WeightDir | Out-Null
if ((-not $SkipWeights) -and (-not (Test-Path -LiteralPath $WeightPath))) {
    Invoke-WebRequest -Uri $WeightUrl -OutFile $WeightZip
    Expand-Archive -LiteralPath $WeightZip -DestinationPath $WeightDir -Force
}

Write-Host "[7/7] Checking the Windows environment..."
$Device = if ($Backend -eq "cuda") { "cuda" } else { "cpu" }
if (-not $SkipSmokeTest) {
    & conda run -n $EnvName python scripts\check_windows_env.py --device $Device --run-clrnet
    if ($LASTEXITCODE -ne 0) { throw "Windows CLRNet smoke test failed." }
} else {
    & conda run -n $EnvName python scripts\check_windows_env.py --device $Device
    if ($LASTEXITCODE -ne 0) { throw "Windows environment check failed." }
}

Write-Host ""
Write-Host "Windows setup completed."
Write-Host "Run commands with:"
Write-Host "  conda run -n $EnvName python run_demo.py --help"
Write-Host "GPU demo:"
Write-Host "  conda run -n $EnvName python run_demo.py --mode single --detector clrnet --device $Device --image data/000001_original.jpg --output-dir outputs/windows_smoke"
