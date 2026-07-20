# SURF 2026 Lane BEV Fusion

This project upgrades the original BEV notebook pipeline by replacing the hand-built Hough lane detector with a CLRNet lane detector adapter.

## Why CLRNet

The BEV module needs lane polylines that are stable across frames. CLRNet is the better fit here because it outputs lane-level curves directly: each lane is represented by ordered image points. That is much easier to project into ground coordinates and fuse over time than a row-classification output or raw edge segments.

Ultra-Fast-Lane-Detection is still useful for fast image-space lane demos, but CLRNet gives cleaner structured lane curves for:

- BEV projection
- multi-frame alignment
- lane-map accumulation
- later trajectory or drivable-area logic

## Pipeline

```text
image
  -> CLRNet lane detector
  -> image-space lane polylines
  -> IPM: image points to ground X/Z
  -> optional pose alignment across frames
  -> BEV lane map
```

The old Hough detector is kept as a fallback with `--detector hough`.

## Local Setup

CLRNet is tracked as a git submodule so every teammate uses the same upstream commit.

```text
surf_2026/
  run_demo.py
  surf_bev/
  data/
  CLRNet/                 # git submodule
```

Clone with submodules:

```bash
git clone --recurse-submodules git@github.com:zcy123-rgf/surf_2026.git
cd surf_2026
```

If the repository was already cloned without submodules:

```bash
git submodule update --init --recursive
```

### Windows workstation setup

The Windows workstation path is isolated from the Mac and legacy Linux server
setups. On the tested workstation profile (Python 3.9, RTX 3090, recent NVIDIA
driver), clone the Windows branch from **Anaconda PowerShell Prompt**:

```powershell
Set-Location $HOME
git clone --recurse-submodules `
  --branch agent/windows-workstation `
  --single-branch `
  https://github.com/zcy123-rgf/surf_2026.git `
  surf_2026_windows
Set-Location .\surf_2026_windows
```

Then run the workstation setup:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_clrnet_windows.ps1
```

The script creates an isolated Python 3.10 `surf2026-win` Conda environment (the
workstation's base Python 3.9 installation is left unchanged), installs PyTorch
`2.5.1` with its CUDA 12.1 runtime, initializes CLRNet, downloads the official
CULane ResNet-18 checkpoint, applies the no-compiler NMS/MMCV compatibility
layer, and runs one real CLRNet inference smoke test. The CUDA runtime bundled
with PyTorch is intentionally independent of the larger CUDA capability number
shown by `nvidia-smi`.

After setup, run the project without relying on PowerShell activation state:

```powershell
conda run -n surf2026-win python run_demo.py --help
conda run -n surf2026-win python run_demo.py `
  --mode single `
  --detector clrnet `
  --device cuda `
  --image data\000001_original.jpg `
  --output-dir outputs\windows_smoke
```

Do not run `setup_clrnet_mac.sh`, `server_python.sh`, or the unmodified
`CLRNet/requirements.txt` on Windows. The latter pins legacy binary packages
that are not the deployment contract for this workstation.

### Mac demo setup

For the Mac CPU/MPS demo, run:

```bash
bash scripts/setup_clrnet_mac.sh
```

This script creates `CLRNet/.venv-clrnet-demo`, downloads the official CULane ResNet-18 weight, and applies the small local compatibility patch needed for Mac/no-CUDA inference.

### General Python setup

For the project-level fallback detector and utilities:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Single-Frame Demo

```bash
CLRNet/.venv-clrnet-demo/bin/python run_demo.py \
  --mode single \
  --detector clrnet \
  --image data/000001_original.jpg \
  --output-dir outputs/surf_bev
```

Outputs:

```text
outputs/surf_bev/single_frame_lanes.jpg
outputs/surf_bev/single_frame_bev.jpg
```

If CLRNet is unavailable, run the notebook-style fallback:

```bash
python run_demo.py --mode single --detector hough --image data/000001_original.jpg
```

## Multi-Frame Fusion

For KITTI-style data, provide calibration and pose files:

```bash
python run_demo.py \
  --mode multi \
  --detector clrnet \
  --image-dir /path/to/data_road/testing/image_2 \
  --image-pattern 'um_{frame_id:06d}.png' \
  --calib /path/to/data_road/testing/calib/um_000000.txt \
  --poses /path/to/poses/00.txt \
  --frame-ids 0,1,2 \
  --ref-id 0 \
  --output-dir outputs/surf_bev
```

Output:

```text
outputs/surf_bev/multi_frame_fused_bev.jpg
```

## Data Needed For Real Fusion

The sample images in `data/` are enough to prove that the code path runs, but they are not enough for accurate multi-frame fusion. For metric BEV fusion, prepare:

```text
image_2/
  um_000000.png
  um_000001.png
  um_000002.png
  ...

calib/
  um_000000.txt
  um_000001.txt
  ...

poses/
  00.txt
```

The calibration file must contain KITTI-style `P2`, `R0_rect`, and `Tr_velo_to_cam` lines. The pose file should contain one 3x4 camera pose per frame, as in KITTI odometry.

If you only have images and no odometry poses, the project can still generate single-frame BEV images, but it cannot correctly align lanes from multiple frames into one map.

## Notes

- `calib` and `poses` are required for true multi-frame fusion.
- Without calibration, the single-frame demo uses approximate camera intrinsics so the BEV output is only a runnable visualization, not metric-accurate.
- For production-quality results, fine-tune CLRNet on the project camera/data distribution and tune `camera-height`, `pitch-deg`, `x-range`, and `z-range`.
