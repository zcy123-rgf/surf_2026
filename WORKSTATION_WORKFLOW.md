# Workstation workflow

This is the fixed procedure for changing and running the project in
`F:\surf_complete`.

## 1. Synchronize code

Run from PowerShell:

```powershell
Set-Location F:\surf_complete
git switch complete_code
git pull --ff-only origin complete_code
git status --short --branch
```

The status should show `complete_code...origin/complete_code` and no modified
files before a run.

## 2. Environment

Use the same Conda environment that already passed the tests. Install only the
declared dependencies when needed:

```powershell
conda activate surf2026-win
python -m pip install -r requirements.txt
python -m pytest -q
```

Do not install packages by editing individual scripts. If a dependency is
missing, add it to `requirements.txt`, test locally, commit it, and pull the
new commit on the workstation.

## 3. The only normal edit points

- `configs/final_windows.json`: recorded experiment ranges and window parameters
  for review; the PowerShell command-line arguments are the active run inputs.
- `scripts/analyze_pose_curvature.py`: pose-curvature diagnostic.
- `scripts/fit_polynomial_windows.py`: window fitting and continuity policy.
- `scripts/scan_lane_tracks.py`: CLRNet candidate selection and project-side
  temporal tracks.
- `scripts/run_final_polynomial_windows.ps1`: orchestration only.

Keep `CLRNet/` as the detector dependency. Do not edit generated files under
`workstation_outputs/`, and do not paste one-off scripts into the release
directory. Use a separate temporary directory for experiments that are not
ready to commit.

## 4. Standard core run

The currently validated core interval is Sequence 01 frames `857-961`:

```powershell
Set-Location F:\surf_complete
Set-ExecutionPolicy -Scope Process Bypass
& .\scripts\run_final_polynomial_windows.ps1 `
  -ImageDir "F:\BaiduNetdiskDownload\kitti\odometry\data_odometry_color\dataset\sequences\01\image_2" `
  -Calib "F:\BaiduNetdiskDownload\kitti\odometry\data_odometry_calib\dataset\sequences\01\calib.txt" `
  -Poses "F:\BaiduNetdiskDownload\kitti\odometry\data_odometry_poses\dataset\poses\01.txt" `
  -StartFrame 857 -EndFrame 961 -SequenceId "01" `
  -ClrnetRoot "F:\surf_complete\CLRNet" -Device cuda
```

Every run creates a new timestamped directory with:

```text
01_lane_scan
02_polynomial_windows
03_pose_curvature
04_run_metrics
```

Never overwrite an older run while comparing results.

## 5. All available pose-curvature sequences

Before running CLRNet on every sequence, run the pose-only diagnostic. It
automatically discovers every numeric pose file that exists. KITTI Odometry
has image sequences `00` through `21`, but the official pose ground truth is
normally available only for `00` through `10`. It writes one folder per
available pose sequence plus a combined CSV, JSON, and overview figure:

```powershell
Set-Location F:\surf_complete
python .\scripts\run_all_pose_curvature.py `
  --poses-root "F:\BaiduNetdiskDownload\kitti\odometry\data_odometry_poses\dataset\poses" `
  --output-root ".\workstation_outputs\all_pose_curvature_$(Get-Date -Format yyyyMMdd_HHmmss)" `
  --smoothing-window 11
```

This step needs only the pose files; it does not load images or CLRNet. Review
`all_sequences_summary.csv` and `all_sequences_curvature_overview.png` before
choosing turning intervals for the more expensive lane-detection pipeline.

## 6. Extension audit

The proposed wider Sequence 01 range is `851-1005`. Run it only as an audit:

```powershell
... -StartFrame 851 -EndFrame 1005 -SequenceId "01" ...
```

Do not call this range validated until `scan.json` confirms both-side
coverage, the curvature plot shows the intended road section, and
`RESULT.json`/`METRICS.json` show acceptable continuity. Missing turning-frame
lanes must remain missing; the scanner must not invent them.

## 7. After a code change

```powershell
python -m pytest -q
git diff --check
git status --short
git add -- <only-the-files-you-changed>
git commit -m "Describe the behavioral change"
git push origin complete_code
```

Then return to the workstation and repeat step 1 before running again.

