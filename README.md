# SURF Final Polynomial Release

这是今天收敛用的最小可交付版本。它只保留一条主线：

```text
KITTI image + calib + pose
  -> CLRNet candidates
  -> temporal_independent left/right tracks
  -> metric IPM X/Z
  -> KITTI pose alignment
  -> overlapping 15-frame windows
  -> robust quadratic parametric polynomial X(q), Z(q)
  -> bracketed short-gap interpolation record
  -> common-reference blending and audit
```

本版本暂不使用 Frenet、B-spline 或 StreamMapNet。窗口的直道/过渡/弯道标签仍然保存，
但只作道路形态诊断，不切换模型。`q` 只是点的顺序参数，所有几何输出仍是米制 X/Z。

CLRNet 运行需要额外的 `addict` 依赖；首次配置工作站环境时执行
`python -m pip install -r requirements.txt`。

## 目录作用

- `scripts/scan_lane_tracks.py`：运行 CLRNet，并用 pose 独立维护左右项目轨迹。
- `scripts/fit_polynomial_windows.py`：对每个窗口拟合鲁棒二次参数多项式，记录短缺口和接缝审计。
- `scripts/run_final_polynomial_windows.ps1`：工作站唯一推荐入口。
- `surf_bev/`：IPM、pose 变换和 CLRNet 适配所需的最小运行模块。
- `configs/`：固定实验范围与参数；不放数据集和模型权重。
- `tests/`：不依赖 GPU 的单元测试。

## 工作站运行

在 Anaconda PowerShell Prompt 中进入本目录后执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
& ".\scripts\run_final_polynomial_windows.ps1" `
  -ImageDir "F:\BaiduNetdiskDownload\kitti\odometry\data_odometry_color\dataset\sequences\01\image_2" `
  -Calib "F:\BaiduNetdiskDownload\kitti\odometry\data_odometry_calib\dataset\sequences\01\calib.txt" `
  -Poses "F:\BaiduNetdiskDownload\kitti\odometry\data_odometry_poses\dataset\poses\01.txt" `
  -StartFrame 857 `
  -EndFrame 961 `
  -SequenceId "01" `
  -ClrnetRoot "F:\2026_surf\CLRNet" `
  -Device cuda
```

Sequence 09 的验收范围使用同一入口，将 `-StartFrame 51 -EndFrame 155 -SequenceId 09` 替换即可。

## 输出验收

每次运行使用新的时间戳目录，不覆盖旧结果。重点查看：

- `01_lane_scan/scan.json`：候选数量、左右选择原因、轨迹代数、单侧缺失情况。
- `01_lane_scan/selected_lane_points.json`：所有候选和最终左右槽位。
- `02_polynomial_windows/RESULT.json`：输入哈希、窗口数量、拟合数量、接缝门限和警告。
- `02_polynomial_windows/short_gap_interpolations.csv`：只记录前后均有观测且缺失不超过 3 帧的短缺口。
- `02_polynomial_windows/polynomial_windows_xz_overview.png`：公共 X/Z 坐标中的窗口和融合曲线。

插值点不能冒充 CLRNet 原始观测；没有前后观测支持的长缺口不会被填补。
