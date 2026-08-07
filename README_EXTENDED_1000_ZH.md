# Windows 工作站：0–1000 帧候选扫描与分离曲线实验

本流程用于扩大候选图片范围。默认扫描 KITTI Odometry Sequence 00 的第 `0` 帧到第 `1000` 帧，首尾都包含，因此实际检查 `1001` 张图片。

扫描 `0–1000` 的目的，是提供更多同一路段候选窗口；程序不会把这 1001 帧跨越的所有道路强行拟合成一条车道线。最终只对一个通过覆盖率检查的候选连续路段分别运行多项式和 B 样条。

## 1. 一键运行

在 Anaconda PowerShell Prompt 中执行：

```powershell
Set-Location F:\2026_surf\clean_workstation_release_20260804_004638
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

& ".\scripts\run_extended_1000_curve_models_windows.ps1" `
  -EnvName "surf2026-win" `
  -DatasetRoot "F:\BaiduNetdiskDownload\kitti\odometry" `
  -Device cuda `
  -StartFrame 0 `
  -EndFrame 1000 `
  -TopCandidateCount 100 `
  -SelectedRank 1 `
  -Model both
```

`EndFrame` 是包含端点的。需要继续扩大到第 1200 帧时，将其改成：

```powershell
-EndFrame 1200
```

## 2. 自动流程

总控脚本依次执行：

```text
0–1000 原图
  -> CLRNet逐帧检测与点式IPM
  -> 保存每帧候选数和左右候选点
  -> 滑动检查10个15帧窗口（步长10，相邻重叠5帧）
  -> 输出最多100个候选路段
  -> 选择SelectedRank指定的候选
  -> 独立运行鲁棒多项式
  -> 独立运行鲁棒三次B样条
```

候选排序优先考虑：

1. 十个窗口是否都达到最低有效帧数；
2. 最弱窗口中有效帧的数量；
3. 十个窗口的总有效帧使用次数；
4. pose 显示的道路转向变化。

该排序只用于寻找可实验路段，不等于车道线准确率。

## 3. 多项式与 B 样条已分开

多项式主体：

```text
scripts\fit_extended_polynomial.py
```

它只比较和拟合鲁棒 `1/2/3` 次 Frenet 多项式 `d=f(s)`，不会调用 B 样条拟合。

B 样条主体：

```text
scripts\fit_extended_bspline.py
```

它只选择平滑参数并拟合鲁棒三次 Frenet B 样条，不调用多项式拟合。

两者共享的数据准备代码：

```text
scripts\extended_curve_model_common.py
```

共享部分只负责读取候选、读取官方 pose/标定、调用已经审核过的 pose 对齐和 Frenet 转换。数学模型仍然位于两个独立脚本中。

## 4. 输出目录

每次运行自动创建：

```text
workstation_outputs\extended_curve_000000_001000_时间戳
```

主要内容：

- `00_scan_frames`：1001帧扫描记录、原图、候选点图、`scan.json` 和 `selected_lane_points.json`；
- `01_ranked_option\candidate_options.csv`：最多100个候选路段及其排名；
- `01_ranked_option\selection.json`：本次实际选择的路段；
- `02_polynomial_only`：多项式独立结果、曲线坐标、交叉验证和图片；
- `03_bspline_only`：B样条独立结果、曲线坐标、平滑参数评价和图片；
- `RUN_STATUS.json`：总运行状态、搜索范围、候选数量和各结果位置。

所有输出均为新目录，不会覆盖以前的 `output` 或 `workstation_outputs`。

## 5. 选择其他候选而不重新扫描1001帧

先打开：

```text
01_ranked_option\candidate_options.csv
```

确认想尝试的 `rank`。假设要运行第 7 名，并复用已经完成的扫描：

```powershell
& ".\scripts\run_extended_1000_curve_models_windows.ps1" `
  -EnvName "surf2026-win" `
  -DatasetRoot "F:\BaiduNetdiskDownload\kitti\odometry" `
  -StartFrame 0 `
  -EndFrame 1000 `
  -TopCandidateCount 100 `
  -SelectedRank 7 `
  -ExistingScanJson "上一轮输出\00_scan_frames\scan.json" `
  -Model both
```

这样只重新选择候选并拟合曲线，不重复运行 CLRNet。

## 6. 只运行一种模型

只运行多项式：

```powershell
-Model polynomial
```

只运行 B 样条：

```powershell
-Model bspline
```

两种都运行：

```powershell
-Model both
```

## 7. 严格限制

- 相机高、俯仰角、IPM、pose 与 Frenet 坐标转换沿用上一版，不在本任务中修改；
- 最左/最右 CLRNet 候选仍只是当前项目的候选选择规则，不保证具有本车道左右边界语义；
- 如果选中路段未通过每个窗口的覆盖率门槛，程序停止并保留候选表，不生成虚假的曲线结果；
- 留出帧 RMSE 衡量的是 CLRNet 派生点的一致性，不是官方车道线真值准确率；
- `selected_lane_points.json` 保存的是本次实际扫描结果，不使用以前的图片结果代替。

