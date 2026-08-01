# Windows 工作站数据集入口

本分支不携带 KITTI 原图、标定、位姿或历史运行结果。默认从工作站已有的
KITTI Odometry 官方分包读取：

```text
F:\BaiduNetdiskDownload\kitti\odometry
├─ data_odometry_color\dataset\sequences\00\image_2
├─ data_odometry_calib\dataset\sequences\00\calib.txt
├─ data_odometry_poses\dataset\poses\00.txt
└─ data_odometry_color\dataset\sequences\00\times.txt
```

运行入口只选取 Sequence 00 左彩色相机的前五帧
`000000.png` 至 `000004.png`。开始调用 CLRNet 前会检查：

1. 五张图的 SHA256 是否与已核验的官方帧一致；
2. `calib.txt` 是否包含有效的 `P2` 投影矩阵；
3. 位姿前五行是否各有 12 个数；
4. 图像、位姿和时间戳是否都各有 4541 项。

人工标注位于：

```text
annotations\kitti00_first5_manual_annotations.json
```

它只是项目内部的左右道路边界伪标注，用于方法对比，不能称为 KITTI
官方车道线真值。

## 运行完整点融合

```powershell
Set-Location F:\2026_surf
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run_kitti00_first5_windows.ps1
```

## 运行 RANSAC 对比

```powershell
Set-Location F:\2026_surf
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run_ransac_improvement_windows.ps1 -Mode Compare
```

如数据集以后移动，只需要传入新根目录：

```powershell
.\scripts\run_ransac_improvement_windows.ps1 `
  -DatasetRoot "F:\新的位置\odometry" `
  -Mode Compare
```

每次运行都写入新的 `workstation_outputs` 时间戳目录；脚本不读取
`results` 或既有 `workstation_outputs`。

## 精确点集输出

主流程在 `01_from_scratch_pipeline\00_metadata` 写出：

```text
denoised_point_sets.json
denoising_point_decisions.csv
```

RANSAC比较流程在 `02_ransac_method_comparison\00_audit` 写出：

```text
denoised_point_sets.json
point_decisions.csv
```

JSON按方法、帧和左右侧分别保存保留点与剔除点的米制 `(X,Z)` 坐标；
CSV每个源点一行，并用0/1列记录各方法的保留决定。各方法文件夹也分别
包含 `kept_points_xz.csv` 和 `rejected_points_xz.csv`。这些文件直接来自
用于绘制PNG的内存点集，不从PNG反推坐标。
