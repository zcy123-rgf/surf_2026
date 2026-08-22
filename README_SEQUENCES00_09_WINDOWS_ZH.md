# Windows 工作站：KITTI Odometry 00–09 独立自动实验

## 数据与边界

脚本使用 KITTI Odometry 的 `00、01、02、03、04、05、06、07、08、09` 十个序列。每个序列独立读取：

- `image_2` 彩色图像；
- 本序列 `calib.txt`；
- 本序列 pose 文件；
- 本序列 `times.txt`。

绝不把不同 sequence 的帧或 pose 拼接。工作站数据完整时，00–09 合计约 22000 帧，但每个 sequence 都从自己的第 0 帧开始编号。

## 一键运行全部序列

在 Anaconda PowerShell Prompt 中执行：

```powershell
Set-Location F:\2026_surf\clean_workstation_release_20260804_004638
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

& ".\scripts\run_sequences00_09_curve_models_windows.ps1" `
  -EnvName "surf2026-win" `
  -DatasetRoot "F:\BaiduNetdiskDownload\kitti\odometry" `
  -Device cuda `
  -TopCandidateCount 100 `
  -SelectedRank 1 `
  -MinimumValidFramesPerBlock 10 `
  -CandidateSelectionMode "ego_adjacent" `
  -Model both
```

流程先一次性核对十个序列的图像、标定、pose 和时间戳行数，再检查一次 CUDA/CLRNet，随后依次运行 00–09。数据不完整时会在耗时检测开始前停止；算法运行中的单序列失败会记录原因并继续下一个序列。

默认候选排序使用 `sustained_curve`：先要求每个15帧窗口至少10帧有效，再依据实际位置轨迹中持续发生转向的窗口数量寻找弯道。相机短时调整朝向不再直接当作道路持续弯曲。

默认候选配对使用 `ego_adjacent`：以本序列 `P2` 标定矩阵的主点 `cx` 为车辆图像中心，在每帧分别选择中心左侧最近和右侧最近的 CLRNet 候选。候选超过两条时不会再取最外侧两条；任一侧没有候选时，该帧记为无效，不补造车道线。`outermost` 仍保留为旧结果复现实验：

```powershell
-CandidateSelectionMode "outermost"
```

几何配对不能证明候选一定是白色车道标线，因此仍需查看 `original_frames`、`all_candidates` 和新增的 `selected_pairs`。`selected_pairs` 中黄色竖线是标定主点，蓝/红点分别是最终进入 IPM 的左/右候选。

## 分序列输出

每次运行创建新的批次目录：

```text
workstation_outputs\sequences00_09_independent_时间戳\
  sequence_00\
  sequence_01\
  ...
  sequence_09\
  BATCH_SUMMARY.csv
  BATCH_STATUS.json
  DATASET_INVENTORY.csv
```

每个 `sequence_XX` 内部独立包含：

```text
00_scan_frames          CLRNet逐帧候选、选中候选审计、原图和点式IPM结果
01_ranked_option        本序列候选路段及选中的连续105帧
02_polynomial_only      本序列多项式曲线结果
03_bspline_only         本序列B样条曲线结果
RUN_STATUS.json         本序列运行状态与真实帧数
```

所有输出都进入新目录，不覆盖以前的 `output` 或 `workstation_outputs`。

## 只跑或重跑某几个序列

例如只运行 02、05、08：

```powershell
& ".\scripts\run_sequences00_09_curve_models_windows.ps1" `
  -EnvName "surf2026-win" `
  -DatasetRoot "F:\BaiduNetdiskDownload\kitti\odometry" `
  -Device cuda `
  -SequenceIds "02","05","08" `
  -CandidateSelectionMode "ego_adjacent" `
  -Model both
```

## 注意

- 本流程沿用已经验证过的 CLRNet、点式 IPM、pose 对齐、多项式和 B 样条代码，不改变数学方法；
- `-Model both` 使用完全相同的筛选后点集分别运行多项式与 B 样条，保证两组实验输入一致；
- 候选排名综合检测覆盖率与位置轨迹弯曲程度，不等于真实车道准确率；
- 十个完整序列会产生大量图片副本和候选图，运行前建议为新输出预留至少 40 GB 空间；
- 需要暂停时可按 `Ctrl+C`。已经完成的 sequence 目录会保留，但再次运行应创建新批次，或只指定尚未完成的序列。
