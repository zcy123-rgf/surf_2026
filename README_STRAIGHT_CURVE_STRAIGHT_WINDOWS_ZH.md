# 直道—弯道—直道连续融合实验（Windows）

## 结论先行

CLRNet没有输出跨帧稳定的官方车道ID。模型输出的是每张图中的车道候选，候选序号经过置信度筛选和NMS后可能变化。本实验在CLRNet外层创建临时的 `ego_left_*`、`ego_right_*` 跟踪ID：利用KITTI pose把上一帧的车道点变换到当前帧，再按米制曲线距离匹配。

这个ID只表示“本次连续实验中关联到的同一条候选轨迹”，不是KITTI地图车道ID，也不能仅凭连续性证明候选不是人行道边缘。

## 推荐实验

默认使用Sequence 02，从完整序列扫描结果中自动挑选105帧连续路段：

- 前两个15帧窗口接近直道；
- 中间窗口至少两个达到弯道阈值；
- 最后两个窗口重新接近直道；
- 每个窗口至少10帧具有可用的左右候选。

选择路段后只对这105帧重新运行CLRNet和时序关联，因此不会重跑整个Sequence，也不会修改过去的output。

```powershell
Set-Location "F:\2026_surf\clean_workstation_release_20260804_004638"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

& ".\scripts\run_straight_curve_straight_windows.ps1" `
  -EnvName "surf2026-win" `
  -DatasetRoot "F:\BaiduNetdiskDownload\kitti\odometry" `
  -ExistingBatchRoot "F:\2026_surf\workstation_outputs\sequences01_09_ego_adjacent_20260809_141900" `
  -SequenceId "02" `
  -WindowCount 10 `
  -ClrnetRoot "F:\2026_surf\CLRNet" `
  -Device cuda
```

`CLRNet`作为外部模型目录只被调用，不会复制到本次实验结果，也不会由脚本修改。

## 输出

每次运行创建新的时间戳目录：

- `00_pose_and_coverage_selection`：从整段Sequence中找“直—弯—直”候选；
- `01_temporal_lane_tracks`：CLRNet候选、逐帧选择、项目track ID和距离门限；
- `02_tracked_window_selection`：用新关联点重新检查每个窗口覆盖率；
- `03_polynomial`：多项式结果；
- `04_bspline`：B样条结果；
- `05_piecewise_fusion`：15帧局部拟合、重叠窗口连续融合；
- `06_meeting_figures`：组会可直接检查的入口、弯道、出口与模型图；
- `STATUS.json`：帧范围、曲率阶段、有效关联帧数和诚实警告。

## 为什么不直接把整个Sequence拟合成两条线

完整Sequence可能包含路口、换道、车道增减、环路和遮挡。同一辆车左、右相邻的边界会随道路拓扑变化，因此“整段Sequence始终只有同两个ID”这个假设通常不成立。完整序列可以扫描和分段建图，但应先按连续性门限切成若干道路段，再分别保存track；不能强迫一条多项式或一条B样条覆盖整段Sequence。
