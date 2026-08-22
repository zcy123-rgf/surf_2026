# Sequence 01 自适应分段 X/Z 曲线实验

这套实验是新增入口，不修改原有算法文件，也不覆盖已有输出。输入是已经保存的
CLRNet 左右候选点、KITTI 官方 pose 和 calib；全部几何结果始终保存在米制笛卡尔
`X/Z` 中。

## 方法

1. 以 15 帧为一个窗口、步长 10 帧；末尾不足一个步长时补一个贴住终点的完整窗口。
2. 每个窗口先用 KITTI pose 对齐；左右观测始终分开。每侧默认至少需要 8/15 帧有效，且最长连续缺失不能超过 3 帧。
3. 用窗口前段和后段轨迹基线计算稳健净转角：`<=1.5°` 为直道，`>=3°` 为弯道，中间为过渡段；逐步转角绝对值之和只作为诊断，不参与分类。
4. 直道使用鲁棒二次参数多项式 `X(q), Z(q)`，弯道使用鲁棒参数三次 B 样条 `X(u), Z(u)`；过渡段在相同留出帧上比较两种模型，RMSE 相差不超过 0.005 m 时选更简单的多项式，否则选 RMSE 更小者。
5. 聚合时只借助局部 pose 折线的累计进度排列急弯中的点；`q/u` 也只表示曲线上的采样顺序，不替代 `X/Z` 坐标。
6. 每个窗口的曲线采样点转换到同一个参考帧，再对通过连续性门限的重叠部分取稳健中值；无重叠、P95 间隙过大、切向不连续或内部支持中断时新建 `segment_id`，不跨缺口画线。
7. 每个可拟合窗口同时计算多项式和 B 样条的留出帧一致性，便于在完全相同输入上比较。

## 在工作站运行 Sequence 01 帧 851–1005

本次范围定义为：`851–990` 是已经人工筛选的核心“入口直道—长弯—出弯过渡”段，
`991–1005` 是用于补足并检查出弯后稳定直道的 15 帧出口延伸段。两者在结果元数据中分开记录。

在 Anaconda PowerShell Prompt 中进入工程目录后，直接运行一键入口。它会从工作站
KITTI Sequence 01 原图重新执行 CLRNet、左右独立跟踪和 X/Z 分窗拟合：

```powershell
Set-Location F:\2026_surf\clean_workstation_release_你的时间戳
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

& ".\scripts\run_sequence01_851_1005_xz_windows.ps1" `
  -EnvName "surf2026-win" `
  -DatasetRoot "F:\BaiduNetdiskDownload\kitti\odometry" `
  -ClrnetRoot "F:\2026_surf\CLRNet" `
  -Device cuda
```

如已经有同一次扫描生成的 `selected_lane_points.json`，才使用下面的二阶段入口，
以免重复运行 CLRNet：

```powershell
Set-Location F:\2026_surf\clean_workstation_release_你的时间戳

$Points = "填入Sequence01扫描结果\selected_lane_points.json"
$Scan = "填入同一次扫描结果\scan.json"
$Poses = "F:\BaiduNetdiskDownload\kitti\odometry\data_odometry_poses\dataset\poses\01.txt"
$Calib = "F:\BaiduNetdiskDownload\kitti\odometry\data_odometry_calib\dataset\sequences\01\calib.txt"

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& ".\scripts\run_adaptive_xz_piecewise_windows.ps1" `
  -EnvName "surf2026-win" `
  -SelectedLanePoints $Points `
  -ScanJson $Scan `
  -Poses $Poses `
  -Calib $Calib `
  -SequenceId "01" `
  -StartFrame 851 `
  -EndFrame 1005 `
  -CoreEndFrame 990 `
  -WindowLength 15 `
  -WindowStride 10 `
  -StraightMaxHeadingDeg 1.5 `
  -CurveMinHeadingDeg 3.0 `
  -PolynomialDegree 2
```

脚本自动创建带时间戳的新目录，默认位于 `workstation_outputs` 下。

## 主要输出

- `adaptive_piecewise_xz_overview.png`：轨迹、各窗口模型、融合后左右曲线和模型一致性概览。
- `window_plan.csv`：每个窗口的帧范围、航向变化、直道/过渡/弯道分类和所选模型。
- `model_comparison.csv`：同一窗口上多项式与 B 样条的训练残差和留出帧一致性。
- `window_continuity.csv`：相邻窗口重叠区的均值、P95、最大间隙及切向夹角。
- `window_curve_samples.csv`：每个窗口变换到公共参考帧后的曲线采样点。
- `blended_lane_nodes.csv`：最终左右曲线节点坐标；必须按 `side + segment_id` 分段绘制。
- `output_segments.csv`：每段输出对应的左右侧、窗口范围、帧范围和节点数。
- `RESULT.json`：参数、输入哈希、坐标定义、数量统计、连续性诊断和限制。
- `STATUS.json`：运行状态及是否存在跳过窗口。

这些误差衡量的是结果与留出 CLRNet/IPM 点的一致程度，不代表官方真实车道位置精度。
