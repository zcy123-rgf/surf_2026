# SURF 最终工作站流程

## 1. 本阶段的真实范围

当前完成的是一条可复查的车道几何原型链路：

`KITTI 图像 -> CLRNet 候选车道 -> 左右观测关联 -> IPM 地面 X/Z 点 -> KITTI pose 对齐 -> 位姿曲率分段 -> 短窗口曲线拟合 -> 重叠窗口融合 -> 遮挡缺口审计 -> 一致性评估`

本阶段不声称已经完成通用 SLAM、完整 4D 语义地图或真实车道位置精度验证。KITTI Odometry 没有与相机帧逐帧对应的左右车道边界真值。

## 2. 文件结构

- `CLRNet/`：单独保存的 Windows CLRNet 运行时和权重。最终脚本只调用它，不修改它。
- `surf_bev/`：CLRNet 调用、IPM 和位姿坐标变换。
- `scripts/run_surf_final_window_windows.ps1`：最终长路段一键入口。
- `scripts/analyze_pose_curvature.py`：由 KITTI pose 计算曲率和直道/过渡/弯道状态。
- `scripts/run_pose_curvature_all_sequences_windows.ps1`：只读 pose 的 Sequence 00--10 轻量曲率筛查。
- `scripts/fit_adaptive_xz_piecewise.py`：分窗口拟合、模型比较、连续性检查和融合。
- `scripts/bridge_occluded_lane_segments.py`：只对通过门限的缺口输出低置信度假设桥。
- `scripts/run_ransac_improvement_windows.ps1`：保留的最初 0--4 帧改进 RANSAC 对比基线，不是长弯道主入口。
- `workstation_outputs/`：每次运行新建带时间戳的结果；程序拒绝覆盖非空目录。
- `SOURCE_MANIFEST.json`：本工作区来源和 CLRNet 版本记录。

## 3. 各阶段的输入、输出和结论边界

### 3.1 CLRNet 与左右观测

输入是 KITTI 相机图像。CLRNet 每帧输出若干候选车道曲线；程序将曲线采样为二维图像点。CLRNet 的候选序号和项目生成的 track ID 都不是跨帧语义车道 ID。

最终批处理暂用 `temporal_independent`：先从自车左右候选初始化，再利用 pose 把上一帧观测预测到当前帧，左右独立关联。它能让一侧在短时缺失时不拖累另一侧，但仍是项目假设，不能保证排除路沿、人行道或换道后的错误关联。

### 3.2 IPM 与 pose 对齐

图像点 `(u,v)` 通过相机内参和项目登记的平坦地面假设变成米制地面点 `(X,Z)`。之后读取 KITTI pose 文件中每帧的 `3x4` 变换，把各窗口的点变换到同一参考相机坐标。保存和拟合的几何始终是笛卡尔 X/Z；程序没有使用 Frenet 坐标系。

### 3.3 位姿曲率与明确阈值

曲率定义为单位路程的航向变化，单位 `1/m`。程序先按弧长均匀重采样 pose 轨迹，再用 15 m Savitzky--Golay 窗口抑制逐帧抖动。弧长只用于求导，输出坐标仍为 X/Z。

没有适用于所有 KITTI 路段的论文统一数值阈值，因此这里把阈值明确登记为“项目操作阈值”，不冒充数据集标注：

- 直道基础下限：`0.0015 1/m`；
- 弯道基础下限：`0.0030 1/m`；
- 若提供已核验直道种子，直道阈值取 `max(0.0015, 中位数 + 3×稳健标准差)`；
- 弯道阈值取 `max(0.0030, 中位数 + 6×稳健标准差, 1.5×直道阈值)`；
- 使用高低双阈值和至少连续 5 帧确认，减少在边界反复跳变。

这套门限的依据是“直线曲率接近 0、圆曲线曲率近似常数、缓和曲线曲率逐渐变化”的道路几何关系，再用本段直道的 pose 噪声确定可复现的数值。若不提供直道种子，程序使用最低曲率四分位作为后备基线，并在结果中明确标注，可信度低于人工核验过的直道种子。

### 3.4 多项式、B 样条与窗口接口

每个窗口默认 15 帧、步长 10 帧，左右分别处理：

- 直道窗口固定使用二次参数多项式；
- 过渡和弯道窗口在同一输入上比较多项式与三次参数 B 样条；
- 比较指标是留出整帧后的 RMSE；差异小于登记的简化容差时优先多项式；
- 15 帧窗口有 5 帧重叠。相邻窗口同时检查重叠区 P95 距离和切向夹角；通过后才加权融合，失败则断开并保留原因。

这相当于检查 G1 兼容性（位置接近、方向接近），但不是用一个全局方程强制整条路处处严格 G1。当前方案可以作为多项式/B 样条接口；其优点是局部失败不会污染整条路。

### 3.5 遮挡桥接

缺少观测的部分不能画成实测车道。程序只在遮挡前后都有足够多节点，并且端点距离、两端切向、桥接曲率及可用时的车道宽度都通过门限时，生成三次 Hermite 桥。桥的两端方向来自各自一段曲线的多个邻域点，不是拿两个孤立检测点直接相连。

所有桥均写为 `hypothesis`、低置信度并用虚线显示；门限失败时保留空缺。这借鉴了 StreamMapNet 的“位姿补偿历史地图信息”思想，但没有训练或声称复现 StreamMapNet。

### 3.6 评估

由于没有官方左右车道真值，报告分成三层：

1. 观测覆盖：左右各自有效帧数、双侧同时有效帧数、最长缺失段；
2. 留出一致性：按整帧留出，报告 Mean、RMSE、Median、P90/P95 和 Max；左对左、右对右；
3. 几何安全检查：窗口接口距离/方向、输出断点、车道宽度、桥接曲率和桥接置信度。

这些指标能比较方法的稳定性和保留能力，不能称为真实世界准确率。早期像素重叠率只保留作历史诊断，不作为最终主指标。

## 4. 在工作站运行

在 Anaconda PowerShell Prompt 中：

```powershell
Set-Location F:\surf_final
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

先做 Sequence 00--10 的轻量 pose 曲率筛查（不运行 CLRNet）：

```powershell
& .\scripts\run_pose_curvature_all_sequences_windows.ps1 `
  -EnvName "surf2026-win" `
  -DatasetRoot "F:\BaiduNetdiskDownload\kitti\odometry"
```

该批量输出使用最低曲率四分位后备基线，只用于发现候选路段。重点实验仍按下面的直道种子重新校准。

先运行重点路段 Sequence 01，帧 851--1005：

```powershell
& .\scripts\run_surf_final_window_windows.ps1 `
  -EnvName "surf2026-win" `
  -DatasetRoot "F:\BaiduNetdiskDownload\kitti\odometry" `
  -SequenceId "01" `
  -StartFrame 851 `
  -EndFrame 1005 `
  -StraightSeedRanges "851-875,991-1005" `
  -Device cuda
```

再运行 Sequence 07 候选弯道，帧 415--514：

```powershell
& .\scripts\run_surf_final_window_windows.ps1 `
  -EnvName "surf2026-win" `
  -DatasetRoot "F:\BaiduNetdiskDownload\kitti\odometry" `
  -SequenceId "07" `
  -StartFrame 415 `
  -EndFrame 514 `
  -StraightSeedRanges "415-435,496-514" `
  -Device cuda
```

Sequence 03 帧 29--128 只作为备用候选。未核验直道种子前不要把后备四分位阈值写成最终结论。

每次完成后先看：

- `FINAL_STATUS.json`：总状态和限制；
- `02_pose_curvature/pose_curvature_overview.png`：曲率分段；
- `03_curve_models_and_fusion/adaptive_piecewise_xz_overview.png`：模型、融合和留出误差；
- `03_curve_models_and_fusion/skipped_items.csv`：为何跳过某侧窗口；
- `04_occlusion_hypotheses/occlusion_bridge_overview.png`：实线观测与虚线假设；
- 根目录下的 `final_seq...review_bundle.zip`：用于上传核验的小包。

## 5. 复现实验时必须保留的表述

- 车道候选来自 CLRNet；跨帧 track ID 是项目生成的，不是 CLRNet 固定语义 ID。
- 曲率阈值是依据直道噪声校准的项目阈值，不是 KITTI 官方阈值。
- 留出误差是内部一致性，不是真值准确率。
- 遮挡桥是低置信度几何假设，不是观测。
- 实际 SLAM、动态目标时序语义和完整 4D 地图不在本阶段交付范围内。

## 6. 方法依据

- StreamMapNet（WACV 2024）：历史 map query 经位姿变换后传播，并结合 BEV temporal fusion；本项目只借鉴“位姿补偿历史几何”思想。<https://openaccess.thecvf.com/content/WACV2024/papers/Yuan_StreamMapNet_Streaming_Mapping_Network_for_Vectorized_Online_HD_Map_Construction_WACV_2024_paper.pdf>
- B 样条接口研究：相邻曲线若要方向连续，应同时约束连接位置与端点切向，即 G1 连续。<https://academic.oup.com/jcde/article/2/4/218/5715267>
- 道路几何曲率：直线、圆曲线与缓和曲线可通过曲率为零、近似常数和逐渐变化来区分。<https://www.mdpi.com/1424-8220/19/24/5373>
