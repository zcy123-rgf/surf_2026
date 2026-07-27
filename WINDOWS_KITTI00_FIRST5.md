# Windows 工作站：KITTI Odometry Sequence 00 前五帧位姿融合

这套入口从仓库中已经核验过的官方数据重新生成结果，不读取旧的融合图片：

- 数据：KITTI Odometry `Sequence 00`，帧 `000000～000004`；
- 图像：左彩色相机 `image_2`；
- 位姿：KITTI Odometry `poses/00.txt` 的前五行；
- 参考帧：`000004`；
- 检测：CLRNet CULane ResNet-18；
- 几何：有序图像点 → 米制地面点 → 位姿变换到参考帧；
- 输出：逐帧结果、米制位姿融合图、未去噪融合、旧 RANSAC 诊断和审计数据。

## 1. 为什么这次能明确证明使用了位姿

脚本不再只保存裁剪后的 `800×800` BEV。它另外保存：

```text
04_pose_aligned_points/five_frame_metric_pose_fusion.png
00_metadata/pose_alignment.csv
00_metadata/aligned_lane_points.json
00_metadata/audit.json
```

米制图按帧着色，叉号是各帧相机原点。`pose_alignment.csv` 给出每帧相对于参考帧的原点和平移距离；`audit.json` 保存每个 4×4 相对位姿矩阵。

Sequence 00 的前五帧是真正的 10 Hz 连续帧，车辆每帧约前进 `0.86 m`，所以五个相机原点只相差约 `3.43 m`。它不会像以前的 KITTI Road `um_000012～um_000016` 关键样本图那样延伸约 `80 m`；这是数据本身的区别，不是漏用了位姿。

## 2. 坐标和标定口径

KITTI Odometry 位姿文件给出 rectified camera 0 的世界位姿，但当前图像来自 `image_2`。脚本读取 `P2` 的第四列，先计算固定的 `camera 0 ↔ image camera` 平移，再形成图像相机位姿，之后计算：

```text
T_reference_from_source = inverse(T_world_reference) @ T_world_source
```

每帧 CLRNet 点先在本帧相机/道路模型下得到 `(X,Z)`，再使用上述相对位姿变换到第 `000004` 帧。

相机高度 `1.65 m`、俯仰角 `0°` 和平坦路面仍是组内实验假设，不会在报告中冒充 KITTI 官方逐帧路面外参。

## 3. 在工作站运行

打开 **Anaconda PowerShell Prompt**：

```powershell
Set-Location F:\2026_surf
git switch agent/windows-workstation
git pull --ff-only origin agent/windows-workstation
git submodule update --init --recursive
```

如果 `surf2026-win` 环境和 CLRNet 单图测试已经成功，不需要再次安装环境。直接运行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run_kitti00_first5_windows.ps1
```

默认新结果目录：

```text
F:\2026_surf\workstation_outputs\kitti00_seq00_first5_pose_fusion
```

脚本要求输出目录为空，防止把旧图片当成新结果。如果默认目录已经有结果，请指定一个新目录：

```powershell
.\scripts\run_kitti00_first5_windows.ps1 `
  -OutputDir F:\2026_surf\workstation_outputs\kitti00_seq00_first5_pose_fusion_run2
```

## 4. 输出目录

```text
00_metadata/                    输入哈希、位姿矩阵、点坐标、指标和运行状态
01_original_frames/             五张原始图
02_clrnet_points/               全部候选点和选中的左右两条车道
03_bev_points/                  各帧本地米制 BEV 点
04_pose_aligned_points/         对齐到第000004帧的逐帧点和米制融合图
05_fusion_without_denoise/      位姿对齐后、未去噪的融合结果
06_legacy_ransac_diagnostic/    旧 X 聚类/RANSAC 的诊断对比，不作为最终方法
07_temporal_consensus_candidate/ 留一帧跨帧一致性候选；证据不足的点默认保留
```

先检查：

```text
04_pose_aligned_points/five_frame_metric_pose_fusion.png
05_fusion_without_denoise/accumulated_points_by_frame.png
05_fusion_without_denoise/weighted_score_heatmap.png
07_temporal_consensus_candidate/weighted_score_heatmap.png
00_metadata/pose_alignment.csv
00_metadata/audit.json
```

## 5. 新的保守去噪候选

完整流程会同时输出旧 RANSAC 诊断和新的留一帧跨帧候选。新方法对当前点只使用“其他帧、同一侧、相同纵向位置”的预测，不允许当前帧支持自己的点；少于两个其他帧支持的点默认保留。

当前实验候选参数为：

```text
基础阈值        0.30 m
MAD 倍数        3.0
最大阈值        1.00 m
最少其他帧数    2
```

这些参数来自受控异常注入和远处保留约束，不是 KITTI 或 CLRNet 官方参数。当前 PowerShell 包装脚本会使用这组 Python 默认参数。运行后可在以下文件核验：

```text
00_metadata/audit.json
00_metadata/temporal_retention_by_lane.csv
00_metadata/temporal_retention_by_distance.csv
```

## 6. 关于 `0.2,0.4,0.6,0.8,1.0`

这些只是项目此前使用的时间递增栅格权重，不是 KITTI、CLRNet 或 RANSAC 论文给出的参数。米制位姿融合图不使用这些权重；只有融合热力图使用。若要做等权基线：

```powershell
.\scripts\run_kitti00_first5_windows.ps1 `
  -Weights "1,1,1,1,1" `
  -OutputDir F:\2026_surf\workstation_outputs\kitti00_seq00_first5_equal_weights
```

报告时必须同时注明权重和输出目录，不得把启发式权重说成论文参数。
