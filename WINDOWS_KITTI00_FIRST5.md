# Windows：KITTI Sequence 00 前五帧位姿融合

运行入口直接读取工作站已有的 KITTI Odometry 官方数据，不读取仓库中的
历史结果，也不要求把数据集复制到项目目录。

默认数据根目录：

```text
F:\BaiduNetdiskDownload\kitti\odometry
```

实际输入：

- 左彩色图：`data_odometry_color\dataset\sequences\00\image_2`
- 相机标定：`data_odometry_calib\dataset\sequences\00\calib.txt`
- 官方位姿：`data_odometry_poses\dataset\poses\00.txt`
- 时间戳：`data_odometry_color\dataset\sequences\00\times.txt`
- 帧号：`000000` 至 `000004`
- 参考帧：`000004`

脚本先核验前五帧 SHA256、`P2`、位姿格式以及 Sequence 00 的完整项数，
通过后才启动 CLRNet。

## 处理流程

1. CLRNet 从每张彩色图输出候选车道的有序二维图像点。
2. 当前项目的平坦路面 IPM 把点变换为相机局部道路平面的米制 `(X,Z)` 点。
3. KITTI `poses/00.txt` 给出 rectified camera 0 的世界位姿；代码用 `P2`
   处理 camera 0 与 `image_2` 相机的固定关系。
4. 对第 `i` 帧计算：

   ```text
   T_000004_from_i = inverse(T_world_000004) @ T_world_i
   ```

5. 将各帧米制点变换到 `000004` 坐标系后累积融合。
6. 需要显示时，再把米制点绘制成 PNG；融合本身不是在 PNG 像素上完成。

相机高度 `1.65 m`、俯仰角 `0°` 和平坦路面仍是项目实验假设，不是
KITTI 官方逐帧路面外参。

## 运行

```powershell
Set-Location F:\2026_surf
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run_kitti00_first5_windows.ps1
```

每次默认新建：

```text
workstation_outputs\kitti00_first5_时间戳
```

优先检查：

```text
00_metadata\audit.json
00_metadata\pose_alignment.csv
04_pose_aligned_points\five_frame_metric_pose_fusion.png
05_fusion_without_denoise\accumulated_points_by_frame.png
05_fusion_without_denoise\weighted_score_heatmap.png
```

自定义数据根目录或输出目录：

```powershell
.\scripts\run_kitti00_first5_windows.ps1 `
  -DatasetRoot "F:\新的位置\odometry" `
  -OutputDir "F:\2026_surf\workstation_outputs\pose_fusion_run2"
```

`0.2,0.4,0.6,0.8,1.0` 仅用于融合热力图的时间递增权重，不是
RANSAC 阈值，也不影响米制位姿对齐散点图。
