# Windows工作站：SemanticKITTI与Odometry 00逐帧标线核验

本工具只新增时间戳结果目录，不会读取旧输出作为输入，也不会覆盖已有结果。

SemanticKITTI标签不是KITTI Odometry下载包的一部分。官方标签地址：

`https://semantic-kitti.org/assets/data_odometry_labels.zip`

在Anaconda PowerShell Prompt中运行：

```powershell
Set-Location F:\2026_surf

.\scripts\run_semantickitti_lane_audit_windows.ps1 `
  -DatasetRoot "F:\BaiduNetdiskDownload\kitti\odometry" `
  -SemanticKittiRoot "F:\BaiduNetdiskDownload\SemanticKITTI" `
  -DownloadLabels
```

第一遍只做快速覆盖审计，不读取全部Velodyne坐标。输出位于：

`F:\2026_surf\generated_runs\semantickitti_lane_audit_时间戳\`

重点查看：

- `summary.json`：4541帧对齐数、含ID 60的帧比例、不同点数门槛下最长连续区间；
- `frame_lane_marking_counts.csv`：每帧原始lane-marking点数；
- `contiguous_lane_marking_runs.csv`：连续可用候选片段。

确认覆盖足够后，再导出米制坐标：

```powershell
.\scripts\run_semantickitti_lane_audit_windows.ps1 `
  -DatasetRoot "F:\BaiduNetdiskDownload\kitti\odometry" `
  -SemanticKittiRoot "F:\BaiduNetdiskDownload\SemanticKITTI" `
  -ExportPoints `
  -ReferenceId 0
```

这会额外生成 `lane_marking_points_reference_camera.csv`。坐标列采用参考相机0坐标：X向右、Y向下、Z向前。变换使用Odometry `calib.txt` 的Velodyne到相机外参和同编号pose。

注意：原始语义ID 60是稀疏lane-marking激光点，没有自车道左/右实例身份，不能直接称为两条连续真值曲线。必须先统计覆盖，再做左右关联和曲线评价。
