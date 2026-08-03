# Windows：从工作站 KITTI 数据重新运行 RANSAC 对比

本入口从工作站原始数据重新执行 CLRNet、点式 BEV、位姿对齐和 RANSAC
对比。它不读取 `results` 或既有 `workstation_outputs`。

默认 KITTI 根目录：

```text
F:\BaiduNetdiskDownload\kitti\odometry
```

仓库只保留算法代码和：

```text
annotations\kitti00_first5_manual_annotations.json
```

该 JSON 是项目内部人工道路边界伪标注，只用于方法对比，不能称为 KITTI
官方车道线真值。KITTI 原图、标定、位姿和历史结果均不作为本次新增输入提交。

## 快速对比

```powershell
Set-Location F:\2026_surf
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run_ransac_improvement_windows.ps1 -Mode Compare
```

它会依次：

1. 从 `image_2` 的前五帧重新运行 CLRNet；
2. 计算米制道路平面点；
3. 读取 `calib.txt` 和完整 `poses\00.txt`，对齐到 `000004`；
4. 对比未去噪、旧固定横向范围加 RANSAC以及改进候选；
5. 将所有内容写入新的时间戳目录。

主要输出：

```text
01_from_scratch_pipeline\
  00_metadata\audit.json
  00_metadata\detected_lane_points.json
  04_pose_aligned_points\five_frame_metric_pose_fusion.png

02_ransac_method_comparison\
  comparison\main_comparison.png
  00_audit\evaluation.json
```

## 完整参数复核

快速对比跑通后再运行：

```powershell
.\scripts\run_ransac_improvement_windows.ps1 -Mode Full
```

`Full` 会继续执行 RANSAC 参数搜索、安全约束扩展和独立随机种子复核，
运行时间明显长于 `Compare`。

## 结果解释边界

- RANSAC 输入是位姿对齐后的米制点，不是 BEV PNG 像素。
- 人工标注只是伪标注，因此相应分数是“一致程度”，不是准确率。
- 受控异常注入用于比较已知异常的识别能力，不能替代真实车道线真值。
- `0.2,0.4,0.6,0.8,1.0` 是融合热力图权重，不是 RANSAC 阈值。
- 新输出目录必须为空，避免把旧图误认为本次运行结果。
