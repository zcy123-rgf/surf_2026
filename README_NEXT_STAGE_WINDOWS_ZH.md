# 下一阶段：急弯分段稀疏融合与人工评估

本阶段不重新运行 CLRNet，也不修改已经完成的 Sequence 00–09 输出。

脚本完成两项独立任务：

1. 对 Sequence 01、06 的已选 105 帧路段运行十个重叠的 15 帧局部拟合；每条局部左右曲线按弧长提取少量特征点，再进行第二次 pose 对齐融合；
2. 从 Sequence 09 导出 12 张独立原图及空白标注模板，用于手工伪真值评估。模板不使用 CLRNet 点作为答案。

在 Anaconda PowerShell Prompt 中运行：

```powershell
Set-Location "F:\2026_surf\ego_adjacent_experiment_20260809_134947"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

& ".\scripts\run_next_stage_lane_experiments_windows.ps1" `
  -EnvName "surf2026-win" `
  -DatasetRoot "F:\BaiduNetdiskDownload\kitti\odometry" `
  -BatchRoot "F:\2026_surf\workstation_outputs\sequences01_09_ego_adjacent_20260809_141900"
```

脚本只创建新的 `workstation_outputs\next_stage_lane_时间戳`，不会覆盖旧输出。结束后上传：

- `NEXT_STAGE_RESULT_BUNDLE.zip`：Sequence 01、06 的分段、特征点压缩和二次融合结果；
- `sequence09_manual_annotation_package.zip`：Sequence 09 的人工标注原图、清单和空白模板。

`NEXT_STAGE_RESULT_BUNDLE.zip` 中的曲线间距离只表示稀疏表示对直接拟合的保真度，不是真实车道精度。只有完成人工标注后，才能在 `04_q4_manual_evaluation` 中报告伪真值几何误差。

