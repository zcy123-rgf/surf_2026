# Windows工作站：弯道与长道路实验

本实验不覆盖任何旧输出，也不重新运行CLRNet。它复用Sequence 00–09批处理已经保存的逐帧CLRNet/IPM点。

## 两组实验

1. 单纯弯道：已有Sequence 01帧857–961结果，约164 m、净转向120.7°，多项式和B样条均已完成。
2. 长道路包含弯道：在完整Sequence扫描结果中选择一段较长、持续发生转向且左右候选覆盖达标的道路；默认使用30个15帧窗口、步长10帧，共305个唯一帧。

长道路实验同时输出：

- 一条全局多项式，作为对照；
- 一条全局B样条，作为对照；
- 15帧局部多项式经过重叠区域融合的结果；
- 15帧局部B样条经过重叠区域融合的结果。

相邻窗口没有共同支持，或B样条在重叠区域的位置/方向差异超过登记阈值时，程序开始新的道路段，不跨缺口强行连接。阈值只是几何连续性启发式门槛，不是真实车道准确率。

## 一键运行

先把GitHub最新分支同步到工作站工程，再在Anaconda PowerShell Prompt中执行：

```powershell
Set-Location "F:\2026_surf\clean_workstation_release_20260804_004638"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

& ".\scripts\run_curved_road_experiments_windows.ps1" `
  -EnvName "surf2026-win" `
  -DatasetRoot "F:\BaiduNetdiskDownload\kitti\odometry" `
  -ExistingBatchRoot "F:\2026_surf\workstation_outputs\sequences01_09_ego_adjacent_20260809_141900" `
  -SequenceId "01" `
  -WindowCount 30
```

`WindowCount 30`对应305个唯一帧：

```text
15 + (30 - 1) × 10 = 305
```

如果会议前时间充足，可使用50个窗口覆盖505帧：

```powershell
& ".\scripts\run_curved_road_experiments_windows.ps1" `
  -EnvName "surf2026-win" `
  -DatasetRoot "F:\BaiduNetdiskDownload\kitti\odometry" `
  -ExistingBatchRoot "F:\2026_surf\workstation_outputs\sequences01_09_ego_adjacent_20260809_141900" `
  -SequenceId "01" `
  -WindowCount 50
```

每次运行使用新的时间戳目录，并在结束时打印：

- `05_meeting_figures`：四张会议可直接查看的结果图；
- `STATUS.json`：选中帧、窗口数和自动分出的道路段数；
- `curved_road_sequence_...zip`：可上传的小型结果包。

## 结果口径

- 全局多项式和全局B样条是对照，不代表推荐把整段道路强行拟合成一条曲线。
- 推荐结果是`04_piecewise_long_road`中的分段重叠融合。
- 模型选择与连续性指标基于CLRNet/IPM派生点，不是KITTI官方车道真值精度。
- IPM仍继承本项目的相机高1.65 m、俯仰0°和平坦路面假设。
