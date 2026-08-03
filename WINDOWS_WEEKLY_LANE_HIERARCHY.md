# 本周车道曲线与层级融合实验（Windows）

本入口处理组会提出的四个问题，并始终新建输出目录：

1. 将多帧 CLRNet/IPM 点通过 KITTI pose 对齐，在 Frenet `s/d` 坐标中分别拟合左右两条鲁棒三次 B 样条；
2. 用留出帧验证比较 1/2/3 次多项式与 B 样条，判断当前路段是否需要非线性曲线；
3. 对 10 个“15 帧、步长 10、相邻重叠 5 帧”的窗口分别拟合，比较每段每侧 4/6/8/12 个锚点的反向重建误差，再用少量锚点完成总融合；
4. 用米制曲线距离和固定容差下的 Precision/Recall/F1 替换像素重叠率。没有人工标注时只生成待标注状态，不报告准确率。

## 一条命令运行

在 Anaconda PowerShell Prompt 中：

```powershell
Set-Location F:\2026_surf
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

& "F:\2026_surf\scripts\run_weekly_meeting_complete_windows.ps1" `
  -EnvName "surf2026-win" `
  -DatasetRoot "F:\BaiduNetdiskDownload\kitti\odometry" `
  -Device cuda
```

脚本优先复用最新且通过覆盖门槛的 `hierarchy_overlap_selection_*`。若不存在，它会先复用已有 CLRNet 扫描；只有连扫描也不存在时才运行新的 CLRNet 可行性扫描。

## 输出

每次运行写入全新目录：

```text
F:\2026_surf\workstation_outputs\weekly_lane_hierarchy_时间戳\
```

关键文件：

- `MEETING_SUMMARY.md`：四个问题的实际方法与本次真实数值；
- `01_q1_two_curves/final_two_curves.png`：最终左右两条曲线；
- `02_q2_model_comparison/model_comparison.png`：模型留出帧误差比较；
- `03_q3_sparse_refusion/`：稀疏锚点、压缩试验、窗口连接连续性与总融合保真度；
- `04_q4_manual_evaluation/PENDING.json`：尚无人工标注时的严格状态；
- `meeting_result_bundle.zip`：用于制作 PPT 和演讲稿的小型结果包。

请上传 `meeting_result_bundle.zip`。如果还要完成人工伪真值评估，也需上传筛选目录中的 `manual_annotation_package.zip`，标注完成后使用：

```powershell
& "F:\2026_surf\scripts\run_weekly_meeting_complete_windows.ps1" `
  -EnvName "surf2026-win" `
  -DatasetRoot "F:\BaiduNetdiskDownload\kitti\odometry" `
  -Device cuda `
  -ManualJson "人工标注JSON的完整路径"
```

## 结果边界

- 拟合误差和稀疏再融合相对直接融合的误差，只表示内部一致性/压缩保真度，不是准确率；
- 人工标注只能称“人工参考”或“伪真值”，不是 KITTI 官方车道线真值；
- pose 只负责坐标对齐，不能修复 CLRNet 选错车道或 IPM 系统误差；
- IPM 仍继承当前项目假设：相机高 1.65 m、俯仰角 0°、平坦路面。
