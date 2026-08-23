# SURF固定评测口径

## 结论

本项目使用两张分开的成绩单，不能把不同参考对象的数值混成一个“准确率”。

1. **CLRNet单帧检测准确率**：预测二维车道折线与有标注数据集中的真实折线比较，固定报告 Precision、Recall、F1@50 和 F1@75。当前工作站没有 CULane，因此只允许引用 CLRNet 论文官方基准，不能称为本机实测。
2. **KITTI项目内部一致性**：拟合曲线与未标注的 CLRNet/IPM 观测比较，固定报告双向 P90、双向 0.5 m 覆盖率、RMSE、Chamfer、Fréchet、可用帧覆盖率和窗口连续性。它评价多帧处理是否稳定，不是真实车道位置准确率。

## 固定比较对象

- `prediction_to_observation`：拟合曲线采样点到最近 CLRNet/IPM 观测点。它约束多余或偏离的输出。
- `observation_to_prediction`：CLRNet/IPM 观测点到最近拟合曲线采样点。它约束漏掉或被删掉的路段。
- 两个方向合并后计算 `symmetric_p90_m`；同时报告两个方向在 0.3、0.5、1.0 m 下的覆盖率。
- 每条曲线先按 0.5 m 等弧长采样，避免不同点密度改变结果。
- 左右车道分别评价，不允许跨侧匹配。

## 固定指标

主指标：

- `symmetric_p90_m`：90%的双向距离不超过该值；越小越好。
- `observation_coverage_at_0p5m`：观测点中落在拟合曲线0.5 m范围内的比例；越大越好。
- `prediction_precision_at_0p5m`：拟合曲线中落在观测点0.5 m范围内的比例；越大越好。

辅助指标：

- Mean、RMSE、Median、P95、Max；
- 对称 Chamfer 距离；
- 分窗口离散 Fréchet 距离；
- 留出帧加权 RMSE 与各窗口P90分布；
- 左右同时可用帧率、完成拟合率、窗口连续性通过率。

## 运行

在 Anaconda PowerShell Prompt 中：

```powershell
Set-Location F:\surf_final
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

& .\scripts\run_fixed_project_metrics_windows.ps1 `
  -EnvName "surf2026-win"
```

脚本自动读取最新的 `workstation_outputs\surf_final_all_sequences_*`，不重跑CLRNet，不修改旧结果，并在新的 `fixed_project_metrics_时间戳` 文件夹输出：

- `FIXED_METRICS.json`；
- `FIXED_METRICS_BY_SEQUENCE.csv`；
- `FIXED_METRICS_BY_WINDOW_SIDE.csv`；
- `FIXED_METRICS_SCORECARD.png`；
- `STATUS.json`。

## 禁止的表述

- 不得把留出帧误差称为 CLRNet 检测准确率；
- 不得把 CLRNet/IPM 观测称为车道线真实值；
- 不得只报告距离而省略覆盖率；
- 不得把 SemanticKITTI 原始类别40道路面当作类别60车道标线；
- 不得把论文官方 CULane 数值写成本工作站运行结果。
