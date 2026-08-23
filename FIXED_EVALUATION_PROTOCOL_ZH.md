# SURF 固定评测口径

## 结论

本项目分两张成绩单，不能把不同参考对象的数值混成一个“准确率”。

1. **CLRNet 单帧检测准确率**：预测二维车道折线与有标注数据集中的真实折线比较，报告 Precision、Recall、F1@50 和 F1@75。当前工作站没有 CULane，因此只能引用 CLRNet 论文官方基准，不能写成本机实测。
2. **KITTI 项目内部一致性**：拟合曲线与未标注的 CLRNet/IPM 观测比较，报告双向 P90、双向 0.5 m 支持率、RMSE、Chamfer、Fréchet、可用帧比例和窗口连续性。它评价多帧处理是否稳定，不是真实车道位置准确率。

## 固定比较对象

- `prediction_to_observation`：拟合曲线采样点到最近 CLRNet/IPM 观测点，约束多余或偏离的输出。
- `observation_to_prediction`：CLRNet/IPM 观测点到最近拟合曲线采样点，约束被删掉或没有覆盖的路段。
- 左右侧分开评价，不允许跨侧匹配。
- 每条曲线按 0.5 m 等弧长采样，避免点密度不同改变结果。

## 窗口评测范围校正

每个 15 帧窗口只输出自己负责的轨迹区间，但相机还会观测到更远处的车道点。评测前必须按车辆轨迹进度，把观测点限制到该窗口实际输出曲线负责的区间；区间外的前视点由后续窗口负责，不能算作当前窗口的漏检。

为与融合脚本的支撑裁剪一致，观测点还必须位于对应局部车辆轨迹 8 m 范围内。这个限制尤其用于序列末端：相机可以看到车辆尚未驶过的远方，但这些位置没有后续 pose，不能当成当前输出遗漏。

## 固定指标

主指标：

- `symmetric_p90_m`：90% 双向距离不超过该值，越小越好。
- `observation_coverage_at_0p5m`：同一支撑区间内，观测点落在拟合曲线 0.5 m 范围内的比例，越大越好。
- `prediction_precision_at_0p5m`：拟合曲线落在观测点 0.5 m 范围内的比例，越大越好。

辅助指标：Mean、RMSE、Median、P95、Max、对称 Chamfer、分窗口离散 Fréchet、留出帧 RMSE/P90、左右同时可用帧率、拟合完成率和连续性通过率。

## 运行

在 Anaconda PowerShell Prompt 中：

```powershell
Set-Location F:\surf_final
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

& .\scripts\run_fixed_project_metrics_windows.ps1 `
  -EnvName "surf2026-win"
```

脚本自动读取最新的 `workstation_outputs\surf_final_all_sequences_*`，不重跑 CLRNet，不修改历史结果，并在新的 `fixed_project_metrics_时间戳` 文件夹输出 JSON、CSV、记分图和状态文件。

## 禁止的表述

- 不得把留出帧误差或内部一致性称为 CLRNet 检测准确率。
- 不得把 CLRNet/IPM 观测称为车道线真实值。
- 不得只报告距离而省略双向支持率和运行可用率。
- 不得把 SemanticKITTI 类别 40 路面当作类别 60 车道标线。
- 不得把 CLRNet 论文的 CULane 官方数值写成本工作站运行结果。
