# Sequence 01 四组模型与遮挡补全对比

该实验比较同一批 Sequence 01 帧851–1005观测上的四种输出：

1. 直道多项式、过渡/弯道B样条的自适应路线，不补线；
2. 同一自适应路线，加安全门限通过的低置信Hermite虚线；
3. 全窗口二次参数多项式路线，不补线；
4. 同一多项式路线，加完全相同门限的低置信Hermite虚线。

补线只使用同一侧车道缺口前后的端点和多节点切向，必须同时通过顺序间隔、端点距离、切向角、弦方向、曲率和车道宽度门限。实线观测段不会被改动，虚线不会作为检测观测，也不会进入拟合误差。

## 工作站运行

在 PowerShell 中执行：

```powershell
Set-Location F:\surf_final
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

& .\scripts\run_model_completion_comparison_windows.ps1 `
  -EnvName "surf2026-win" `
  -ProjectRoot "F:\surf_final" `
  -WorkstationOutputs "F:\surf_final\workstation_outputs"
```

脚本会自动寻找最新的 Sequence 01 帧851–1005自适应B样条结果和全多项式结果，并创建一个全新的输出目录，不修改历史结果。

## 首先检查

- `05_four_way_lane_map_comparison.png`：四张地图并列；
- `06_model_fit_metric_comparison.png`：同窗口留出帧指标；
- `07_completion_audit_comparison.png`：实线支持长度和虚线假设长度；
- `STATUS.json`：输入一致性、模型计数和补线数量；
- `model_metric_pairs.csv`：同窗口、同车道侧的多项式/B样条配对指标；
- `completion_audit_metrics.csv`：两种路线的线段、补线、长度和置信度。

输出旁会自动生成同名 ZIP，上传该 ZIP 即可进行最终核验。
