# 弯道分段B样条复试

首轮实测表明：`temporal_joint`没有增加Sequence 07有效帧；中心/正宽度耦合B样条使留出RMSE由0.3130 m增加到0.7503 m。因此二者均不直接采用。

本复试不重新运行CLRNet，而是读取首轮工作站输出，完成两项工作：

1. 统计100帧无效的真实原因；
2. 使用有支持边界的局部重叠B样条，避免一个全局模型跨越急弯稀疏区强连。

在包含最新代码的新工程目录中执行：

```powershell
& ".\scripts\run_piecewise_retry_windows.ps1" `
  -EnvName "surf2026-win" `
  -ExistingRunRoot "F:\2026_surf\curved_improvement_source_20260816_162833\workstation_outputs\curved_improvement_20260816_162856"
```

该复试复用已有`scan.json`、`selected_lane_points.json`、pose和calib，不调用CLRNet、不需要GPU，也不会修改首轮输出。

结束时上传终端显示的：

```text
piecewise_retry_bundle.zip
```

验收规则：

- 先根据`failure_summary.csv`决定覆盖率下一步改哪一层；
- `piecewise_selection_gate_passed`必须为`true`；
- `piecewise_curves_cross`必须为`false`；
- 留出帧Mean RMSE应不高于独立B样条0.3130 m；
- 最终图不能用拟合线跨越没有局部模型支持的内部空白区。
