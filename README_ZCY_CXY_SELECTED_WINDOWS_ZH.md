# zcy 与 cxy 两项任务：固定 100 帧自动实验

## 1. 实验范围

- 首选：KITTI Odometry Sequence 07，帧 415--514，共 100 帧；
- 备用：Sequence 03，帧 29--128，共 100 帧；
- 窗口清单：`configs/selected_windows_qly_20260814.csv`；
- 每次运行使用新的时间戳目录，不修改任何既有 output。

## 2. zcy 部分做什么

同一批图像分别运行两种左右车道选择：

1. `ego_adjacent`：每帧在相机中心左右各取最近候选，只使用当前帧；
2. `temporal_ego`：把上一帧左右曲线通过 KITTI pose 变到当前帧，再按几何距离关联。

比较指标包括有效双车道帧率、最长连续有效帧、相邻帧位姿对齐后的 Mean/Median/P90/Max 距离、连续性门限失败次数，以及两种方法明显不同的帧。CLRNet 候选序号只在单帧内有效；`ego_left/ego_right` 是本项目生成的时序轨迹名，不是 CLRNet 或 KITTI 官方车道 ID。

主要结果：

- `03_zcy_identity_comparison/01_metrics/mode_summary.csv`；
- `03_zcy_identity_comparison/01_metrics/continuity_by_frame.csv`；
- `03_zcy_identity_comparison/02_figures/continuity_comparison.png`；
- `03_zcy_identity_comparison/03_method_overlays/`。

判定原则：只有在 `temporal_ego` 没有造成不可接受的有效帧损失、同时连续性误差下降时，才优先采用它；差异较大的帧必须看叠加图。连续并不能单独证明检测对象不是人行道边缘。

## 3. cxy 部分做什么

读取同编号 Velodyne 点云和 SemanticKITTI `.label`，从标签低 16 位提取原始语义 ID 60（lane-marking），通过 `Tr`、`P2` 和 KITTI pose 变换到选定参考帧。类别 60 是官方稀疏标线语义点，不是连续的左右车道真值曲线，也没有官方左右实例 ID。

左右参考点只关联一次：使用 `temporal_ego` 的左右点作为固定锚点，将每个类别 60 点最多分给一侧；多项式和 B 样条共同使用同一份参考点，不能各自挑选参考。评测输出：

- 参考点到拟合曲线的 Mean、RMSE、Median、P90、Max；
- 0.3 m、0.5 m、1.0 m 覆盖率；
- 支持区内的反向距离及双向 F1；
- 类别 60 覆盖不足时的留出帧一致性结果。

主要结果：

- `07_cxy_semantic_evaluation/00_audit/semantic_frame_coverage.csv`；
- `07_cxy_semantic_evaluation/01_metrics/semantic_method_summary.csv`；
- `07_cxy_semantic_evaluation/01_metrics/semantic_metrics_by_side.csv`；
- `07_cxy_semantic_evaluation/01_metrics/heldout_consistency_fallback.csv`；
- `07_cxy_semantic_evaluation/02_figures/semantic_reference_overlay.png`。

只有 `RESULT.json` 中 `semantic_coverage_gate_passed=true` 时，类别 60 指标才能作为本段的主要稀疏参考结论；否则只能报告留出帧一致性，不能称为真实准确率。

## 4. 工作站一键运行

在 Anaconda PowerShell Prompt 中进入包含本 README 的工程目录，然后运行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

& ".\scripts\run_selected_zcy_cxy_windows.ps1" `
  -EnvName "surf2026-win" `
  -DatasetRoot "F:\BaiduNetdiskDownload\kitti\odometry" `
  -SemanticKittiRoot "F:\BaiduNetdiskDownload\SemanticKITTI" `
  -ClrnetRoot "F:\2026_surf\CLRNet" `
  -Device cuda
```

默认依次运行首选和备用窗口；某个窗口失败时会保留诊断并继续另一个。结束后控制台会显示：

```text
ZCY + CXY AUTOMATED RUN FINISHED
Output root: ...
Upload this small result bundle: ...\zcy_cxy_meeting_bundle.zip
```

将该小 ZIP 上传即可核验结论，不必上传包含全部原始帧的大目录。

## 5. 程序入口与主体代码

- 一键入口：`scripts/run_selected_zcy_cxy_windows.ps1`；
- CLRNet 候选和两种左右选择：`scripts/scan_clrnet_lane_counts.py`；
- zcy 指标与差异图：`scripts/analyze_lane_identity_modes.py`；
- 固定 100 帧选择文件：`scripts/create_fixed_window_selection.py`；
- 多项式：`scripts/fit_extended_polynomial.py`；
- B 样条：`scripts/fit_extended_bspline.py`；
- cxy 类别 60 自动评测：`scripts/evaluate_semantickitti_curve_reference.py`。

## 6. 已完成的代码核验

- Python 文件可编译；
- PowerShell 入口语法解析无错误；
- 新增测试及原工程测试共 73 项通过；
- 本地没有 Sequence 07/03 的完整工作站数据，因此最终数值必须由工作站真实运行生成，代码中没有预填实验结果。
