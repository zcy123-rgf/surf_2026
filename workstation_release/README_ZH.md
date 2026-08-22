# Windows工作站精简可运行版

本目录只给出两个明确入口，旧结果、PPT生成、候选片段搜索、SemanticKITTI审计和本周稀疏层级融合不作为默认流程。原工程与旧输出不会被修改。

## 默认：RANSAC参考版

入口：`01_ransac_reference/run_windows.ps1`

流程：KITTI 00前五帧原图 → CLRNet候选点 → 点式IPM → KITTI pose对齐到第4帧 → 固定的安全优先分侧三次多项式RANSAC → 点坐标、融合图和审计JSON。

固定配置来自已完成的安全扩展审核：

- 左右分开拟合 `X=f(Z)`；
- 三次多项式；
- Z归一化，改善数值条件；
- 阈值 `min(0.35 + 0.015×max(Z-3,0), 0.80)` 米；
- 1000次RANSAC；
- 随机样本至少覆盖10米Z范围；
- 按3–10、10–20、20–30、30–40、40–50米分段均衡评分；
- 1次局部重拟合。

它通过了本次预设防过删约束，但对远处渐变漂移的独立注入试验召回只有13.7%。因此它是“安全优先参考配置”，不是KITTI车道线准确率意义上的全局最优。

运行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& ".\workstation_release\01_ransac_reference\run_windows.ps1" `
  -EnvName "surf2026-win" `
  -DatasetRoot "F:\BaiduNetdiskDownload\kitti\odometry" `
  -Device cuda
```

## 可选：多项式/B样条版

入口：`02_curve_models/run_windows.ps1`

该入口读取相同的pose对齐点，保留1/2/3次鲁棒多项式和三次B样条，通过留出帧RMSE比较模型，并导出直接拟合的左右B样条。它在模型比较后停止，不运行本周PPT中的短路段锚点压缩和十段稀疏再融合。

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& ".\workstation_release\02_curve_models\run_windows.ps1" `
  -EnvName "surf2026-win" `
  -DatasetRoot "F:\BaiduNetdiskDownload\kitti\odometry" `
  -Device cuda
```

## 文件作用

- `surf_bev/detectors.py`：CLRNet封装与候选点输出。
- `surf_bev/geometry.py`：IPM、KITTI pose读取与帧间坐标变换。
- `surf_bev/optimized_ransac.py`：固定方法族的分侧多项式RANSAC实现，不包含参数搜索。
- `surf_bev/point_export.py`：逐点保留/拒绝决定及坐标导出。
- `scripts/run_full_point_pipeline.py`：从原图生成CLRNet、IPM和pose对齐中间数据；其历史诊断输出不代表默认最终去噪方法。
- `scripts/run_selected_ransac_reference.py`：应用固定RANSAC配置并输出结果。
- `scripts/fit_first5_two_curves.py`：左右鲁棒三次B样条及平滑参数选择。
- `scripts/analyze_lane_curve_hierarchy.py`：复用其中已验证的多项式和B样条比较函数；稀疏层级部分不由精简入口调用。
- `scripts/compare_polynomial_bspline.py`：仅运行1/2/3次多项式与B样条比较，不运行稀疏层级融合。
- `scripts/kitti00_workstation_input.ps1`：解析工作站KITTI目录并检查图片、pose、calib行数。
- `scripts/check_windows_env.py`、`setup_clrnet_windows.ps1`：Windows环境与CLRNet兼容检查。

## 输出边界

- 每次运行创建新的 `clean_outputs` 时间戳目录，不覆盖旧结果。
- 0.2、0.4、0.6、0.8、1.0只是五帧栅格显示权重，不参与pose变换，也不是RANSAC点权重。
- 人工标注是防过删伪参考，不是KITTI官方车道线真值。
- 拟合误差和留出帧误差是对CLRNet/IPM派生点的内部一致性，不是官方准确率。
