# SURF 最终工作区使用说明

## 1. 项目范围

本工作区交付的是一个可复查的车道几何原型：

`KITTI Odometry 图像 → CLRNet 候选车道点 → 自车左右候选关联 → IPM 米制地面点 → KITTI pose 多帧对齐 → 位姿曲率分段 → 分窗口鲁棒拟合 → 多项式/B样条选择 → 重叠窗口融合 → 定量审计`

正式路段为 **KITTI Odometry Sequence 01，帧 851–1005**。全 Sequence 01 的 0–1100 帧作为规模扩展实验。所有几何输出均为公共参考相机下的米制笛卡尔 `X/Z`，没有使用 Frenet 坐标。

本阶段没有实现新的 SLAM 后端和完整 4D 语义地图；KITTI pose 是已发布的位姿输入。没有逐帧官方左右车道边界真值，因此所有留出误差均为内部一致性指标，不称为真实准确率。

## 2. 组员十分钟理解路线

第一次打开项目时，按下面顺序阅读即可：

1. 先看本文件第3节和第4节，了解怎样打开项目、目录分别保存什么；
2. 再看`SURF_FINAL_PIPELINE_ZH.md`，理解每个模块的输入、处理和输出；
3. 需要了解实验结论时看`SURF_FINAL_DECISIONS_ZH.md`和`FIXED_EVALUATION_PROTOCOL_ZH.md`；
4. 需要重跑时，从`scripts/`中的PowerShell入口开始，不直接改`results/`中的正式结果；
5. 需要调试算法时，再进入对应Python实现或`surf_bev/`公共模块。

整个项目的代码关系如下：

| 阶段 | 一键/批处理入口 | 主要Python实现 | 关键输出 |
|---|---|---|---|
| 数据定位与校验 | `kitti_odometry_workstation_input.ps1` | PowerShell内部完成 | 图像、标定、pose、时间戳路径与数量检查 |
| CLRNet检测与左右关联 | 由`run_surf_final_window_windows.ps1`调用 | `scan_clrnet_lane_counts.py`、`surf_bev/detectors.py` | `scan.json`、`selected_lane_points.json`、逐帧诊断图 |
| IPM与pose对齐 | 同上 | `surf_bev/geometry.py` | 公共参考相机下的米制`X/Z`点 |
| pose曲率与路段分类 | 同上；全量只扫pose可用`run_pose_curvature_all_sequences_windows.ps1` | `analyze_pose_curvature.py` | 曲率CSV、阈值JSON、直道/过渡/弯道分段图 |
| 分窗口去噪、拟合与融合 | `run_adaptive_xz_piecewise_windows.ps1` | `fit_adaptive_xz_piecewise.py`、`fit_first5_two_curves.py` | 模型比较、窗口曲线、融合点、接口诊断 |
| 遮挡缺口审计 | 由总控脚本调用 | `bridge_occluded_lane_segments.py` | 观测段、被拒缺口、低置信度虚线桥接 |
| 固定定量评测 | `run_fixed_project_metrics_windows.ps1` | `evaluate_fixed_project_metrics.py` | Sequence 00–10指标CSV、JSON和总览图 |

三个最常用入口：

- 重跑155帧正式案例：`run_surf_final_sequence01_windows.ps1`；
- 重跑Sequence 00–10全量实验：`run_surf_final_all_sequences_windows.ps1`；
- 不重跑CLRNet、只重新汇总指标：`run_fixed_project_metrics_windows.ps1`。

## 3. 打开方式

### PyCharm

打开项目目录：

`F:\surf_final`

选择已经配置好的解释器：

`C:\Users\IR713\anaconda3\envs\surf2026-win\python.exe`

PyCharm 主要用于查看和调试 Python 文件。正式批处理入口是 PowerShell 脚本，建议在 Anaconda PowerShell Prompt 中运行。

### Anaconda PowerShell Prompt

```powershell
Set-Location F:\surf_final
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 4. 顶层目录

### `CLRNet/`

外部车道检测模型运行时、配置和权重。最终脚本只调用它，不在实验过程中修改它。

主要内容：

- `configs/`：模型结构、数据预处理和推理配置；
- `clrnet/models/`：CLRNet 网络、检测头和损失等模型实现；
- `clrnet/datasets/`：输入预处理与数据接口；
- `clrnet/ops/`：NMS 等算子；Windows 工作站可能使用兼容实现；
- `clrnet/utils/`：配置、车道曲线对象和通用工具；
- `weights/`：预训练权重。

CLRNet 每帧输出若干候选车道曲线；程序将每条候选采样为有序二维图像点。候选顺序和本项目生成的 track ID 都不是跨帧固定语义车道 ID。

### `annotations/`

- `kitti00_first5_manual_annotations.json`：最早5帧RANSAC比较使用的人工伪标注。只用于方法一致性比较，不能称为KITTI官方真值。

### `scripts/`

项目的可运行入口和实验实现，逐文件见第5节。

### `surf_bev/`

被脚本调用的共用运行模块，逐文件见第6节。最终工作区只保留实际依赖文件，旧的硬编码演示脚本不复制进来。

### `results/`

经过核验后固定保存的正式结果：

- `sequence01_851_1005_formal/`：155帧正式结果；不在这里重新运行代码，也不覆盖文件。

### `workstation_outputs/`

新实验输出区。每次运行建立新的时间戳目录，程序拒绝覆盖非空目录。这里允许出现多次正式运行结果，但不放手工草稿。

### `backups/`

由工作区审计脚本生成的源码与CLRNet运行时备份。备份不包含体积较大的全量实验输出；正式输出继续保存在`workstation_outputs/`。

## 5. `scripts/` 中每个文件的作用

### A. 最终长路段主流程

#### `run_surf_final_sequence01_windows.ps1`

155帧正式实验的一键入口。固定使用Sequence 01帧851–1005、直道种子851–875和991–1005、15帧窗口和10帧步长。可加`-RunFullPoseAudit`，先对Sequence 00–10运行pose-only曲率扫描。

#### `run_surf_final_sequence01_full_windows.ps1`

全Sequence 01规模实验入口，处理0–1100共1101帧。仍使用经过核验的两段直道种子标定阈值；直道固定多项式，过渡和弯道比较多项式与B样条。该实验用于检查规模、覆盖和稳定性，不是官方准确率评测。

#### `run_surf_final_all_sequences_windows.ps1`

Sequence 00–10全量批处理入口，共覆盖23,201帧。每个Sequence使用独立目录，单个失败不会中断后续任务；再次运行时会复用帧范围完全相同且状态完整的既有结果，因此不会重复计算已完成的Sequence 01。每处理完一个Sequence都会刷新`BATCH_SUMMARY.csv`和`BATCH_STATUS.json`，意外中断后可指定原`-OutputRoot`继续。

目前只有Sequence 01登记了人工核验的直道种子；其他Sequence使用低曲率四分位自动基线，汇总表会保留`threshold_baseline`字段。全量结果用于覆盖率、稳定性和可扩展性统计，不作为官方车道定位准确率。

#### `run_surf_final_window_windows.ps1`

通用单路段总控脚本。依次执行车道检测/关联、曲率分类、分窗口模型拟合与融合、缺口审计，并生成`FINAL_STATUS.json`、`FINAL_METRICS.json`和review ZIP。

#### `run_fixed_project_metrics_windows.ps1`

读取已经完成的Sequence 00–10结果，按固定协议重新汇总内部一致性指标，不重跑CLRNet、不修改历史输出。最终协议为`2.0-domain-aligned`：只在每个窗口实际输出曲线负责的轨迹区间内做双向距离比较。

#### `evaluate_fixed_project_metrics.py`

固定评测实现。左右分开计算双向P90、0.5 m支持率、RMSE、Chamfer、Fréchet，并同时保留可用帧、拟合完成率和接口连续性。结果不是官方车道真值准确率。

#### `audit_surf_final_workspace.ps1`

只读检查`F:\surf_final`的必需文件、CLRNet权重、全量结果、固定评测、顶层额外内容、缓存和未登记输出。可选生成包含CLRNet运行时的源码备份；不会删除或移动原文件。目录规范见`FINAL_WORKSPACE_INVENTORY_ZH.md`。

#### `clean_surf_final_workspace.ps1`

只用于2026-08-23最终审计后的精确清理。运行前强制检查11/11全量结果、协议2.0评测和CLRNet运行时备份；只删除清单中逐项确认的旧补丁、旧错误评测目录和可再生Python缓存，随后自动重新审计。不会删除完整Sequence 01、Sequence 00–10、正式评测或备份。

#### `kitti_odometry_workstation_input.ps1`

解析工作站KITTI目录，定位指定Sequence的`image_2`、`calib.txt`、`poses/xx.txt`和`times.txt`；检查图像、位姿、时间戳数量及帧编号是否对齐。

#### `scan_clrnet_lane_counts.py`

逐帧调用CLRNet，保存所有候选及候选数量；初始化自车左/右边界后，按pose补偿的历史几何分别关联左右观测。缺失一侧时保留另一侧，不伪造缺失车道。

主要输入：相机图像、CLRNet、标定、pose。主要输出：`scan.json`、`lane_counts.csv`、`selected_lane_points.json`及逐帧诊断图。

#### `analyze_pose_curvature.py`

从KITTI pose平移轨迹计算米制曲率。先按路程均匀重采样，再用15 m Savitzky–Golay局部多项式平滑和求导；使用直道噪声标定高低双阈值，并要求连续5帧才切换直道/弯道状态。

主要输出：逐帧曲率、窗口分类、连续状态段、曲率总览图和阈值JSON。

#### `run_pose_curvature_all_sequences_windows.ps1`

只读取pose，对Sequence 00–10做完整轻量曲率扫描；不运行CLRNet。没有人工确认直道种子时使用最低曲率四分位作候选发现基线，因此不能把其阈值直接当成最终正式阈值。

#### `fit_adaptive_xz_piecewise.py`

长路段核心拟合程序。每个窗口左右分开聚合：直道使用鲁棒二次参数多项式；过渡和弯道在相同留出帧上比较多项式与三次参数B样条。随后将窗口曲线变换到公共参考帧，检查重叠区P95距离和切向夹角，通过才融合。

主要输出：所有模型的训练/留出指标、选择原因、窗口曲线、融合节点、连续性诊断、断点和总览图。

#### `run_adaptive_xz_piecewise_windows.ps1`

`fit_adaptive_xz_piecewise.py`的Windows参数封装，检查输入、组织conda命令并报告运行状态。

#### `fit_first5_two_curves.py`

提供两侧点的公共读取、鲁棒聚合和参数曲线拟合函数；也保留最初连续帧拟合实验。当前长路段程序复用其中的标定/投影解析和曲线工具。

#### `bridge_occluded_lane_segments.py`

只审计同侧相邻观测段之间的缺口。端点距离、两端切向、桥曲率等均通过门限时，才输出三次Hermite虚线假设；观测段从不被修改。假设桥不是检测结果。

### B. 改进RANSAC参考流程

RANSAC部分保留作为早期5帧去噪基线和方法比较，不参与Sequence 01长弯道的最终模型选择。

#### `run_ransac_improvement_windows.ps1`

RANSAC实验总入口，生成旧方法、候选改进方法、参数搜索和核验结果。每次输出到新目录。

#### `kitti00_workstation_input.ps1`

只解析KITTI Sequence 00前5帧所需图像、标定、pose和时间戳。

#### `run_full_point_pipeline.py`

从原始图像重新生成5帧CLRNet点、IPM点、pose对齐点、去噪点和阶段图，不读取旧结果图作为输入。

#### `evaluate_denoise_methods.py`

在相同5帧米制点上比较未去噪、旧RANSAC和候选改进方案，输出点级保留、远端保留及伪标注一致性指标。

#### `optimize_ransac_parameters.py`

在不改变“分侧多项式RANSAC”方法族的条件下搜索阈值、距离增量、次数和采样限制；调参使用注入的已知离群点，真实点集只做安全约束。

#### `optimize_ransac_safety_expansion.py`

围绕安全候选继续扩大同类参数搜索，重点防止为提高重叠而删除远处真实道路点。

#### `audit_ransac_shortlist.py`

使用与调参阶段不同的随机种子重新核验候选配置；任一预先登记的安全门限失败就不接受。

#### `run_selected_ransac_reference.py`

直接运行已经选定的固定RANSAC配置，不再重新调参。输出可复现的RANSAC参考结果。

## 6. `surf_bev/` 中每个文件的作用

#### `__init__.py`

声明`surf_bev`为Python包。

#### `detectors.py`

封装CLRNet初始化、权重加载、Windows无临时配置文件加载和推理输出转换。输入一张图像，输出若干有序二维候选车道点。

#### `geometry.py`

读取KITTI标定与pose；完成图像点到平坦地面的IPM米制`X/Z`变换，以及不同帧之间的pose坐标变换。

#### `point_export.py`

把每个米制点的帧、左右侧、原索引和保留/删除原因保存为CSV/JSON，使去噪过程可以追查。

#### `optimized_ransac.py`

固定的改进RANSAC实现：左右分别拟合`X=f(Z)`，归一化Z，支持随距离增长的残差阈值、距离分段平衡和局部重拟合。

#### `temporal_denoise.py`

保守的跨帧留一统计工具：某帧点只与其他帧同侧几何比较，避免用本帧自身证明本帧正确。它是诊断工具，不是最终长弯道主模型。

## 7. 顶层文件

#### `README_SURF_FINAL_ZH.md`

本文件；工作区入口、逐文件说明和运行方法。

#### `SURF_FINAL_PIPELINE_ZH.md`

最终流程的输入输出、算法边界、参数和历史运行说明。

#### `SURF_FINAL_DECISIONS_ZH.md`

对组会后八个问题的最终决策、Sequence 01正式阈值、模型选择结论和论文依据。

#### `requirements-windows.txt`

Windows工作站Python依赖版本清单。当前工作站继续使用已经验证的`surf2026-win`环境。

#### `SOURCE_MANIFEST.json`

由工作区创建脚本自动生成，记录源提交、CLRNet版本、复制内容和排除项，用于确认工作区来源。

## 8. 正式运行

### 8.1 重跑155帧正式结果

```powershell
& .\scripts\run_surf_final_sequence01_windows.ps1 `
  -EnvName "surf2026-win" `
  -DatasetRoot "F:\BaiduNetdiskDownload\kitti\odometry" `
  -Device cuda
```

### 8.2 跑完整Sequence 01

```powershell
& .\scripts\run_surf_final_sequence01_full_windows.ps1 `
  -EnvName "surf2026-win" `
  -DatasetRoot "F:\BaiduNetdiskDownload\kitti\odometry" `
  -Device cuda
```

全序列会运行1101次CLRNet推理，时间明显长于155帧。运行期间看到`torch.load FutureWarning`不代表失败；应以最后的`SURF FULL SEQUENCE 01 EXPERIMENT FINISHED`和输出目录中的状态JSON为准。

### 8.3 跑Sequence 00–10全量实验

```powershell
Set-Location F:\surf_final
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

& .\scripts\run_surf_final_all_sequences_windows.ps1 `
  -EnvName "surf2026-win" `
  -DatasetRoot "F:\BaiduNetdiskDownload\kitti\odometry" `
  -Device cuda
```

默认约处理23,201帧，所需时间远长于单个Sequence。脚本会复用已完成的Sequence 01，并在每个Sequence结束后写入批量进度。运行中断时，从控制台记录输出根目录，然后增加`-OutputRoot "该目录"`重新执行即可继续；只有明确需要全部重算时才加`-ForceRerun`。

## 9. 每次结果目录

- `01_lane_detection_and_tracking/`：CLRNet候选、左右关联和点数据；
- `02_pose_curvature/`：逐帧曲率、阈值、直道/过渡/弯道分段和图；
- `03_curve_models_and_fusion/`：模型比较、留出误差、窗口曲线、接口和融合结果；其中`blended_lane_nodes.csv`是只包含观测支持的融合曲线采样点；
- `04_occlusion_hypotheses/`：缺口审计及可选虚线假设；其中`final_lane_nodes_with_hypotheses.csv`同时保存观测点和通过门限的低置信度补齐点，并保留来源、置信度字段；
- `05_review_bundle/`：便于上传核验的小结果包，同时包含上述两份最终坐标CSV；
- `FINAL_STATUS.json`：完成状态和限制；
- `FINAL_METRICS.json`：覆盖率、模型选择、接口通过率和桥接统计；
- `final_seq...review_bundle.zip`：上传和汇报整理用压缩包。

## 10. 当前正式结论

Sequence 01帧851–1005的直道阈值为`0.0021036211 1/m`，弯道阈值为`0.0033192514 1/m`。左侧观测覆盖100%，双侧覆盖94.2%，窗口侧拟合完成率93.3%，接口通过率96.2%。

过渡段4/4组B样条留出RMSE更低；弯道20组中19组B样条更低，平均比多项式低约0.181 m。最终选择18组B样条和2组多项式，说明合理方案是“直道固定多项式、弯道按同输入留出误差比较”，而不是全局只用一种模型。

以上误差衡量对CLRNet/IPM观测的一致性，不代表现实世界车道定位准确率。
