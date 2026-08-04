# Windows 工作站 PyCharm 复现指南

本指南用于在 Windows 工作站上从代码重新生成结果。它只说明已经实际验证过的流程。

## 1. 打开正确的 PyCharm 项目

在 PyCharm 中选择 `File -> Open`，打开：

```text
F:\2026_surf\clean_workstation_release_20260804_004638
```

下文用 `<PROJECT_ROOT>` 表示这个目录。复制参数时必须把 `<PROJECT_ROOT>` 替换成实际绝对路径，PyCharm 不会自动展开这个占位符。

## 2. 选择解释器

工作站已配置好 Conda 环境 `surf2026-win`，组员只需要在当前 PyCharm 项目中选择它，不需要重新创建环境或安装依赖。

进入 `File -> Settings -> Project -> Python Interpreter`，在解释器列表中选择：

```text
C:\Users\IR713\anaconda3\envs\surf2026-win\python.exe
```

该解释器已包含本项目验证过的 Python 3.10、PyTorch、CUDA、OpenCV、SciPy 和 Matplotlib 环境。PyCharm 会按项目和用户记录解释器选择，因此第一次打开该项目时需要确认一次。

只有当列表中没有 `surf2026-win` 时，才选择 `Add Interpreter -> Add Local Interpreter -> Conda Environment -> 使用现有环境`，并指定上述 `python.exe`。

直接运行 `pycharm_entrypoints` 中的 Python 文件时，程序会自动确定项目根目录、输入参数和时间戳输出目录。组员选好 `surf2026-win` 后即可直接运行，无需填写 Parameters、Interpreter options 或 Working directory。入口也会禁止向 CLRNet 源码目录写入 Python 字节码缓存。

## 3. 直接运行四个 Python 入口

数据依赖关系：

```text
00 环境和 CLRNet 检查
           |
01 原图 -> CLRNet -> IPM -> pose 对齐
           |
   +-------+----------------+
   |                        |
02 改进 RANSAC       03 多项式/B样条比较
```

`02` 和 `03` 是并列实验：`03` 不依赖 `02` 的结果。

组员在 PyCharm 左侧展开 `pycharm_entrypoints`，打开对应 Python 文件，然后右键选择 `Run`。这些入口会在同一 Python 进程中调用现有核心代码，因此可以用 `Debug` 运行，并在 `scripts` 或 `surf_bev` 中设置断点。

输入路径统一写在 `pycharm_entrypoints/common.py`：

- KITTI 默认根目录：`F:\BaiduNetdiskDownload\kitti\odometry`；
- CLRNet 默认目录：`<PROJECT_ROOT>\CLRNet`；
- PyCharm 结果根目录：`<PROJECT_ROOT>\pycharm_outputs`。

### 3.1 `00_check_environment.py`

作用：检查解释器、PyTorch、CUDA、GPU、CLRNet 代码、权重，并实际推理一张样例图。

```text
<PROJECT_ROOT>\pycharm_entrypoints\00_check_environment.py
```

该入口调用 `scripts/check_windows_env.py`，已内置 `--device cuda --run-clrnet`，无需手工填写 Parameters。

成功标志：退出代码为 `0`，并且控制台包含：

```json
"cuda_available": true,
"gpu": "NVIDIA GeForce RTX 3090",
"clrnet_submodule": true,
"clrnet_checkpoint": true,
"detected_lanes": 3
```

`detected_lanes` 的具体数量可以随图片和模型输出变化；只要存在该字段且没有 Traceback，就说明模型完成了实际推理。

### 3.2 `01_generate_pose_aligned_points.py`

作用：从 KITTI 00 前五帧原图开始，完成 CLRNet 候选点、左右车道选择、点式 IPM、KITTI pose 对齐和中间数据导出。它还保留旧 RANSAC/时序方法的诊断输出，但这些不是当前默认最终方法。

```text
<PROJECT_ROOT>\pycharm_entrypoints\01_generate_pose_aligned_points.py
```

该入口调用 `scripts/run_full_point_pipeline.py`，自动生成时间戳输出目录，并记录最新有效上游结果。

成功标志：控制台输出 `status: complete`，并生成：

```text
pycharm_outputs\01_pipeline_时间戳\00_metadata\detected_lane_points.json
pycharm_outputs\01_pipeline_时间戳\00_metadata\aligned_lane_points.json
pycharm_outputs\LATEST_PIPELINE.json
```

- `detected_lane_points.json`：CLRNet 的图像二维有序点；
- `aligned_lane_points.json`：经过 IPM 和 pose 对齐后的米制 X/Z 点。

### 3.3 `02_run_optimized_ransac.py`

作用：读取 CLRNet 二维点，使用同一标定和 pose 重建对齐点，然后左右分开运行固定的安全优先改进 RANSAC。该入口不使用 B 样条，也不调用 CLRNet 模型。

```text
<PROJECT_ROOT>\pycharm_entrypoints\02_run_optimized_ransac.py
```

它自动读取 `LATEST_PIPELINE.json`。如果该文件还未生成，则自动查找 `pycharm_outputs` 中最新且包含完整 metadata 的上游目录。

成功标志：退出代码为 `0`，控制台包含：

```text
"status": "complete"
"method_family": "RANSAC only; no B-spline or temporal denoiser"
```

关键输出：

```text
02_ransac_时间戳\00_audit\result.json
02_ransac_时间戳\01_points\metric_plot.png
02_ransac_时间戳\02_fusion\weighted_binary.png
02_ransac_时间戳\03_data\denoised_points.json
```

当前配置是安全优先参考配置，不是 KITTI 官方真值意义上的全局最优。若 `kept == input`，含义是没有点超过本次阈值，不应把它描述成明显去噪。

### 3.4 `03_compare_curve_models.py`

作用：读取 pose 对齐点，比较鲁棒 1/2/3 次多项式和鲁棒三次参数 B 样条，并导出左右两条平滑曲线。该入口不依赖 RANSAC 结果，不运行稀疏分段再融合。

```text
<PROJECT_ROOT>\pycharm_entrypoints\03_compare_curve_models.py
```

它自动读取与配置 02 相同的最新上游结果，但不读取 RANSAC 输出。

成功标志：退出代码为 `0`，控制台包含：

```text
"status": "complete"
"scope": "polynomial/B-spline comparison only; no sparse refusion"
```

关键输出：

```text
03_curve_models_时间戳\model_comparison.png
03_curve_models_时间戳\bspline_left_right_curves.png
03_curve_models_时间戳\model_lofo.csv
03_curve_models_时间戳\RESULT.json
```

模型比较指标是对 CLRNet/IPM 派生点的留出帧内部一致性，不是对官方车道线真值的准确率。

## 4. PyCharm 配置与自动输出

当前工作站上已创建的 `00_check_windows_env`、`01_full_point_pipeline`、`02_optimized_ransac` 和 `03_polynomial_bspline` 配置可以继续使用。组员远程进入同一 Windows 账号、打开同一项目时，通常可以直接看到它们。

新增的 `pycharm_entrypoints` 是组内推荐入口：它们自动生成带微秒时间戳的新输出目录，因此每次可直接重跑，不需要手工改 `--output-dir`。

如果为了单独调试而直接运行 `scripts` 中的核心 CLI，仍需提供完整参数，并为 `--output-dir` 指定新目录。

## 5. 目录与代码职责

### `CLRNet/`

项目内独立的 CLRNet 代码、Windows 兼容层和 `weights/culane_r18.pth`。`surf_bev/detectors.py` 只在检测阶段加载它。正常推理使用 `model.eval()` 和 `torch.no_grad()`，不会训练或保存权重。

如果建立工作站公共模型目录，在 `pycharm_entrypoints/common.py` 中将 `CLRNET_ROOT` 设为：

```text
CLRNET_ROOT = Path(r"F:\2026_surf_models\CLRNet_culane_r18_windows")
```

公共目录先完成权重 SHA-256 一致性检查，再运行入口 01 验证。入口 02/03 本来就不调用 CLRNet。项目内已验证的 CLRNet 副本作为回退备份保留。

### `annotations/`

前五帧手工伪参考。用于检查过度删点，不是官方真值。

### `data/`

少量环境检查和原始 demo 样例，不是完整 KITTI 数据集。

### `scripts/`

实验入口和流程编排：

- `check_windows_env.py`：环境与 CLRNet 冒烟测试；
- `run_full_point_pipeline.py`：统一上游点流程；
- `run_selected_ransac_reference.py`：改进 RANSAC 入口；
- `compare_polynomial_bspline.py`：多项式/B样条入口；
- `evaluate_denoise_methods.py`：RANSAC 入口复用的数据、评价和画图函数；
- `fit_first5_two_curves.py`：B 样条拟合函数；
- `analyze_lane_curve_hierarchy.py`：当前仅复用其中的模型比较函数；
- `kitti00_workstation_input.ps1`：PowerShell 下定位和校验 KITTI 数据；
- `setup_clrnet_windows.ps1`：环境安装/修复脚本。环境已通过时不要重复运行。

### `pycharm_entrypoints/`

组内直接运行的四个入口：

- `00_check_environment.py`：调用环境和 CLRNet 检查代码；
- `01_generate_pose_aligned_points.py`：调用统一上游点流程；
- `02_run_optimized_ransac.py`：调用改进 RANSAC；
- `03_compare_curve_models.py`：调用多项式/B样条比较。

`common.py` 集中保存数据路径、CLRNet 路径、输出命名和最新上游结果查找逻辑。四个入口只组织参数，算法实现仍在 `scripts` 和 `surf_bev`，不会形成多份算法副本。

### `surf_bev/`

被入口调用的算法库，不应逐个点击运行：

- `detectors.py`：CLRNet 封装和二维候选点输出；
- `geometry.py`：IPM、标定和 pose 变换；
- `optimized_ransac.py`：当前固定的改进 RANSAC；
- `point_export.py`：逐点决定和坐标导出；
- `RANSAC.py`：原始旧 RANSAC 参考，不是当前默认实现；
- `temporal_denoise.py`：历史诊断方法，不是当前默认实现；
- `pipeline.py`、`visualization.py`：原始 demo 与绘图支持。

### `workstation_release/`

PowerShell 一键入口。PyCharm 用户不需要把 `.ps1` 当作 Python 运行：

- `01_ransac_reference/run_windows.ps1`：命令行完整 RANSAC 流程；
- `02_curve_models/run_windows.ps1`：命令行多项式/B样条流程。

### 输出目录

- `pycharm_outputs/`：PyCharm 运行结果；
- `clean_outputs/`：PowerShell 一键入口结果；
- `.runtime/tmp/`：Windows 权限兼容所需临时文件，不是实验结果。

两类输出不要混用。结果文件不提交到代码仓库。

## 6. 常见问题

### `torch.load(... weights_only=False)` FutureWarning

这是 PyTorch 对未来默认行为和不可信 pickle 的安全提醒，不是运行失败，也不表示权重被修改。只使用来源可追溯、哈希已记录的权重；不要加载来历不明的 `.pth`。

### PyCharm 显示 `matplotlib`、`scipy` 无法解析

通常是解释器选成了 base。重新选择 `surf2026-win`，等待 PyCharm 重建索引。不要因为编辑器红线直接重装依赖。

### `cuda_available` 为 false

检查 PyCharm 配置所用解释器是否确实是 `surf2026-win`，再在同一解释器中检查 `torch.cuda.is_available()`。不要根据系统显示的 CUDA 13.2 去替换 PyTorch 自带的 CUDA 12.1 运行时。

### `PermissionError` 指向 TEMP

Working directory 必须是 `<PROJECT_ROOT>`。`check_windows_env.py` 会使用 `<PROJECT_ROOT>/.runtime/tmp`，不要改回受限系统临时目录。

### 应该打开哪个文件运行

| 目标 | 打开并运行的文件 |
|---|---|
| 环境和 CLRNet 检查 | `pycharm_entrypoints/00_check_environment.py` |
| 生成 CLRNet/IPM/pose 对齐点 | `pycharm_entrypoints/01_generate_pose_aligned_points.py` |
| 改进 RANSAC | `pycharm_entrypoints/02_run_optimized_ransac.py` |
| 多项式/B样条比较 | `pycharm_entrypoints/03_compare_curve_models.py` |

`workstation_release/*.ps1` 是 PowerShell 命令行入口；PyCharm 使用上表的 Python 入口。

## 7. 调试原则

1. 在入口文件、`scripts` 或 `surf_bev` 中设置断点，使用 PyCharm `Debug`；
2. 先确认输入文件存在，再检查算法；
3. 入口会为每次运行自动生成新输出目录，保留控制台日志和 JSON 审计文件；
4. 查看 `STATUS.json`、`RESULT.json`、`result.json` 和输入 SHA-256，不只看图片；
5. 区分候选车道、最终左右车道、米制点和栅格显示图；
6. 不把内部一致性、像素重叠率或手工伪参考一致性称为官方准确率；
7. 环境已经通过时，不随意重新安装依赖或重新运行兼容补丁。

## 8. 最短复现清单

1. 打开 `<PROJECT_ROOT>`；
2. 选择 `surf2026-win`；
3. 运行 `pycharm_entrypoints/00_check_environment.py`；
4. 运行 `pycharm_entrypoints/01_generate_pose_aligned_points.py`；
5. 分别运行 `pycharm_entrypoints/02_run_optimized_ransac.py` 和 `03_compare_curve_models.py`；
6. 核对退出代码、JSON、点坐标和图片；
7. 重跑时由入口自动创建时间戳目录，旧结果保持不变。
