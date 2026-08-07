# Windows 工作站 PyCharm 复现指南

本指南用于在 Windows 工作站上从代码重新生成结果。它只说明已经实际验证过的流程。

当前统一运行方式：在 PyCharm 中直接运行 `pycharm_entrypoints` 下的 `00`—`03` 四个 Python 文件。PyCharm 会自动为当前文件建立运行配置，无需继续使用以前手工填写参数的四个配置，也无需使用 `rundemo`。

## 1. 打开正确的 PyCharm 项目

在 PyCharm 中选择 `File -> Open`，打开：

```text
F:\2026_surf\clean_workstation_release_20260804_004638
```

下文用 `<PROJECT_ROOT>` 表示这个目录。

## 2. 选择解释器

工作站已配置好 Conda 环境 `surf2026-win`。组员打开项目后先查看 PyCharm 右下角：如果已经显示 `Python 3.10 (surf2026-win)`，可以直接运行；如果显示其他解释器，就在当前项目中选择一次 `surf2026-win`。

进入 `File -> Settings -> Project -> Python Interpreter`，在解释器列表中选择：

```text
C:\Users\IR713\anaconda3\envs\surf2026-win\python.exe
```

该解释器已包含本项目验证过的 Python 3.10、PyTorch、CUDA、OpenCV、SciPy 和 Matplotlib 环境。组员远程使用同一台工作站、同一 Windows 账号和同一项目目录时，通常会直接继承现有选择，只需核对右下角显示。

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

组员在 PyCharm 左侧展开 `pycharm_entrypoints`，按下面的方法运行：

1. 双击打开对应 Python 文件；
2. 在编辑区内右键，选择 `Run '文件名'`；
3. 第一次运行后，PyCharm 会自动生成与该文件同名的运行配置，并显示在右上角；
4. 以后可以继续右键运行，也可以在右上角选择这个同名配置后点击绿色三角；
5. 需要查看内部过程时，在 `scripts` 或 `surf_bev` 中设置断点，再选择 `Debug '文件名'`。

这些入口会在同一 Python 进程中调用现有核心代码，因此断点可以进入真实算法，而不是只看到一个外部命令。

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

## 4. 四个入口与主体代码的关系

### 4.1 先理解三层代码结构

`pycharm_entrypoints` 下的四个文件不是算法主体，而是便于组员在 PyCharm 中直接运行的薄入口。完整调用关系是：

```text
pycharm_entrypoints/*.py
    负责：工作站路径、固定参数、运行顺序、自动新建输出目录
                    |
                    v
scripts/*.py
    负责：读取输入、依次组织各模块、评价、绘图、写 JSON/CSV
                    |
                    v
surf_bev/*.py 和 CLRNet/clrnet/*
    负责：网络推理、IPM、pose 变换、RANSAC、曲线拟合和点导出等核心实现
```

入口通过 `pycharm_entrypoints/common.py` 中的 `run_main()` 在同一个 Python 进程里调用 `scripts` 的 `main()`。所以在 PyCharm 中使用 `Debug` 时，可以从入口逐步进入真正的主体代码，不是启动一个看不见内部过程的外部程序。

`common.py` 还负责：

- 确定项目、KITTI、标定、pose 和 CLRNet 路径；
- `new_output()` 为每次运行生成新的时间戳目录；
- `record_latest_pipeline()` 记录最近一次有效的 01 输出；
- `latest_pipeline_files()` 为 02/03 找到有效的上游点数据；
- `require_file()` 和 `require_directory()` 在运行前检查输入。

### 4.2 阶段 00：环境和 CLRNet 实际推理检查

调用链：

```text
pycharm_entrypoints/00_check_environment.py
  -> scripts/check_windows_env.py : main()
  -> surf_bev/detectors.py : CLRNetLaneDetector
  -> CLRNet/clrnet/models/registry.py : build_net()
  -> CLRNet 网络前向推理
  -> CLRNet/clrnet/models/heads/clr_head.py : get_lanes()
  -> surf_bev/detectors.py : _prediction_to_polyline()
```

各部分功能：

- `00_check_environment.py`：只传入 `--device cuda --run-clrnet`；
- `check_windows_env.py`：检查 Python、PyTorch、torchvision、CUDA、GPU、项目内 CLRNet 和权重，并读取一张样例图；
- `CLRNetLaneDetector.__init__()`：读取 CLRNet 配置，建立网络，加载 `weights/culane_r18.pth`，切换到 `eval()`；
- `CLRNetLaneDetector.detect()`：预处理图像，在 `torch.no_grad()` 中执行推理；
- `CLRHead.get_lanes()`：对车道/非车道分类分数做 softmax，按置信度阈值筛选，再做 NMS，得到保留的车道候选参数；
- `_prediction_to_polyline()`：将每个候选中的有效横坐标、起点和长度解码为原图像坐标系中的有序二维点 `(u,v)`。

查看或修改位置：

- 环境检查内容：`scripts/check_windows_env.py`；
- CLRNet 与本项目的接口、图像预处理和二维点解码：`surf_bev/detectors.py`；
- CLRNet 网络结构：`CLRNet/clrnet/models/`；
- 置信度、NMS 和候选解码：`CLRNet/clrnet/models/heads/clr_head.py`；
- 模型配置：`CLRNet/configs/clrnet/clr_resnet18_culane.py`；
- 权重：`CLRNet/weights/culane_r18.pth`。

阶段 00 只验证模型能否正确加载并完成一次推理，不生成正式的五帧实验结果。

### 4.3 阶段 01：原图到 pose 对齐米制点

调用链：

```text
pycharm_entrypoints/01_generate_pose_aligned_points.py
  -> scripts/run_full_point_pipeline.py : main()
     -> surf_bev/detectors.py : CLRNetLaneDetector.detect()
     -> run_full_point_pipeline.py : select_outer_two()
     -> surf_bev/geometry.py : image_to_ground_ipm()
     -> surf_bev/geometry.py : transform_lane_points_by_pose()
     -> run_full_point_pipeline.py : weighted_raster_fusion()
     -> surf_bev/point_export.py : 点坐标和每点保留/拒绝决定导出
```

实际处理步骤：

1. 入口固定读取 KITTI Odometry Sequence 00 的 `000000`—`000004` 五帧、`calib.txt` 和 `poses/00.txt`；
2. `CLRNetLaneDetector.detect()` 对每帧输出若干候选，每个候选是原图像像素坐标中的有序二维点；
3. `select_outer_two()` 按候选靠近图像底部位置的横坐标排序，暂时选择最左和最右候选，并记录被排除候选；这是当前项目的候选选择规则，不等于 CLRNet 原生输出了“本车道左右边界”语义；
4. `image_to_ground_ipm()` 使用内参矩阵 `K`、相机高、俯仰角和平坦路面假设，让每个图像点的相机射线与地面相交，得到当前帧地面坐标 `(X,Z)`，单位为米；
5. `transform_lane_points_by_pose()` 使用
   `T_ref_from_src = inv(T_world_ref) @ T_world_src`，把每帧点变换到参考图像相机坐标系；当前入口的参考帧是第 4 帧；
6. `weighted_raster_fusion()` 只在生成显示图时把米制点栅格化并使用 `0.2,0.4,0.6,0.8,1.0` 权重；这些权重不参与 pose 坐标变换；
7. `point_export.py` 保留共同的原始米制点顺序，并导出不同诊断方法对每个点的保留/拒绝决定；
8. 最后写出输入哈希、标定、pose、检测候选、二维点、米制点、图像和运行状态。

主体文件职责：

- `scripts/run_full_point_pipeline.py`：完整流程编排、左右候选选择、范围过滤、栅格显示、诊断方法和文件输出；
- `surf_bev/detectors.py`：CLRNet 模型适配与图像二维点输出；
- `surf_bev/geometry.py`：KITTI 标定读取、点式 IPM、相机/地面坐标转换和 pose 对齐；
- `surf_bev/point_export.py`：为每个源点建立可追溯编号，并导出 JSON/CSV；
- `surf_bev/temporal_denoise.py`：01 中保留的时序候选诊断，不是当前默认 RANSAC；
- `surf_bev/RANSAC.py` 和 `run_full_point_pipeline.py` 中的旧直线 RANSAC：历史诊断基线，不是阶段 02 的当前实现。

阶段 01 最重要的两个机器可读输出：

- `detected_lane_points.json`：原图像坐标中的 CLRNet 候选和已选两条候选；
- `aligned_lane_points.json`：变换到参考帧 4 坐标系后的左右米制点。

### 4.4 阶段 02：当前保留的改进 RANSAC

调用链：

```text
pycharm_entrypoints/02_run_optimized_ransac.py
  -> scripts/run_selected_ransac_reference.py : run()
     -> scripts/evaluate_denoise_methods.py : 读取、IPM、pose 对齐和评价函数
     -> surf_bev/optimized_ransac.py : SideAwarePolynomialRansac
     -> run_selected_ransac_reference.py : 坐标、图片、指标和审计导出
```

这里没有再次运行 CLRNet。入口读取阶段 01 保存的 `detected_lane_points.json`，再用相同标定和 pose 重建米制对齐点。这样 RANSAC 的输入来源和坐标转换过程可以单独审计。

`surf_bev/optimized_ransac.py` 是当前 RANSAC 主体，主要过程是：

1. 根据阶段 01 保存的 `side` 字段，把左、右车道分开拟合；
2. 对每一侧拟合三次多项式 `X=f(Z)`；
3. 对 `Z` 归一化，减小远距离数值过大造成的病态；
4. 每个假设至少覆盖 `10 m` 的纵向范围，避免只抽到局部相邻点；
5. 点到模型的当前残差为同一 `Z` 下的横向差 `|X-X_model(Z)|`；
6. 内点阈值随距离增长：
   `threshold(Z)=min(0.35+0.015*max(Z-3,0), 0.80)`，单位为米；
7. 评分同时考虑全体内点比例和不同距离段的内点比例，降低近处密集点支配结果的风险；
8. 选择最佳假设后进行一次局部最小二乘重拟合；
9. 左右两侧掩码合并后导出去噪坐标和逐点决定。

主体文件职责：

- `scripts/run_selected_ransac_reference.py`：固定并记录本次采用的参数，调用算法，生成输出和审计 JSON；
- `surf_bev/optimized_ransac.py`：改进 RANSAC 的抽样、阈值、评分和重拟合实现；
- `scripts/evaluate_denoise_methods.py`：复用二维点读取、手工标注读取、IPM、pose 对齐、栅格显示和评价函数；
- `annotations/kitti00_first5_manual_annotations.json`：只用于防止过度删点的手工伪参考评价，不参与模型训练，也不是 KITTI 官方真值。

要修改 RANSAC 数学过程，应进入 `surf_bev/optimized_ransac.py`；要更换固定参数，应查看 `run_selected_ransac_reference.py` 中的 `SELECTED_CONFIG`；要修改输入重建或评价，应进入 `evaluate_denoise_methods.py`。

### 4.5 阶段 03：多项式与 B 样条曲线比较

调用链：

```text
pycharm_entrypoints/03_compare_curve_models.py
  -> scripts/compare_polynomial_bspline.py : run()
     -> scripts/analyze_lane_curve_hierarchy.py
        -> select_smoothing()
        -> polynomial_lofo()
        -> fit_sides()
     -> scripts/fit_first5_two_curves.py
        -> aggregate_equal_frame_bins()
        -> fit_robust_spline()
        -> leave_one_frame_out()
        -> export_fit()
```

阶段 03 直接读取阶段 01 的 `aligned_lane_points.json`，不读取阶段 02 的 RANSAC 输出。当前用途是公平比较曲线模型，不是“RANSAC 后再拟合”。

实际处理过程：

1. 左右车道始终分开；
2. `aggregate_equal_frame_bins()` 按 `Z` 分箱，每帧在每个箱中只贡献一个中位数，再跨帧取中位数，避免某帧点数更多就获得更大权重；
3. `polynomial_lofo()` 分别比较 1、2、3 次鲁棒多项式 `X=f(Z)`；
4. `fit_robust_spline()` 使用弦长参数表示曲线，同时拟合 `X(u),Z(u)` 的三次参数 B 样条；
5. Huber-IRLS 根据当前残差反复降低离群聚合点的权重，但最低权重保留为 `0.05`；
6. `select_smoothing()` 在给定平滑参数网格中做留一帧评价，并使用一标准误差规则选择更平滑且评价没有明显变差的结果；
7. 留一帧评价每次拿出一帧，用其余帧拟合，再计算被拿出帧的点到曲线距离；它反映对当前 CLRNet/IPM 点的跨帧一致性，不是官方车道线准确率；
8. 最后导出左右曲线坐标、多项式/B样条对比图、每折指标和 `RESULT.json`。

主体文件职责：

- `scripts/compare_polynomial_bspline.py`：模型比较的总编排和结果汇总；
- `scripts/analyze_lane_curve_hierarchy.py`：平滑参数选择、鲁棒多项式、留一帧评价及分段稀疏融合研究函数；当前 03 只调用其中模型比较部分；
- `scripts/fit_first5_two_curves.py`：左右点分组、等帧权重分箱、鲁棒 B 样条、曲线距离、绘图和坐标导出。

阶段 03 当前明确停止在“多项式/B样条比较和两条曲线导出”，不执行后续的分段特征点再融合。

### 4.6 在 PyCharm 中顺着调用链看代码

1. 从 `pycharm_entrypoints` 的入口开始；
2. 对导入的模块名或函数名按住 `Ctrl` 并单击，进入 `scripts`；
3. 在 `scripts` 中继续对 `CLRNetLaneDetector`、`image_to_ground_ipm`、`transform_lane_points_by_pose`、`SideAwarePolynomialRansac` 等名称按 `Ctrl+B`；
4. 在目标函数左侧单击设置断点；
5. 回到入口文件，右键选择 `Debug`；
6. 查看 Variables 中的 `lanes`、`ground`、`aligned`、`keep`、`model_rows` 等中间变量。

修改规则：工作站路径和默认实验参数放在 `pycharm_entrypoints`；模块调用顺序、输入输出和评价放在 `scripts`；数学和几何核心放在 `surf_bev`；CLRNet 网络内部放在独立的 `CLRNet`。这样可以避免把同一算法复制成多份。

## 5. PyCharm 运行配置怎么处理

组内统一使用右键运行 `pycharm_entrypoints` 后由 PyCharm 自动生成的四个同名配置：

- `00_check_environment`：验证环境和 CLRNet；
- `01_generate_pose_aligned_points`：生成检测点、IPM 点和 pose 对齐点；
- `02_run_optimized_ransac`：运行当前保留的改进 RANSAC；
- `03_compare_curve_models`：比较多项式和 B 样条。

以前的 `00_check_windows_env`、`01_full_point_pipeline`、`02_optimized_ransac`、`03_polynomial_bspline` 和 `rundemo` 属于此前手工参数或样例运行记录。保留它们不会改变代码和结果；为了让组员只看到当前统一入口，可以这样整理：

1. 打开 `Run -> Edit Configurations`；
2. 在左侧依次选中上述旧配置；
3. 点击左上角减号 `-`；
4. 点击 `Apply -> OK`；
5. 回到 `pycharm_entrypoints`，依次右键运行 `00`—`03`，PyCharm 会自动建立当前四个配置。

删除运行配置只会清理 PyCharm 的启动记录，不会删除 Python 文件、CLRNet、数据或实验结果。

四个新入口会自动生成带微秒时间戳的新输出目录，因此每次可直接重跑，不需要手工改 `--output-dir`。如果为了单独调试而直接运行 `scripts` 中的核心 CLI，才需要提供完整参数，并为 `--output-dir` 指定新目录。

## 6. 目录与代码职责

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

## 7. 常见问题

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

## 8. 调试原则

1. 在入口文件、`scripts` 或 `surf_bev` 中设置断点，使用 PyCharm `Debug`；
2. 先确认输入文件存在，再检查算法；
3. 入口会为每次运行自动生成新输出目录，保留控制台日志和 JSON 审计文件；
4. 查看 `STATUS.json`、`RESULT.json`、`result.json` 和输入 SHA-256，不只看图片；
5. 区分候选车道、最终左右车道、米制点和栅格显示图；
6. 不把内部一致性、像素重叠率或手工伪参考一致性称为官方准确率；
7. 环境已经通过时，不随意重新安装依赖或重新运行兼容补丁。

## 9. 最短复现清单

1. 打开 `<PROJECT_ROOT>`；
2. 核对右下角为 `Python 3.10 (surf2026-win)`；
3. 运行 `pycharm_entrypoints/00_check_environment.py`；
4. 运行 `pycharm_entrypoints/01_generate_pose_aligned_points.py`；
5. 分别运行 `pycharm_entrypoints/02_run_optimized_ransac.py` 和 `03_compare_curve_models.py`；
6. 核对退出代码、JSON、点坐标和图片；
7. 重跑时由入口自动创建时间戳目录，旧结果保持不变。

## 10. 扩展到 0–1000 帧的候选实验

扩大候选图片范围时，使用：

```text
scripts/run_extended_1000_curve_models_windows.ps1
```

该总控脚本默认扫描第 0–1000 帧，输出最多 100 个候选连续路段，再只对指定候选分别调用：

- `scripts/fit_extended_polynomial.py`：多项式独立主体；
- `scripts/fit_extended_bspline.py`：B 样条独立主体；
- `scripts/extended_curve_model_common.py`：共享读取和已审核的 pose/Frenet 数据准备，不包含曲线模型。

完整命令、候选排名、复用扫描和输出说明见：

```text
README_EXTENDED_1000_ZH.md
```
