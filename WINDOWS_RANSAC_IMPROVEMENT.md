# Windows工作站：从空输出目录复现RANSAC改进

本入口只读取用户放在`input_data/`中的原始输入和人工伪标注，不读取仓库中的历史融合图片或实验输出。所有新结果写入已忽略的`workstation_outputs/`，不会被Git提交。

## 1. GitHub分支与文件范围

目标分支：

```text
agent/windows-ransac-improvements
```

该分支只提交：

- Python算法代码；
- Windows PowerShell运行入口；
- 单元测试；
- 使用说明。

以下内容不会作为本次改动提交：

- `results/`中的历史结果；
- `denoise_experiments/`
- `meeting_materials_*/`
- `workstation_outputs/`
- `generated_runs/`
- `input_data/`
- PNG/JPG结果图；
- 下载的官方压缩包和临时审计目录。

说明：基础分支的历史提交中原本就有`results/`目录，本次提交既不新增也不修改其中任何文件。新的Windows入口不会读取该目录；工作站运行结果只写入新的`workstation_outputs/`时间戳目录。

## 2. 工作站输入目录

在工作站准备：

```text
F:\2026_surf\input_data\kitti00_first5\
├─ images\
│  ├─ frame_000000.png
│  ├─ frame_000001.png
│  ├─ frame_000002.png
│  ├─ frame_000003.png
│  └─ frame_000004.png
├─ calib.txt
├─ poses_00_first5.txt
└─ manual_annotations.json
```

说明：

- 五张图是KITTI Odometry Sequence 00的`000000–000004`；
- `calib.txt`使用Sequence 00标定；
- `poses_00_first5.txt`是官方`poses/00.txt`前五行；
- `manual_annotations.json`是项目内部人工伪标注，只用于安全核验，不能称为KITTI车道线真值；
- CLRNet权重仍放在`CLRNet\weights\culane_r18.pth`，不提交到Git。

如果实际目录或文件名不同，可以在运行时显式传参，不需要移动历史结果。

## 3. 先跑真实流程和方法对比

在**Anaconda PowerShell Prompt**中执行：

```powershell
Set-Location F:\2026_surf
git fetch origin
git switch agent/windows-ransac-improvements
git pull --ff-only origin agent/windows-ransac-improvements
git submodule update --init --recursive

Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run_ransac_improvement_windows.ps1 -Mode Compare
```

`Compare`会：

1. 从五张原图重新运行CLRNet；
2. 重新计算点式BEV；
3. 重新读取KITTI pose并对齐到`000004`；
4. 从本次CLRNet输出重新比较未去噪、旧固定X+RANSAC及改进候选；
5. 写入一个带时间戳的新输出目录。

它不会读取旧结果目录。

## 4. 完整参数搜索与独立复核

确认`Compare`跑通后，再运行：

```powershell
.\scripts\run_ransac_improvement_windows.ps1 -Mode Full
```

`Full`在前述流程之后继续运行：

- RANSAC参数网格搜索；
- 安全约束扩展；
- 独立随机种子复核。

这一模式主要消耗CPU，运行时间明显长于`Compare`。不要因为运行较慢而重复启动多个副本。

`Full`默认也不读取任何旧审计结果。安全扩展脚本只有在用户显式传入`--prior-audit`时才会累计旧试验次数；Windows包装入口不传该参数，因此工作站运行是独立的从零审计。

## 5. 自定义输入与输出位置

```powershell
.\scripts\run_ransac_improvement_windows.ps1 `
  -Mode Compare `
  -ImageDir F:\your_data\images `
  -Calib F:\your_data\calib.txt `
  -Poses F:\your_data\poses_00_first5.txt `
  -ManualJson F:\your_data\manual_annotations.json `
  -OutputRoot F:\2026_surf\workstation_outputs\ransac_run_01
```

脚本拒绝向非空`OutputRoot`写入，从机制上避免把旧图误当成新结果。

## 6. 最先检查的输出

```text
01_from_scratch_pipeline/
  00_metadata/audit.json
  00_metadata/detected_lane_points.json
  04_pose_aligned_points/five_frame_metric_pose_fusion.png

02_ransac_method_comparison/
  comparison/main_comparison.png
  00_audit/evaluation.json
  EVALUATION_REPORT_ZH.md
```

完整模式另外检查：

```text
04_ransac_safety_expansion/FINAL_RANSAC_REPORT_ZH.md
05_ransac_heldout_audit/HELDOUT_AUDIT_REPORT_ZH.md
```

## 7. 结果解释边界

- RANSAC输入是位姿对齐后的米制点，不是BEV图片；
- `0.2–1.0`是栅格累积权重，不是RANSAC阈值；
- 人工标注是伪标注，不是KITTI官方车道线真值；
- 受控异常注入用于比较方法识别已知异常的能力，不能替代真实车道线真值；
- 所有报告必须保留输入哈希、参数、随机种子和输出目录。
