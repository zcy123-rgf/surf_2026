# 100帧弯道改进实验（Windows工作站）

## 本次针对什么问题

1. 旧`temporal_ego`每帧都按P2主点把候选分为左右两组。急弯时两条可见边界可能同时位于图像中心同一侧，造成不必要的无效帧。
2. 旧B样条左右独立拟合。单侧观测稀疏或异常时，可能出现局部鼓包；程序只能事后检查交叉，不能从模型结构上阻止交叉。
3. SemanticKITTI类别60在已选Sequence 07帧415--514和Sequence 03帧29--128中均为0，不能用它生成绝对精度。实验继续报告有效帧率、跨帧连续性和留出帧一致性。

## 新增方法

- `temporal_joint`：首个有效帧仍使用保守的本车道左右初始化；后续不再强制候选分居图像中心两侧，而是在全部候选中联合选择两个不重复、底部横坐标次序正确、且分别通过pose预测距离门限的候选。
- `coupled B-spline`：不再完全独立拟合左右两侧，而是拟合中心偏移`c(s)`和对数宽度`log(w(s))`，再恢复`left=c-w/2`、`right=c+w/2`。宽度由数据估计并平滑，且由于`w=exp(log(w))`，左右曲线在模型结构上不会交叉。
- 所有旧方法均保留：同一次运行会产生`ego_adjacent`、`temporal_ego`、`temporal_joint`，以及多项式、独立B样条、耦合B样条三组结果。

## 在工作站运行

在Anaconda PowerShell Prompt中进入已经更新的项目根目录：

```powershell
Set-Location F:\2026_surf\clean_workstation_release_20260804_004638
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

& ".\scripts\run_curved_improvement_windows.ps1" `
  -EnvName "surf2026-win" `
  -DatasetRoot "F:\BaiduNetdiskDownload\kitti\odometry" `
  -ClrnetRoot "F:\2026_surf\CLRNet" `
  -Device cuda `
  -Purpose primary
```

`-Purpose primary`只运行首选Sequence 07帧415--514。首选段完成并确认后，可将它改为`all`，同时运行Sequence 03备用段。脚本每次自动建立新的时间戳输出目录，不会覆盖以前的output。

## 输出结构

新结果位于：

```text
workstation_outputs\curved_improvement_时间戳\
```

每个路段内主要目录：

- `01_ego_adjacent`：逐帧基线；
- `02_temporal_ego`：旧时序分侧方法；
- `03_temporal_joint`：新联合时序方法；
- `04_identity_old`：逐帧与旧时序对比；
- `05_identity_joint`：逐帧与新时序对比；
- `07_polynomial`：Frenet多项式；
- `08_independent_bspline`：旧左右独立B样条；
- `09_coupled_bspline`：新中心/正宽度耦合B样条；
- `10_improvement_summary`：统一CSV、JSON和对比图。

运行结束后，终端会显示：

```text
Upload this result bundle: ...\curved_improvement_bundle.zip
```

将这个ZIP复制到本地并上传即可继续核验和更新PPT。

## 如何判断改进是否成立

不能只看一张拟合图。至少同时检查：

1. `temporal_joint`有效帧数是否高于旧`temporal_ego`；
2. 覆盖增加时，连续性P90和Max是否仍处于可接受范围；
3. 耦合B样条的留出帧Mean RMSE是否不劣于独立B样条；
4. 新图中的急弯鼓包是否减小；
5. `width_check`必须保持`curves_cross=false`。

如果覆盖率提高但连续性明显变差，不能采用联合匹配；如果曲线不交叉但留出误差明显增大，也不能声称耦合B样条更优。留出误差仍然只代表CLRNet/IPM点的一致性，不是相对真实车道的绝对准确率。
