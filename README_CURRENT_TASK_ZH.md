# SURF 2026 当前任务：Windows 复现、点式 BEV、多帧融合与去噪核查

更新日期：2026-07-21  
当前工作分支：`agent/windows-workstation`

## 1. 本阶段任务

本阶段不是先追求一张更“好看”的融合图，而是建立一条可解释、可复现、可审计的处理链：

1. 在 Windows 工作站上从原始输入重新运行整套流程。
2. 对每个模块说明用途、选择理由、输入、输出、单位、坐标系和依据。
3. 明确区分车道点、折线、像素 mask、BEV 图像和米制地面点。
4. 定位当前去噪导致远处车道点消失的真正原因。
5. 在有对照实验后再选择新的去噪方法。
6. 最终解决多帧对齐、融合和可信评估问题。

本文件是任务和技术核查文档。当前阶段不修改检测、BEV、位姿、去噪或融合算法代码。

## 2. 证据规则

- CLRNet 的定义以论文和官方代码为准。
- OpenCV 变换的定义以官方文档为准。
- 项目行为以当前仓库调用链、保存数据和审计 JSON 为准。
- “重叠率”只表示多帧结果的一致程度，不表示车道检测准确率。
- 人工逐点标注是人工参考，不是真值标注；没有真值时不得报告虚构的准确率、F1 或 IoU。
- 对于尚未验证的原因使用“待验证”或“推测”，不得写成已经证实的结论。

## 3. 当前最重要的技术结论

### 3.1 CLRNet 输出的是有序车道点，不是整幅分割图

CLRNet 使用沿图像纵向等间距采样的二维点序列表示车道。项目适配器将网络预测解码成每条车道的 `N×2 float64` 原图像素坐标点 `(u,v)`：

- 预处理：[surf_bev/detectors.py](surf_bev/detectors.py#L134-L140)
- 预测到点序列：[surf_bev/detectors.py](surf_bev/detectors.py#L142-L162)
- 检测接口输出：[surf_bev/detectors.py](surf_bev/detectors.py#L164-L175)

结果图中的“线”是 `cv2.polylines` 将相邻有序点连接并加粗后的栅格化结果：[surf_bev/visualization.py](surf_bev/visualization.py#L15-L22)。

因此，后续处理应保留以下信息，而不是只保留一张画好线的图片：

- 点的浮点坐标；
- 点的先后顺序；
- 点所属的帧；
- 点所属的左/右车道；
- 原始候选编号和可获得的置信信息。

### 3.2 不需要先把点画成线再做 BEV

对 CLRNet 的稀疏车道点，可以直接执行点投影：

\[
\tilde p_{bev}=H\tilde p_{img},\qquad
(U,V)=\left(\frac{\tilde p_x}{\tilde p_w},\frac{\tilde p_y}{\tilde p_w}\right)
\]

OpenCV 将两种操作明确分开：

- `perspectiveTransform`：变换稀疏二维/三维点；
- `warpPerspective`：重采样整幅像素图像。

所以只有在需要完整彩色 BEV、稠密语义图、像素 mask 或最终展示图时，才需要 `warpPerspective` 或栅格化。位姿对齐、曲线拟合、去噪和点级融合不要求先生成 BEV 像素图。

### 3.3 “点坐标是像素单位”不等于“已经栅格化成像素图”

CLRNet 点最初是原图坐标 `(u,v)`，单位是图像像素，但它们仍然是稀疏浮点坐标，不是一张二值图。

点式 BEV 有两种常见输出定义：

1. 使用 `H` 将原图点变成 BEV 画布坐标 `(U,V)`。如果 `H` 的目标四点以画布像素定义，则输出仍是 BEV 像素坐标点。
2. 使用相机内参和路面模型做射线—平面相交，直接得到米制地面点 `(X,Z)`。当前仓库的 [surf_bev/geometry.py](surf_bev/geometry.py#L50-L72) 采用这一形式。

在平坦路面和固定相机模型成立时，两种形式可以由单应关系互相表达。区别主要在于输出坐标定义和单位。

### 3.4 推荐采用“点优先、最后栅格化”的主流程

需要先区分“当前成品实际做法”和“下一步建议做法”：

- 当前成品：位姿对齐和RANSAC处理米制点，随后每帧点集分别栅格化成 `800×800` mask，再进行像素权重累加。
- 建议基线：位姿对齐、去噪和多帧汇总都保留为点/曲线表示，仅在最终输出和像素指标计算时栅格化一次。

下面流程是待实验验证的建议主线，不是对当前代码行为的虚假描述。

```text
原始图像
  -> CLRNet
  -> 每条车道的有序图像点 (u,v)
  -> 选择并保留左右车道身份
  -> 点式 IPM / 单应投影
  -> 米制地面点 (X,Z)
  -> 位姿对齐到参考帧
  -> 点级去噪或稳健曲线估计
  -> 点级多帧融合
  -> 最后栅格化为 BEV mask / 热力图 / 成品图
```

这样做的优点是：

- 不会过早引入取整、插值、抗锯齿和线宽；
- 保留亚像素坐标与车道拓扑关系；
- 位姿和去噪始终在明确的米制坐标系中工作；
- 可以分别分析不同距离、不同帧和左右车道；
- 最终仍然可以生成与上周结果相同风格的 BEV 成品图。

注意：点优先不等于把一条车道打散成无序点云。车道仍应保存为“有序点序列/折线”，只是暂时不绘制成有宽度的像素线。

## 4. 模块输入输出和作用

| 模块 | 输入 | 核心处理 | 输出 |
|---|---|---|---|
| Windows 环境 | Git、Conda、RTX 3090、网络 | 创建独立 Python 3.10 环境，安装 PyTorch/依赖、权重和兼容层 | 可运行的推理环境与 smoke JSON |
| CLRNet | `H×W×3 uint8` BGR 图像、配置、checkpoint | 车道候选预测、置信筛选、NMS、坐标解码 | 每帧若干条 `N×2 float64` 图像点序列 |
| 两车道选择 | 每帧多条候选折线 | 当前规则按折线底部横坐标排序，保留最左和最右 | 左右两条有序图像点序列 |
| 点式 BEV/IPM | `(u,v)`、内参 `K`、相机高度、俯仰角、路面假设 | 图像射线与路面相交，或直接应用 `H` | 米制 `(X,Z)` 点或 BEV 坐标点 |
| 位姿对齐 | `(X,Z)` 点、源帧 pose、参考帧 pose | `inv(T_w_ref) @ T_w_src` | 参考帧坐标系中的车道点 |
| 去噪 | 带帧/车道/距离信息的对齐点 | 检查跨帧一致性并降低异常点影响 | 保留点、软权重或稳健车道曲线 |
| 点级融合（建议） | 多帧对齐后的左右车道点 | 按位置、时间、置信度和不确定性汇总 | 融合后的左右车道几何表示 |
| 当前像素加权融合 | 每帧对齐或去噪后的折线、时间权重 | 每帧先栅格化为 mask，再进行 `float32` 像素累加 | 热力图、二值图和像素重叠指标 |
| 栅格化 | 融合点/曲线、BEV范围、分辨率、线宽 | 米制坐标转换为画布坐标并绘制 | `uint8/float32` BEV mask、热力图和成品图 |
| 评估 | 未去噪、去噪、人工参考、逐帧数据 | 完整性、距离残差、跨帧一致性和人工参考对比 | 表格、审计 JSON、对比图 |

## 5. 当前 RANSAC 去噪的问题

### 5.1 当前实现实际上是“X 聚类 + 直线 RANSAC”

现有模块先对所有对齐点按横向 `X` 重新聚类：

- 左侧中心：负 `X` 的 75 分位数；
- 右侧中心：正 `X` 的 25 分位数；
- 默认只保留中心 `±0.8 m`，点太少才放宽至 `±1.2 m`。

代码见 [ransac_denoise.py](results/kitti00_first5_lane_bev_fusion/07_manual_comparison/code/ransac_denoise.py#L14-L33)。之后才使用两点直线模型、`0.3 m` 阈值和 100 次迭代：[ransac_denoise.py](results/kitti00_first5_lane_bev_fusion/07_manual_comparison/code/ransac_denoise.py#L36-L68)。

### 5.2 已确认：本次远处点主要不是被直线 RANSAC 删除，而是被前置 X 聚类删除

2026-07-21 使用保存的五帧 CLRNet 点、现有 IPM、位姿和去噪函数进行只读复算，结果为：

- 去噪输入：585 点；
- X 聚类后：左 212 点、右 276 点，共 488 点；
- 直线 RANSAC 后：仍为左 212 点、右 276 点；
- 因此减少的 97 点全部来自前置 X 聚类，本数据上直线内点判定没有继续删除点。

按参考帧纵向距离 `Z` 分箱后的保留情况：

| 距离 | 保留点 | 保留率 |
|---|---:|---:|
| 3–10 m | 309 / 309 | 100.0% |
| 10–20 m | 139 / 164 | 84.8% |
| 20–30 m | 31 / 62 | 50.0% |
| 30–40 m | 9 / 30 | 30.0% |
| 40–50 m | 0 / 20 | 0.0% |

已有审计数据也与这一现象一致：[two_lane_bev_audit.json](results/kitti00_first5_lane_bev_fusion/audit/two_lane_bev_audit.json#L201-L298)。点数保留率是 `83.4%`，但阈值成图覆盖仅保留原结果的 `34.8%`，说明少量被删除的远端点对应了很长的道路覆盖范围。

严谨结论是：

> 当前去噪模块没有显式“为了提高重叠率而删除远处点”。它先使用固定横向窗口保留靠近全局中心的点；由于近处点密集、远处点稀疏且横向偏差更大，远处点被系统性排除。覆盖区域缩小以后，剩余区域的重叠率被动升高，因此高重叠率不能证明去噪更正确。

### 5.3 造成问题的具体机制

1. **重复聚类破坏已有车道身份**：CLRNet选择阶段已经知道左/右车道，去噪却把全部帧点拼接后重新按全局 `X` 聚类。
2. **固定横向窗口不适合弯道和远处误差**：道路曲率、位姿误差和 IPM 误差都会使远处 `X` 偏离近处中心。
3. **采样密度偏置**：CLRNet沿图像纵向采样；IPM后近处点很密、远处点很稀。按“内点数量”评分会偏向近处。
4. **模型方向不合适**：现有模型写成 `Z=aX+b`。BEV车道通常更接近沿 `Z` 延伸，应优先建模 `X=f(Z)`。
5. **单一直线表达能力不足**：真实道路可能弯曲，位姿/IPM残差也可能随距离变化。
6. **固定 `0.3 m` 阈值没有距离不确定性**：远处图像测量误差经 IPM 放大，却与近处使用相同阈值。
7. **高度和俯仰角是实验假设**：`1.65 m`、`0°` 和平坦路面不是 Sequence 00 的逐帧真实路面外参，远距误差需要单独评估。

## 6. 新去噪方法的候选路线

本阶段不直接选定最终算法。建议按下面顺序做可解释的消融实验。

### 6.1 最小修正基线

1. 取消 `simple_cluster_by_x`。
2. 直接沿用前一阶段已确定的左右车道身份。
3. 将模型方向改为 `X=f(Z)`。
4. 分别处理左、右车道，不把两侧重新混合。

这一步可以验证远处消失是否主要由固定 `X` 窗口造成。

### 6.2 距离均衡的点级方法

- 沿 `Z` 等距重采样，或按 `Z` 分箱后让每个距离区间等权；
- 不能只按总点数最大化得分；
- 增加纵向覆盖约束，例如要求模型在近、中、远距离都有支持；
- 分别报告每个距离区间的保留率。

### 6.3 稳健曲线而不是全局直线

候选模型包括：

- `X=f(Z)` 二次多项式；
- 分段多项式；
- 三次样条或 B 样条；
- 局部 RANSAC / LO-RANSAC；
- 使用 Huber、`soft_l1` 等稳健损失的曲线拟合。

车道研究中已有使用三次样条、Bezier 曲线和 RANSAC曲线拟合的先例，但这些论文只能证明方法可行，不能替代本数据上的消融实验。

### 6.4 跨帧同距离融合

更符合当前任务的方向是：

1. 将每帧左右车道分别重采样到相同的 `Z` 网格；
2. 在每个 `Z` 上比较多帧横向位置 `X`；
3. 使用中位数、Huber估计或带不确定性的加权估计；
4. 输出融合中心、帧支持数和横向离散程度；
5. 低支持区域保留为低置信度，不直接硬删除。

这种方法可以自然保留道路纵向结构，也便于解释每一个被降低权重的点。

## 7. 后续评估不能只看重叠率

建议至少记录：

- 每帧、每条车道、每个距离区间的输入和保留点数；
- 3–10、10–20、20–30、30–40、40–50 m 的覆盖率；
- 融合曲线的纵向最大可见距离；
- 同一 `Z` 下的跨帧横向中位绝对偏差；
- 每个融合位置得到多少帧支持；
- 去噪前后相对于人工参考的横向距离；
- 原始结果、去噪结果和融合结果的完整可视化；
- 所有参数、输入文件 SHA-256 和运行环境。

重叠率必须与覆盖率一起报告。例如“重叠率提高但 40–50 m 覆盖变为 0”应判定为完整性明显下降，而不能直接写成效果提升。

## 8. Windows 复现现状与阻塞项

### 已验证

- 独立 `surf2026-win` Python 3.10 环境；
- PyTorch `2.5.1+cu121`；
- RTX 3090 可用；
- CLRNet模型、权重和一次单图推理不再报错；
- 之前的结果文件仍保存在 `results/kitti00_first5_lane_bev_fusion/`。

### 尚未闭环

1. [scripts/check_windows_env.py](scripts/check_windows_env.py#L52-L75) 只验证单图 `detect()`，没有运行 BEV、位姿、去噪和融合。
2. [surf_bev/pipeline.py](surf_bev/pipeline.py#L65-L112) 的通用多帧入口没有接入成品使用的时间权重和 RANSAC。
3. [scripts/package_first5_results.py](scripts/package_first5_results.py#L205-L257) 只复制、排版已有结果，不会重新生成检测、BEV或融合。
4. 完整成品所依赖的检测导出、数据顺序校正、双车道总编排、公用融合和原始RANSAC脚本尚未全部进入 Windows 分支。
5. 打包的 `metadata/calib.txt` 只有 `P0–P3`，而当前 `load_kitti_calib()` 还要求 `R0_rect` 和 `Tr_velo_to_cam`；因此打包人工对比脚本从现有 metadata 直接重跑会触发缺失键错误。这是明确的复现阻塞项。
6. 当前 Windows 兼容层使用 Python NMS 后处理，和官方 CUDA NMS 不是严格数值等价；需要在最终复现报告中说明。

## 9. 建议的解决顺序

- [x] 明确 CLRNet 输出、点、折线和像素 mask 的区别。
- [x] 确认车道点可以直接做 BEV，不必先画线或 warp 整图。
- [x] 定位本次远处点删除主要发生在 RANSAC 前的 X 聚类。
- [ ] 将完整五帧成品生成脚本和所需元数据纳入 Windows 分支。
- [ ] 建立一条命令的 Windows 端到端复现入口。
- [ ] 保存未去噪的点式融合基线和逐距离审计。
- [ ] 完成“取消 X 聚类”的最小消融。
- [ ] 比较 `X=f(Z)` 直线、二次曲线、稳健样条和跨帧中位轨迹。
- [ ] 选择新去噪方法并确定阈值/权重依据。
- [ ] 最后生成融合图、去噪对比图、逐距离数据和完整审计。

## 10. 主要依据

- CLRNet论文：[CLRNet: Cross Layer Refinement Network for Lane Detection](https://openaccess.thecvf.com/content/CVPR2022/papers/Zheng_CLRNet_Cross_Layer_Refinement_Network_for_Lane_Detection_CVPR_2022_paper.pdf)
- CLRNet官方预测头：[clr_head.py](https://github.com/Turoad/CLRNet/blob/main/clrnet/models/heads/clr_head.py)
- CLRNet官方车道点表示：[lane.py](https://github.com/Turoad/CLRNet/blob/main/clrnet/utils/lane.py)
- OpenCV稀疏点透视变换：[perspectiveTransform](https://docs.opencv.org/4.x/d2/de8/group__core__array.html#gad327659ac03e5fd6894b90025e6900a7)
- OpenCV图像透视变换：[warpPerspective / getPerspectiveTransform](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html)
- OpenCV单应性基础：[Basic concepts of the homography](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html)
- 三次样条车道建模与RANSAC：[Robust Lane Detection and Tracking in Challenging Scenarios](https://escholarship.org/content/qt50n0c8cg/qt50n0c8cg_noSplash_e3c90177116dc6884b3a6ed315e4f4a4.pdf)
- Bezier样条车道拟合：[Real time Detection of Lane Markers in Urban Streets](https://arxiv.org/abs/1411.7113)
- LO-RANSAC：[Fixing the Locally Optimized RANSAC](https://www.bmva-archive.org.uk/bmvc/2012/BMVC/paper095/index.html)
- SciPy稳健最小二乘及Huber/soft-L1损失：[scipy.optimize.least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
