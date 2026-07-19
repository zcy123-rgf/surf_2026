# 人工标注与 CLRNet 的五帧 BEV 融合对比

本目录追加在实验结果分支中，用于修改和补充上一版结果；未合并到 `main`。

## 标注定义

这 5 帧道路没有连续、清晰的喷涂车道线。人工方法逐帧标注的是“视觉上可辨认的左右行驶边界”：

- 红点：人工逐点锚点；
- 青/黄线：锚点之间的分段线性插值；
- 右侧边界受停放车辆遮挡，遮挡段为人工插值；
- 该标注是人工伪标注，不是车道线真值。

全部原始像素坐标保存在 `manual_annotations.json`，没有调用 CLRNet 或其他车道线检测器生成这些坐标。CLRNet 保存结果只在最后比较阶段读取。

## 相同处理条件

人工标注和 CLRNet 都使用：

- 相同 KITTI Sequence 00 帧 `000000`—`000004`；
- 相同固定 IPM 参数：高度 `1.65 m`、俯仰角 `0°`；
- 相同 BEV 范围：`X[-10,10] m`、`Z[3,50] m`；
- 相同位姿，统一对齐到 `000004`；
- 相同时间权重 `[0.2,0.4,0.6,0.8,1.0]`；
- 相同 RANSAC：阈值 `0.3 m`、100 次迭代。

## 本地结果

- `results/01_manual_annotations_five_frames.png`：5 帧人工锚点及插值边界。
- `results/02_manual_fusion_without_denoise.png`：人工标注未去噪融合。
- `results/03_manual_fusion_with_ransac.png`：人工标注 RANSAC 后融合。
- `results/04_manual_vs_clrnet_fusion_comparison.png`：人工与 CLRNet 融合成品对比；沿用此前的白底米制坐标、逐帧着色、参考车原点和网格形式。
- `results/frame_*_manual_bev.png`：逐帧人工 BEV。
- `results/frame_*_manual_aligned_to_000004.png`：逐帧位姿对齐结果。

## 对比结果

| 项目 | 人工标注 | CLRNet |
|---|---:|---:|
| 未去噪点数 | 580 | 585 |
| RANSAC 后点数 | 435 | 488 |
| 未去噪多帧重叠率 | 21.1% | 43.1% |
| RANSAC 后多帧重叠率 | 36.9% | 71.0% |

- 人工与 CLRNet 在原图公共高度范围内的平均横向差为 `36.5 px`。
- 采用 BEV 中 `±12 px` 容差，两种方法的成图一致率为：未去噪 `4.7%`，RANSAC 后 `7.3%`。
- 人工 RANSAC 后成图覆盖保留约 `52.1%`；右侧各帧只保留 `20—39 / 56—59` 个点，主要原因是右侧遮挡段的人工插值跨帧不够一致。

这些数值描述的是两种方法和多帧之间的一致程度，不是检测准确率。固定 IPM 会明显放大图像远处几十像素的标注差异，因此不能根据本次比较直接断言人工或 CLRNet 哪一个更接近真实道路边界。

成品对比图只复用此前融合图的可视化形式，并未修改本次数据、位姿、BEV 范围或统计结果。此前示例使用了另一组帧和更长的纵向坐标范围；本图严格显示当前五帧实际有效范围 `Z[3,50] m`，并额外显示位于 `(0,0)` 的参考车位置。

完整逐帧数据、参数和指标见 `audit/manual_vs_clrnet_audit.json`。

在仓库根目录安装依赖后，可运行 `python results/kitti00_first5_lane_bev_fusion/07_manual_comparison/code/run_manual_comparison.py` 重新生成本目录结果。
