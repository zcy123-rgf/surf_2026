"""Tune a leave-one-frame-out temporal lane-point denoiser.

The real five-frame sample has no lane ground truth.  To avoid selecting a
method merely because it deletes many points, this runner injects repeatable,
known outliers into the real aligned point geometry.  Hyperparameters are
selected using outlier F1 under strict clean-data and far-range retention
constraints.  The untouched real sample is then evaluated separately.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.evaluate_denoise_methods as base  # noqa: E402
from surf_bev.temporal_denoise import (  # noqa: E402
    consensus_mask_from_statistics,
    leave_one_out_consensus_mask,
    leave_one_out_statistics,
)


DEFAULT_ANNOTATION = (
    ROOT / "annotations" / "kitti00_first5_manual_annotations.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize leave-one-frame-out temporal point denoising."
    )
    parser.add_argument(
        "--clrnet-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--manual-json",
        type=Path,
        default=DEFAULT_ANNOTATION,
    )
    parser.add_argument(
        "--calib", type=Path, required=True
    )
    parser.add_argument(
        "--poses",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "workstation_outputs"
        / "kitti00_first5_temporal_optimized_20260725",
    )
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def load_points(args: argparse.Namespace) -> tuple[
    np.ndarray,
    list[dict[str, object]],
    np.ndarray,
    list[dict[str, object]],
]:
    calibration = base.parse_projection(args.calib)
    poses_cam0 = base.load_kitti_poses(args.poses)
    poses = [pose @ calibration["cam0_from_image"] for pose in poses_cam0]
    image_lanes = base.load_image_lanes(args.clrnet_json)
    manual_lanes = base.load_manual_lanes(args.manual_json)
    _, points, ranges = base.align_image_lanes(image_lanes, calibration["K"], poses)
    _, manual_points, manual_ranges = base.align_image_lanes(
        manual_lanes, calibration["K"], poses
    )
    return points, ranges, manual_points, manual_ranges


def inject_isolated(
    clean: np.ndarray,
    ranges: list[dict[str, object]],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    del ranges
    contaminated = clean.copy()
    outlier = np.zeros(len(clean), dtype=bool)
    indexes = rng.choice(len(clean), max(1, round(0.06 * len(clean))), replace=False)
    magnitude = rng.uniform(0.8, 1.5, len(indexes))
    direction = rng.choice([-1.0, 1.0], len(indexes))
    contaminated[indexes, 0] += direction * magnitude
    outlier[indexes] = True
    return contaminated, outlier


def inject_segments(
    clean: np.ndarray,
    ranges: list[dict[str, object]],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    contaminated = clean.copy()
    outlier = np.zeros(len(clean), dtype=bool)
    for item in ranges:
        start, end = int(item["start"]), int(item["end"])
        length = min(5, max(2, (end - start) // 8))
        low = start + max(2, (end - start) // 5)
        high = end - length - max(2, (end - start) // 5)
        segment_start = int(rng.integers(low, max(low + 1, high + 1)))
        indexes = np.arange(segment_start, segment_start + length)
        shift = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.6, 1.0))
        contaminated[indexes, 0] += shift
        outlier[indexes] = True
    return contaminated, outlier


def inject_far_drift(
    clean: np.ndarray,
    ranges: list[dict[str, object]],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    contaminated = clean.copy()
    outlier = np.zeros(len(clean), dtype=bool)
    selected_records = rng.choice(len(ranges), 3, replace=False)
    for record_index in selected_records:
        item = ranges[int(record_index)]
        start, end = int(item["start"]), int(item["end"])
        lane = clean[start:end]
        local = np.flatnonzero(lane[:, 1] >= 25.0)
        if not len(local):
            continue
        indexes = start + local
        z = lane[local, 1]
        scale = (z - np.min(z)) / max(float(np.ptp(z)), 1e-6)
        direction = float(rng.choice([-1.0, 1.0]))
        contaminated[indexes, 0] += direction * (0.45 + 0.75 * scale)
        outlier[indexes] = True
    return contaminated, outlier


def inject_mixed(
    clean: np.ndarray,
    ranges: list[dict[str, object]],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    first, first_mask = inject_isolated(clean, ranges, rng)
    second, second_mask = inject_segments(clean, ranges, rng)
    contaminated = clean.copy()
    contaminated[first_mask] = first[first_mask]
    remaining = second_mask & ~first_mask
    contaminated[remaining] = second[remaining]
    return contaminated, first_mask | second_mask


INJECTORS = {
    "isolated_lateral": inject_isolated,
    "short_shifted_segments": inject_segments,
    "far_range_drift": inject_far_drift,
    "mixed_isolated_and_segments": inject_mixed,
}


def classifier_metrics(
    keep: np.ndarray, known_outlier: np.ndarray, clean_points: np.ndarray
) -> dict[str, float | int]:
    rejected = ~keep
    true_positive = int(np.count_nonzero(rejected & known_outlier))
    false_positive = int(np.count_nonzero(rejected & ~known_outlier))
    false_negative = int(np.count_nonzero(keep & known_outlier))
    true_negative = int(np.count_nonzero(keep & ~known_outlier))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    clean_far = (~known_outlier) & (clean_points[:, 1] >= 30.0)
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "tn": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "clean_retention": true_negative / max(np.count_nonzero(~known_outlier), 1),
        "clean_far_retention": (
            np.count_nonzero(keep & clean_far) / max(np.count_nonzero(clean_far), 1)
        ),
    }


def parameter_grid() -> list[dict[str, float | int]]:
    return [
        {
            "base_threshold": base_threshold,
            "mad_multiplier": mad_multiplier,
            "threshold_cap": threshold_cap,
            "minimum_other_frames": minimum_other_frames,
        }
        for base_threshold, mad_multiplier, threshold_cap, minimum_other_frames in itertools.product(
            (0.10, 0.15, 0.20, 0.25, 0.30),
            (1.5, 2.0, 2.5, 3.0),
            (0.40, 0.60, 0.80, 1.00, 1.20),
            (2, 3),
        )
    ]


def config_id(config: dict[str, float | int]) -> str:
    return (
        f"b{config['base_threshold']:.2f}_"
        f"m{config['mad_multiplier']:.1f}_"
        f"c{config['threshold_cap']:.2f}_"
        f"s{config['minimum_other_frames']}"
    )


def aggregate_grid(
    rows: list[dict[str, object]],
    clean_rows: dict[str, dict[str, object]],
    manual_safety_rows: dict[str, dict[str, float | int]],
) -> list[dict[str, object]]:
    output = []
    ids = sorted({str(row["config_id"]) for row in rows})
    for identifier in ids:
        selected = [row for row in rows if row["config_id"] == identifier]
        scenario_f1 = {}
        for scenario in INJECTORS:
            values = [
                float(row["f1"]) for row in selected if row["scenario"] == scenario
            ]
            scenario_f1[scenario] = float(np.mean(values))
        clean = clean_rows[identifier]
        manual_safety = manual_safety_rows[identifier]
        output.append(
            {
                "config_id": identifier,
                "base_threshold": selected[0]["base_threshold"],
                "mad_multiplier": selected[0]["mad_multiplier"],
                "threshold_cap": selected[0]["threshold_cap"],
                "minimum_other_frames": selected[0]["minimum_other_frames"],
                "mean_f1": float(np.mean([row["f1"] for row in selected])),
                "worst_scenario_f1": min(scenario_f1.values()),
                "mean_precision": float(
                    np.mean([row["precision"] for row in selected])
                ),
                "mean_recall": float(np.mean([row["recall"] for row in selected])),
                "mean_clean_retention_in_injection": float(
                    np.mean([row["clean_retention"] for row in selected])
                ),
                "mean_clean_far_retention_in_injection": float(
                    np.mean([row["clean_far_retention"] for row in selected])
                ),
                "real_point_retention": clean["point_retention"],
                "real_far_retention": clean["far_point_retention_z_ge_30m"],
                "real_cross_frame_mean_deviation_m": clean[
                    "mean_cross_frame_abs_deviation_m"
                ],
                "manual_safety_point_retention": manual_safety[
                    "point_retention"
                ],
                "manual_safety_far_retention": manual_safety[
                    "far_retention"
                ],
                **{
                    f"f1_{scenario}": value
                    for scenario, value in scenario_f1.items()
                },
            }
        )
    return output


def choose_config(rows: list[dict[str, object]]) -> dict[str, object]:
    eligible = [
        row
        for row in rows
        if float(row["real_point_retention"]) >= 0.95
        and float(row["real_far_retention"]) >= 0.90
        and float(row["mean_clean_retention_in_injection"]) >= 0.95
        and float(row["manual_safety_point_retention"]) >= 0.95
        and float(row["manual_safety_far_retention"]) >= 0.90
    ]
    if not eligible:
        raise RuntimeError("No parameter set satisfies the retention constraints.")
    return max(
        eligible,
        key=lambda row: (
            float(row["worst_scenario_f1"]),
            float(row["mean_f1"]),
            float(row["mean_clean_retention_in_injection"]),
            float(row["real_point_retention"]),
        ),
    )


def draw_injection_example(
    path: Path,
    contaminated: np.ndarray,
    known_outlier: np.ndarray,
    keep: np.ndarray,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 9))
    for axis, title, show_keep in (
        (axes[0], "Injected known outliers", np.ones(len(contaminated), dtype=bool)),
        (axes[1], "Selected denoiser output", keep),
    ):
        clean_visible = ~known_outlier & show_keep
        injected_visible = known_outlier & show_keep
        axis.scatter(
            contaminated[clean_visible, 0],
            contaminated[clean_visible, 1],
            s=8,
            color="#2775b6",
            label="clean point",
        )
        axis.scatter(
            contaminated[injected_visible, 0],
            contaminated[injected_visible, 1],
            s=18,
            color="#d62728",
            marker="x",
            label="injected outlier",
        )
        if title.endswith("output"):
            rejected_clean = ~known_outlier & ~keep
            axis.scatter(
                contaminated[rejected_clean, 0],
                contaminated[rejected_clean, 1],
                s=22,
                facecolors="none",
                edgecolors="#ff9f1c",
                label="clean point rejected",
            )
        axis.set(
            xlim=base.FUSION_RANGE[0],
            ylim=base.FUSION_RANGE[1],
            xlabel="X right [m]",
            ylabel="Z forward [m]",
            title=title,
        )
        axis.set_aspect("equal")
        axis.grid(alpha=0.3)
        axis.legend(loc="upper right")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def create_real_comparison(
    path: Path,
    points: np.ndarray,
    ranges: list[dict[str, object]],
    selected_mask: np.ndarray,
) -> None:
    raw_mask = np.ones(len(points), dtype=bool)
    legacy_mask, _ = base.legacy_mask(points)
    quadratic_mask, _ = base.polynomial_ransac_mask(
        points, ranges, degree=2, threshold=0.3
    )
    methods = [
        ("No denoising", raw_mask),
        ("Legacy fixed-X + RANSAC", legacy_mask),
        ("Quadratic X=f(Z), 0.3 m", quadratic_mask),
        ("Optimized leave-one-out gate", selected_mask),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(11, 13))
    for axis, (name, mask) in zip(axes.flat, methods):
        for item in ranges:
            start, end = int(item["start"]), int(item["end"])
            lane = points[start:end][mask[start:end]]
            if len(lane):
                frame_id = int(item["frame_id"])
                axis.plot(
                    lane[:, 0],
                    lane[:, 1],
                    "o-",
                    linewidth=1.1,
                    markersize=2,
                    color=base.FRAME_COLORS_RGB[frame_id],
                )
        far = points[:, 1] >= 30.0
        far_retention = np.count_nonzero(mask & far) / max(np.count_nonzero(far), 1)
        axis.set(
            xlim=base.FUSION_RANGE[0],
            ylim=base.FUSION_RANGE[1],
            xlabel="X right [m]",
            ylabel="Z forward [m]",
            title=(
                f"{name}\n{np.count_nonzero(mask)}/{len(mask)} points, "
                f"Z>=30 m retention {far_retention:.1%}"
            ),
        )
        axis.set_aspect("equal")
        axis.grid(alpha=0.3)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive.")
    base.require_empty_output(args.output_dir)
    points, ranges, manual_points, manual_ranges = load_points(args)
    grid = parameter_grid()

    injection_rows: list[dict[str, object]] = []
    clean_rows: dict[str, dict[str, object]] = {}
    manual_safety_rows: dict[str, dict[str, float | int]] = {}
    clean_method_cache: dict[str, tuple[np.ndarray, dict[str, object]]] = {}
    clean_statistics = leave_one_out_statistics(points, ranges)
    manual_statistics = leave_one_out_statistics(manual_points, manual_ranges)
    manual_far = manual_points[:, 1] >= 30.0
    for config in grid:
        identifier = config_id(config)
        clean_mask, details = consensus_mask_from_statistics(
            clean_statistics, **config
        )
        clean_frames = base.frames_from_mask(points, ranges, clean_mask)
        _, clean_fusion = base.lane_mask(clean_frames, *base.FUSION_RANGE)
        clean_metrics = base.method_metrics(
            identifier,
            points,
            ranges,
            clean_mask,
            manual_points,
            clean_fusion,
        )
        clean_rows[identifier] = clean_metrics
        clean_method_cache[identifier] = clean_mask, details
        manual_mask, _ = consensus_mask_from_statistics(
            manual_statistics, **config
        )
        manual_safety_rows[identifier] = {
            "points": int(np.count_nonzero(manual_mask)),
            "point_retention": float(np.mean(manual_mask)),
            "far_points": int(np.count_nonzero(manual_mask & manual_far)),
            "far_retention": float(
                np.count_nonzero(manual_mask & manual_far)
                / max(np.count_nonzero(manual_far), 1)
            ),
        }

    for repeat in range(args.repeats):
        for scenario_index, (scenario, injector) in enumerate(INJECTORS.items()):
            rng = np.random.default_rng(base.SEED + repeat * 100 + scenario_index)
            contaminated, known_outlier = injector(points, ranges, rng)
            contaminated_statistics = leave_one_out_statistics(
                contaminated, ranges
            )
            for config in grid:
                identifier = config_id(config)
                keep, _ = consensus_mask_from_statistics(
                    contaminated_statistics, **config
                )
                metrics = classifier_metrics(keep, known_outlier, points)
                injection_rows.append(
                    {
                        "config_id": identifier,
                        **config,
                        "repeat": repeat,
                        "scenario": scenario,
                        "known_outliers": int(np.count_nonzero(known_outlier)),
                        **metrics,
                    }
                )

    aggregated = aggregate_grid(
        injection_rows, clean_rows, manual_safety_rows
    )
    selected = choose_config(aggregated)
    selected_id = str(selected["config_id"])
    selected_config = {
        "base_threshold": float(selected["base_threshold"]),
        "mad_multiplier": float(selected["mad_multiplier"]),
        "threshold_cap": float(selected["threshold_cap"]),
        "minimum_other_frames": int(selected["minimum_other_frames"]),
    }
    selected_mask, selected_details = clean_method_cache[selected_id]
    selected_frames = base.frames_from_mask(points, ranges, selected_mask)
    _, selected_fusion = base.lane_mask(selected_frames, *base.FUSION_RANGE)
    selected_metrics = clean_rows[selected_id]
    selected_manual_safety = manual_safety_rows[selected_id]

    audit_dir = args.output_dir / "00_audit"
    base.write_csv(audit_dir / "all_injection_trials.csv", injection_rows)
    base.write_csv(audit_dir / "parameter_grid_summary.csv", aggregated)
    base.write_csv(
        audit_dir / "real_data_grid_metrics.csv",
        [
            {
                "config_id": identifier,
                **metrics,
            }
            for identifier, metrics in clean_rows.items()
        ],
    )
    create_real_comparison(
        args.output_dir / "comparison" / "real_data_methods.png",
        points,
        ranges,
        selected_mask,
    )

    example_rng = np.random.default_rng(base.SEED + 999)
    example_points, example_outlier = inject_mixed(points, ranges, example_rng)
    example_keep, _ = leave_one_out_consensus_mask(
        example_points, ranges, **selected_config
    )
    draw_injection_example(
        args.output_dir / "comparison" / "known_outlier_example.png",
        example_points,
        example_outlier,
        example_keep,
    )
    base.imwrite(
        args.output_dir / "selected_method" / "points_by_frame.png",
        base.draw_points(selected_frames),
    )
    base.imwrite(
        args.output_dir / "selected_method" / "weighted_heatmap.png",
        selected_fusion["heatmap"],
    )
    base.imwrite(
        args.output_dir / "selected_method" / "weighted_binary.png",
        selected_fusion["binary"],
    )
    base.plot_points(
        args.output_dir / "selected_method" / "metric_plot.png",
        selected_frames,
        f"optimized_leave_one_out_{selected_id}",
    )

    scenario_selected = [
        row for row in injection_rows if row["config_id"] == selected_id
    ]
    scenario_summary = {}
    for scenario in INJECTORS:
        rows = [row for row in scenario_selected if row["scenario"] == scenario]
        scenario_summary[scenario] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in (
                "precision",
                "recall",
                "f1",
                "clean_retention",
                "clean_far_retention",
            )
        }

    audit = {
        "status": "complete",
        "purpose": (
            "Tune a leave-one-frame-out temporal point gate using known synthetic "
            "outliers injected into the real aligned geometry."
        ),
        "inputs": {
            "clrnet_json": str(args.clrnet_json.resolve()),
            "clrnet_sha256": base.sha256(args.clrnet_json),
            "calib": str(args.calib.resolve()),
            "calib_sha256": base.sha256(args.calib),
            "poses": str(args.poses.resolve()),
            "poses_sha256": base.sha256(args.poses),
            "raw_points": len(points),
        },
        "selection_constraints": {
            "real_point_retention_minimum": 0.95,
            "real_far_retention_minimum": 0.90,
            "injection_clean_retention_minimum": 0.95,
            "manual_pseudo_label_point_retention_minimum": 0.95,
            "manual_pseudo_label_far_retention_minimum": 0.90,
            "ranking": "worst scenario F1, then mean F1, then retention",
        },
        "selected_config_id": selected_id,
        "selected_parameters": selected_config,
        "selected_internal_details": selected_details,
        "selected_real_metrics": selected_metrics,
        "selected_manual_pseudo_label_safety": selected_manual_safety,
        "selected_injection_metrics": scenario_summary,
        "limitations": [
            "Injected outliers are controlled tests, not a substitute for lane ground truth.",
            "Manual annotations remain pseudo-labels; they are used only as an "
            "anti-over-deletion retention constraint, not as accuracy.",
            "Unsupported far points are deliberately kept rather than guessed to be outliers.",
            "The five-frame result must be revalidated on longer sequences.",
        ],
    }
    (audit_dir / "optimization.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = [
        "# 留一帧跨帧去噪优化核验",
        "",
        "## 做了什么",
        "",
        "- 每个点只与其他帧的同侧车道比较，当前帧不参与自己的参考中值。",
        "- 在真实位姿对齐点上注入位置已知的孤立偏移、连续段偏移、远处漂移和混合异常。",
        f"- 共测试 {len(grid)} 组参数 × {len(INJECTORS)} 类异常 × {args.repeats} 次重复。",
        "- 参数选择同时约束真实数据总点保留率、30 m 后保留率和注入实验的干净点保留率。",
        "",
        "## 选择结果",
        "",
        f"- 参数编号：`{selected_id}`",
        f"- 基础阈值：{selected_config['base_threshold']:.2f} m",
        f"- MAD 倍数：{selected_config['mad_multiplier']:.1f}",
        f"- 最大阈值：{selected_config['threshold_cap']:.2f} m",
        f"- 至少需要其他帧数：{selected_config['minimum_other_frames']}",
        f"- 真实数据保留：{selected_metrics['points']}/{len(points)} "
        f"({selected_metrics['point_retention']:.1%})",
        f"- 真实数据 30 m 后保留率：{selected_metrics['far_point_retention_z_ge_30m']:.1%}",
        f"- 人工伪标注安全检查保留：{selected_manual_safety['points']}/{len(manual_points)} "
        f"({selected_manual_safety['point_retention']:.1%})",
        f"- 人工伪标注 30 m 后安全保留率：{selected_manual_safety['far_retention']:.1%}",
        "",
        "## 已知异常注入结果（多次平均）",
        "",
        "| 场景 | Precision | Recall | F1 | 干净点保留 | 远处干净点保留 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scenario, metrics in scenario_summary.items():
        report.append(
            f"| {scenario} | {metrics['precision']:.1%} | "
            f"{metrics['recall']:.1%} | {metrics['f1']:.1%} | "
            f"{metrics['clean_retention']:.1%} | "
            f"{metrics['clean_far_retention']:.1%} |"
        )
    report.extend(
        [
            "",
            "## 使用边界",
            "",
            "- 这些指标证明的是对“注入异常”的识别能力，不是 KITTI 车道线准确率。",
            "- 没有足够其他帧支持的远处点默认保留，避免再次发生后段被系统删除。",
            "- 该参数可作为下一版候选，但仍应在更长序列和人工核验上继续验证。",
        ]
    )
    (args.output_dir / "OPTIMIZATION_REPORT_ZH.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output_dir),
                "grid_configs": len(grid),
                "trials": len(injection_rows),
                "selected": selected_id,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
