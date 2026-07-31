"""Safety-first expansion of the same side-aware polynomial RANSAC method."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.evaluate_denoise_methods as base  # noqa: E402
import scripts.optimize_ransac_parameters as opt  # noqa: E402
import scripts.optimize_temporal_denoise as injection  # noqa: E402


SCREEN_SEED_OFFSET = 900_000
TUNING_SEED_OFFSET = 1_000_000
HELDOUT_SEED_OFFSET = 1_200_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand safe RANSAC parameters.")
    parser.add_argument(
        "--clrnet-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--manual-json",
        type=Path,
        default=opt.DEFAULT_ANNOTATION,
    )
    parser.add_argument(
        "--calib",
        type=Path,
        required=True,
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
        / "kitti00_first5_ransac_safety_expanded_20260727",
    )
    parser.add_argument(
        "--prior-audit",
        type=Path,
        default=None,
        help=(
            "Optional earlier held-out audit used only to report a cumulative "
            "controlled-trial count. Omit for a clean from-scratch run."
        ),
    )
    parser.add_argument("--screen-safety-seeds", type=int, default=100)
    parser.add_argument("--tuning-injection-repeats", type=int, default=60)
    parser.add_argument("--heldout-safety-seeds", type=int, default=500)
    parser.add_argument("--heldout-injection-repeats", type=int, default=200)
    parser.add_argument("--manual-far-buffer", type=float, default=0.947)
    return parser.parse_args()


def prior_trial_count(prior_audit: Path | None) -> int:
    """Read an explicitly supplied prior audit; clean runs have no dependency."""

    if prior_audit is None:
        return 0
    if not prior_audit.is_file():
        raise FileNotFoundError(f"Prior audit does not exist: {prior_audit}")
    record = json.loads(prior_audit.read_text(encoding="utf-8"))
    value = int(record["cumulative_controlled_trials"])
    if value < 0:
        raise ValueError("Prior cumulative controlled-trial count cannot be negative.")
    return value


def expanded_grid() -> list[dict[str, object]]:
    output = {}
    for (
        degree,
        base_threshold,
        slope,
        cap,
        iterations,
        span,
        score,
        refinements,
    ) in itertools.product(
        (3,),
        (0.30, 0.35, 0.40),
        (0.015, 0.020),
        (0.80, 1.00, 1.20),
        (1000, 2000),
        (5.0, 10.0),
        ("balanced",),
        (1, 2),
    ):
        config = {
            "degree": degree,
            "base_threshold": base_threshold,
            "distance_slope": slope,
            "threshold_cap": cap,
            "iterations": iterations,
            "minimum_sample_z_span": span,
            "score_mode": score,
            "local_refinements": refinements,
        }
        output[opt.config_id(config)] = config
    return list(output.values())


def scenario_summary(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    output = {}
    for scenario in injection.INJECTORS:
        selected = [row for row in rows if row["scenario"] == scenario]
        output[scenario] = {
            key: float(np.mean([row[key] for row in selected]))
            for key in (
                "precision",
                "recall",
                "f1",
                "clean_retention",
                "clean_far_retention",
            )
        }
        output[scenario]["f1_std"] = float(
            np.std([row["f1"] for row in selected])
        )
    return output


def main() -> None:
    args = parse_args()
    for name in (
        "screen_safety_seeds",
        "tuning_injection_repeats",
        "heldout_safety_seeds",
        "heldout_injection_repeats",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name} must be positive")
    base.require_empty_output(args.output_dir)
    points, ranges, manual_points, manual_ranges = opt.load_points(args)
    configs = expanded_grid()

    safety, _ = opt.evaluate_safety(
        configs,
        points,
        ranges,
        manual_points,
        manual_ranges,
        seeds=args.screen_safety_seeds,
        seed_offset=SCREEN_SEED_OFFSET,
    )
    safe_ids = {
        str(row["config_id"])
        for row in safety
        if float(row["real_point_retention_min"]) >= 0.95
        and float(row["real_far_retention_min"]) >= 0.90
        and float(row["manual_point_retention_min"]) >= 0.98
        and float(row["manual_far_retention_min"]) >= args.manual_far_buffer
    }
    safe_configs = [
        config for config in configs if opt.config_id(config) in safe_ids
    ]
    if not safe_configs:
        raise RuntimeError("No expanded configuration passed the safety screen.")
    safety_by_id = {str(row["config_id"]): row for row in safety}
    safe_safety = [safety_by_id[opt.config_id(config)] for config in safe_configs]

    tuning_trials = opt.evaluate_injections(
        safe_configs,
        points,
        ranges,
        repeats=args.tuning_injection_repeats,
        seed_offset=TUNING_SEED_OFFSET,
    )
    tuning_summary = opt.aggregate_trials(
        safe_configs, tuning_trials, safe_safety
    )
    buffered = [
        row
        for row in tuning_summary
        if opt.eligible(row)
        and float(row["manual_far_retention_min"])
        >= args.manual_far_buffer
    ]
    if not buffered:
        raise RuntimeError("No safe configuration passed injection constraints.")
    selected_row = opt.rank_rows(buffered)[0]
    selected_id = str(selected_row["config_id"])
    selected_config = next(
        config for config in safe_configs if opt.config_id(config) == selected_id
    )

    heldout_safety, _ = opt.evaluate_safety(
        [selected_config],
        points,
        ranges,
        manual_points,
        manual_ranges,
        seeds=args.heldout_safety_seeds,
        seed_offset=HELDOUT_SEED_OFFSET,
    )
    heldout_trials = opt.evaluate_injections(
        [selected_config],
        points,
        ranges,
        repeats=args.heldout_injection_repeats,
        seed_offset=HELDOUT_SEED_OFFSET + 100_000,
    )
    heldout_summary = opt.aggregate_trials(
        [selected_config], heldout_trials, heldout_safety
    )[0]
    accepted = opt.eligible(heldout_summary)

    final_mask, final_details, final_metrics, final_fusion = opt.selected_real_result(
        selected_config,
        points,
        ranges,
        manual_points,
        base.SEED + HELDOUT_SEED_OFFSET + 999_999,
    )
    final_frames = base.frames_from_mask(points, ranges, final_mask)
    selected_dir = args.output_dir / "selected_method"
    base.imwrite(selected_dir / "points_by_frame.png", base.draw_points(final_frames))
    base.imwrite(selected_dir / "weighted_heatmap.png", final_fusion["heatmap"])
    base.imwrite(selected_dir / "weighted_binary.png", final_fusion["binary"])
    base.plot_points(
        selected_dir / "metric_plot.png",
        final_frames,
        f"safety-expanded RANSAC: {selected_id}",
    )

    legacy_mask, _ = base.legacy_mask(points)
    prior_quadratic, _ = base.polynomial_ransac_mask(
        points, ranges, degree=2, threshold=0.3
    )
    opt.draw_point_comparison(
        args.output_dir / "comparison" / "final_best_vs_baselines.png",
        points,
        ranges,
        [
            ("No denoising", np.ones(len(points), dtype=bool), "590 points"),
            ("Legacy", legacy_mask, f"{np.count_nonzero(legacy_mask)} points"),
            (
                "Previous quadratic",
                prior_quadratic,
                f"{np.count_nonzero(prior_quadratic)} points",
            ),
            (
                "Safety-expanded RANSAC",
                final_mask,
                f"{np.count_nonzero(final_mask)} points",
            ),
        ],
    )
    opt.draw_parameter_summary(
        args.output_dir / "comparison" / "expanded_parameter_summary.png",
        tuning_summary,
        selected_id,
    )
    example_rng = np.random.default_rng(base.SEED + HELDOUT_SEED_OFFSET + 123)
    example_points, example_outlier = injection.inject_mixed(
        points, ranges, example_rng
    )
    example_keep, _ = opt.RansacEvaluator(
        example_points, ranges, base.SEED + HELDOUT_SEED_OFFSET + 456
    ).apply(selected_config)
    injection.draw_injection_example(
        args.output_dir / "comparison" / "known_outlier_example.png",
        example_points,
        example_outlier,
        example_keep,
    )

    diagnostics = opt.fit_geometry_diagnostics(
        points, ranges, final_mask, int(selected_config["degree"])
    )
    audit_dir = args.output_dir / "00_audit"
    base.write_csv(audit_dir / "expanded_safety_screen.csv", safety)
    base.write_csv(audit_dir / "safe_tuning_trials.csv", tuning_trials)
    base.write_csv(audit_dir / "safe_tuning_summary.csv", tuning_summary)
    base.write_csv(audit_dir / "heldout_safety.csv", heldout_safety)
    base.write_csv(audit_dir / "heldout_trials.csv", heldout_trials)

    previous_trials = prior_trial_count(args.prior_audit)
    cumulative_trials = (
        previous_trials + len(tuning_trials) + len(heldout_trials)
    )
    scenarios = scenario_summary(heldout_trials)
    audit = {
        "status": "accepted" if accepted else "rejected_by_predeclared_constraints",
        "accepted": accepted,
        "expanded_configurations": len(configs),
        "safety_screen_seeds_per_configuration": args.screen_safety_seeds,
        "safe_configurations_after_screen": len(safe_configs),
        "manual_far_safety_buffer": args.manual_far_buffer,
        "tuning_injection_repeats_per_scenario": args.tuning_injection_repeats,
        "heldout_safety_seeds": args.heldout_safety_seeds,
        "heldout_injection_repeats_per_scenario": args.heldout_injection_repeats,
        "prior_audit": (
            str(args.prior_audit.resolve()) if args.prior_audit is not None else None
        ),
        "prior_controlled_trials": previous_trials,
        "cumulative_controlled_trials": cumulative_trials,
        "selected_config_id": selected_id,
        "selected_parameters": selected_config,
        "heldout_summary": heldout_summary,
        "heldout_scenarios": scenarios,
        "deterministic_real_run": {
            "metrics": final_metrics,
            "details": final_details,
        },
        "geometry_diagnostics": diagnostics,
        "limitations": [
            "No lane-line ground truth is available for the five real frames.",
            "Injected anomalies are controlled tests, not KITTI lane accuracy.",
            "Manual annotations are pseudo-labels used only for over-deletion safety.",
            "The two outer detected lines are not validated as one traffic lane's two boundaries.",
        ],
    }
    (audit_dir / "final_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = [
        "# RANSAC安全扩展最终核验",
        "",
        f"- 状态：{'通过预设约束' if accepted else '未通过预设约束'}",
        f"- 扩展测试{len(configs)}组同类RANSAC配置，安全筛选后剩余{len(safe_configs)}组。",
        f"- 累计受控异常试验：{cumulative_trials}次。",
        f"- 最终配置：`{selected_id}`。",
        f"- 参数：{json.dumps(selected_config, ensure_ascii=False)}",
        f"- 真实点确定性运行：{final_metrics['points']}/{len(points)}，"
        f"30m后{final_metrics['far_point_retention_z_ge_30m']:.1%}。",
        f"- 500种子真实点最差保留：{heldout_summary['real_point_retention_min']:.1%}，"
        f"30m后{heldout_summary['real_far_retention_min']:.1%}。",
        f"- 500种子人工伪标注最差保留：{heldout_summary['manual_point_retention_min']:.1%}，"
        f"30m后{heldout_summary['manual_far_retention_min']:.1%}。",
        "",
        "## 已知异常独立验收（200次平均）",
        "",
        "| 场景 | Precision | Recall | F1 | 干净点保留 | 远处干净点保留 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scenario, values in scenarios.items():
        report.append(
            f"| {scenario} | {values['precision']:.1%} | {values['recall']:.1%} | "
            f"{values['f1']:.1%} | {values['clean_retention']:.1%} | "
            f"{values['clean_far_retention']:.1%} |"
        )
    report.extend(
        [
            "",
            "## 真实结论",
            "",
            "- 安全阈值扩大后，过删风险降低，但远处渐变漂移的召回会进一步下降。",
            "- 这说明在只有五帧且必须保留远处真实点时，纯曲线RANSAC无法同时高召回识别远处缓慢漂移。",
            "- 车道宽度3.0–3.75m不适用于当前两条外侧线，不启用该硬限制。",
        ]
    )
    (args.output_dir / "FINAL_RANSAC_REPORT_ZH.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "accepted": accepted,
                "selected": selected_id,
                "safe_configurations": len(safe_configs),
                "cumulative_trials": cumulative_trials,
            },
            ensure_ascii=False,
        )
    )
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
