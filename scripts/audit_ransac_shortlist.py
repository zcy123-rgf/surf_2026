"""Re-audit the RANSAC shortlist with disjoint random seeds.

This is a continuation of ``optimize_ransac_parameters.py``.  It does not
re-use the tuning seeds for final acceptance and refuses to mark a candidate
accepted when any predeclared safety constraint fails.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.evaluate_denoise_methods as base  # noqa: E402
import scripts.optimize_ransac_parameters as opt  # noqa: E402
import scripts.optimize_temporal_denoise as injection  # noqa: E402


SHORTLIST_SEED_OFFSET = 500_000
HELDOUT_SEED_OFFSET = 700_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit top RANSAC configurations.")
    parser.add_argument(
        "--clrnet-json",
        type=Path,
        default=opt.DEFAULT_DATA / "clrnet_lanes.json",
    )
    parser.add_argument(
        "--manual-json",
        type=Path,
        default=opt.DEFAULT_DATA / "manual_annotations.json",
    )
    parser.add_argument(
        "--calib",
        type=Path,
        default=opt.DEFAULT_PACKAGE / "calib.txt",
    )
    parser.add_argument(
        "--poses",
        type=Path,
        default=opt.DEFAULT_PACKAGE / "poses_00_first5.txt",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT
        / "workstation_outputs"
        / "kitti00_first5_ransac_optimized_20260727_full",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "workstation_outputs"
        / "kitti00_first5_ransac_shortlist_audit_20260727",
    )
    parser.add_argument("--shortlist-size", type=int, default=30)
    parser.add_argument("--shortlist-safety-seeds", type=int, default=80)
    parser.add_argument("--shortlist-injection-repeats", type=int, default=40)
    parser.add_argument("--heldout-safety-seeds", type=int, default=200)
    parser.add_argument("--heldout-injection-repeats", type=int, default=100)
    parser.add_argument(
        "--shortlist-manual-far-min",
        type=float,
        default=0.90,
        help="Stronger shortlist-only safety buffer; held-out acceptance remains 0.90.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def row_to_config(row: dict[str, object]) -> dict[str, object]:
    return {
        "degree": int(row["degree"]),
        "base_threshold": float(row["base_threshold"]),
        "distance_slope": float(row["distance_slope"]),
        "threshold_cap": float(row["threshold_cap"]),
        "iterations": int(row["iterations"]),
        "minimum_sample_z_span": float(row["minimum_sample_z_span"]),
        "score_mode": str(row["score_mode"]),
        "local_refinements": int(row["local_refinements"]),
    }


def row_eligible_text(row: dict[str, str]) -> bool:
    converted = {
        key: float(row[key])
        for key in (
            "real_point_retention_min",
            "real_far_retention_min",
            "manual_point_retention_min",
            "manual_far_retention_min",
            "mean_clean_retention_in_injection",
            "mean_clean_far_retention_in_injection",
        )
    }
    return opt.eligible(converted)


def phase2_order(row: dict[str, str]) -> tuple[float, ...]:
    return (
        -float(row["worst_scenario_f1"]),
        -float(row["mean_f1"]),
        float(row["f1_std_all_trials"]),
        -float(row["manual_far_retention_min"]),
        -float(row["real_point_retention_min"]),
    )


def scenario_summary(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    return {
        scenario: {
            "precision": float(
                np.mean(
                    [row["precision"] for row in rows if row["scenario"] == scenario]
                )
            ),
            "recall": float(
                np.mean([row["recall"] for row in rows if row["scenario"] == scenario])
            ),
            "f1": float(
                np.mean([row["f1"] for row in rows if row["scenario"] == scenario])
            ),
            "f1_std": float(
                np.std([row["f1"] for row in rows if row["scenario"] == scenario])
            ),
            "clean_retention": float(
                np.mean(
                    [
                        row["clean_retention"]
                        for row in rows
                        if row["scenario"] == scenario
                    ]
                )
            ),
            "clean_far_retention": float(
                np.mean(
                    [
                        row["clean_far_retention"]
                        for row in rows
                        if row["scenario"] == scenario
                    ]
                )
            ),
        }
        for scenario in injection.INJECTORS
    }


def main() -> None:
    args = parse_args()
    for name in (
        "shortlist_size",
        "shortlist_safety_seeds",
        "shortlist_injection_repeats",
        "heldout_safety_seeds",
        "heldout_injection_repeats",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name} must be positive")
    base.require_empty_output(args.output_dir)

    source_summary = read_csv(args.source_dir / "00_audit" / "phase2_summary.csv")
    source_eligible = sorted(
        [
            row
            for row in source_summary
            if row_eligible_text(row)
            and float(row["manual_far_retention_min"])
            >= args.shortlist_manual_far_min
        ],
        key=phase2_order,
    )
    shortlist_source = source_eligible[: args.shortlist_size]
    configs = [row_to_config(row) for row in shortlist_source]

    points, ranges, manual_points, manual_ranges = opt.load_points(args)
    shortlist_safety, _ = opt.evaluate_safety(
        configs,
        points,
        ranges,
        manual_points,
        manual_ranges,
        seeds=args.shortlist_safety_seeds,
        seed_offset=SHORTLIST_SEED_OFFSET,
    )
    shortlist_trials = opt.evaluate_injections(
        configs,
        points,
        ranges,
        repeats=args.shortlist_injection_repeats,
        seed_offset=SHORTLIST_SEED_OFFSET + 100_000,
    )
    shortlist_summary = opt.aggregate_trials(
        configs, shortlist_trials, shortlist_safety
    )
    buffered_shortlist = [
        row
        for row in shortlist_summary
        if opt.eligible(row)
        and float(row["manual_far_retention_min"])
        >= args.shortlist_manual_far_min
    ]
    if not buffered_shortlist:
        raise RuntimeError(
            "No shortlist configuration satisfied the stronger manual-far buffer."
        )
    selected_row = opt.rank_rows(buffered_shortlist)[0]
    selected_id = str(selected_row["config_id"])
    selected_config = next(
        config for config in configs if opt.config_id(config) == selected_id
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
        f"held-out RANSAC candidate: {selected_id}",
    )

    legacy_mask, _ = base.legacy_mask(points)
    prior_quadratic, _ = base.polynomial_ransac_mask(
        points, ranges, degree=2, threshold=0.3
    )
    opt.draw_point_comparison(
        args.output_dir / "comparison" / "heldout_best_vs_baselines.png",
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
                "Held-out candidate",
                final_mask,
                f"{np.count_nonzero(final_mask)} points",
            ),
        ],
    )
    opt.draw_parameter_summary(
        args.output_dir / "comparison" / "shortlist_summary.png",
        shortlist_summary,
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
        args.output_dir / "comparison" / "heldout_known_outlier_example.png",
        example_points,
        example_outlier,
        example_keep,
    )

    diagnostics = opt.fit_geometry_diagnostics(
        points, ranges, final_mask, int(selected_config["degree"])
    )
    audit_dir = args.output_dir / "00_audit"
    base.write_csv(audit_dir / "shortlist_trials.csv", shortlist_trials)
    base.write_csv(audit_dir / "shortlist_summary.csv", shortlist_summary)
    base.write_csv(audit_dir / "heldout_trials.csv", heldout_trials)
    base.write_csv(audit_dir / "heldout_safety.csv", heldout_safety)

    scenarios = scenario_summary(heldout_trials)
    previous_audit = json.loads(
        (args.source_dir / "00_audit" / "ransac_optimization.json").read_text(
            encoding="utf-8"
        )
    )
    previous_trials = int(previous_audit["search"]["total_controlled_trials"])
    new_trials = len(shortlist_trials) + len(heldout_trials)
    audit = {
        "status": "accepted" if accepted else "rejected_by_predeclared_constraints",
        "source_search": str(args.source_dir.resolve()),
        "source_selected_candidate_rejected_reason": (
            "The first selected candidate reached only 89.47% worst-case manual "
            "pseudo-label far retention in its 50-seed audit, below the 90% rule."
        ),
        "shortlist": {
            "configurations": len(configs),
            "safety_seeds_per_configuration": args.shortlist_safety_seeds,
            "injection_repeats_per_scenario": args.shortlist_injection_repeats,
            "manual_far_retention_buffer": args.shortlist_manual_far_min,
            "controlled_trials": len(shortlist_trials),
        },
        "heldout": {
            "safety_seeds": args.heldout_safety_seeds,
            "injection_repeats_per_scenario": args.heldout_injection_repeats,
            "controlled_trials": len(heldout_trials),
            "seed_offset": HELDOUT_SEED_OFFSET,
        },
        "cumulative_controlled_trials": previous_trials + new_trials,
        "selected_config_id": selected_id,
        "selected_parameters": selected_config,
        "accepted": accepted,
        "heldout_summary": heldout_summary,
        "heldout_scenarios": scenarios,
        "deterministic_real_run": {
            "metrics": final_metrics,
            "details": final_details,
        },
        "geometry_diagnostics": diagnostics,
        "limitations": [
            "No lane-line ground truth is available for these five frames.",
            "Injected anomalies are controlled tests and not KITTI accuracy.",
            "Manual annotations are pseudo-labels used only to audit over-deletion.",
            "The tested two outer lines do not satisfy a one-lane 3.0-3.75 m width assumption.",
        ],
    }
    (audit_dir / "heldout_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = [
        "# RANSAC候选独立复核",
        "",
        f"- 状态：{'通过预设约束' if accepted else '未通过预设约束'}",
        f"- 初轮候选因人工伪标注远处最差保留率89.47%而未直接接受。",
        f"- 重新复核前{len(configs)}组配置：每组{args.shortlist_safety_seeds}个安全种子，"
        f"每类异常{args.shortlist_injection_repeats}次。",
        f"- 最终候选再用{args.heldout_safety_seeds}个全新安全种子和每类异常"
        f"{args.heldout_injection_repeats}次独立验收。",
        f"- 累计受控异常试验：{previous_trials + new_trials}次。",
        "",
        "## 最终候选",
        "",
        f"- `{selected_id}`",
        f"- 参数：{json.dumps(selected_config, ensure_ascii=False)}",
        f"- 真实点确定性运行：{final_metrics['points']}/{len(points)}，"
        f"30m后{final_metrics['far_point_retention_z_ge_30m']:.1%}",
        f"- 真实点200种子最差保留：{heldout_summary['real_point_retention_min']:.1%}，"
        f"30m后最差{heldout_summary['real_far_retention_min']:.1%}",
        f"- 人工伪标注200种子最差保留：{heldout_summary['manual_point_retention_min']:.1%}，"
        f"30m后最差{heldout_summary['manual_far_retention_min']:.1%}",
        "",
        "## 已知异常独立验收（100次平均）",
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
            "## 结论边界",
            "",
            "- 若状态为未通过，则不能把该候选写成最终可用方法。",
            "- 即使状态通过，也只说明通过本次预设安全与注入异常约束，不是KITTI准确率。",
            "- 车道宽度和曲率仅作诊断，不作为硬删除。",
        ]
    )
    (args.output_dir / "HELDOUT_AUDIT_REPORT_ZH.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(json.dumps({"accepted": accepted, "selected": selected_id}, ensure_ascii=False))
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
