"""Summarize the registered curved-road identity and curve-model comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-identity-result", type=Path, required=True)
    parser.add_argument("--joint-identity-result", type=Path, required=True)
    parser.add_argument("--polynomial-result", type=Path, required=True)
    parser.add_argument("--bspline-result", type=Path, required=True)
    parser.add_argument("--coupled-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def selected_row(rows: list[dict[str, object]]) -> dict[str, object]:
    selected = [
        row for row in rows if bool(row.get("selected_by_one_standard_error_rule"))
    ]
    if len(selected) != 1:
        raise ValueError("Expected exactly one selected cross-validation row.")
    return selected[0]


def best_polynomial(result: dict[str, object]) -> dict[str, object]:
    degree = int(result["selected_degree"])
    matches = [
        row
        for row in result["models"]
        if str(row["model"]).endswith(f" {degree}")
    ]
    if len(matches) != 1:
        raise ValueError("Cannot locate the selected polynomial model row.")
    return matches[0]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    old_identity = load(args.old_identity_result)
    joint_identity = load(args.joint_identity_result)
    polynomial = load(args.polynomial_result)
    bspline = load(args.bspline_result)
    coupled = load(args.coupled_result)

    old_modes = {row["method"]: row for row in old_identity["mode_summaries"]}
    joint_modes = {row["method"]: row for row in joint_identity["mode_summaries"]}
    identities = [
        old_modes["ego_adjacent"],
        old_modes["temporal_ego"],
        joint_modes["temporal_joint"],
    ]
    identity_rows = [
        {
            "method": row["method"],
            "valid_two_lane_frames": row["valid_two_lane_frames"],
            "valid_two_lane_rate": row["valid_two_lane_rate"],
            "longest_consecutive_valid_run": row["longest_consecutive_valid_run"],
            "continuity_mean_m": row["continuity_mean_m"],
            "continuity_p90_m": row["continuity_p90_m"],
            "continuity_max_m": row["continuity_max_m"],
            "continuity_gate_failure_count": row["continuity_gate_failure_count"],
        }
        for row in identities
    ]

    polynomial_row = best_polynomial(polynomial)
    bspline_row = selected_row(bspline["cross_validation"])
    coupled_row = selected_row(coupled["cross_validation"])
    model_rows = [
        {
            "method": f"polynomial_degree_{polynomial['selected_degree']}",
            "mean_fold_rmse_m": polynomial_row["mean_fold_rmse_m"],
            "fold_rmse_standard_error_m": polynomial_row[
                "fold_rmse_standard_error_m"
            ],
            "curve_coupling": "independent left/right",
            "positive_width_by_construction": False,
        },
        {
            "method": "independent_bspline",
            "mean_fold_rmse_m": bspline_row["mean_fold_rmse_m"],
            "fold_rmse_standard_error_m": bspline_row[
                "fold_rmse_standard_error_m"
            ],
            "curve_coupling": "independent left/right",
            "positive_width_by_construction": False,
        },
        {
            "method": "coupled_positive_width_bspline",
            "mean_fold_rmse_m": coupled_row["mean_fold_rmse_m"],
            "fold_rmse_standard_error_m": coupled_row[
                "fold_rmse_standard_error_m"
            ],
            "curve_coupling": "centre plus smoothed log width",
            "positive_width_by_construction": True,
        },
    ]
    write_csv(args.output_dir / "identity_comparison.csv", identity_rows)
    write_csv(args.output_dir / "curve_model_comparison.csv", model_rows)

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    names = [row["method"] for row in identity_rows]
    rates = [100.0 * float(row["valid_two_lane_rate"]) for row in identity_rows]
    axes[0].bar(names, rates, color=("0.45", "#1f77b4", "#2ca02c"))
    axes[0].set_ylabel("valid two-lane frames [%]")
    axes[0].set_title("Identity-selection coverage")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(True, axis="y", linewidth=0.4, alpha=0.35)

    model_names = [row["method"] for row in model_rows]
    rmses = [float(row["mean_fold_rmse_m"]) for row in model_rows]
    errors = [float(row["fold_rmse_standard_error_m"]) for row in model_rows]
    axes[1].bar(
        model_names,
        rmses,
        yerr=errors,
        capsize=3,
        color=("0.45", "#1f77b4", "#2ca02c"),
    )
    axes[1].set_ylabel("held-out-frame mean RMSE [m]")
    axes[1].set_title("Curve consistency (not absolute accuracy)")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(True, axis="y", linewidth=0.4, alpha=0.35)
    figure.tight_layout()
    figure.savefig(args.output_dir / "improvement_summary.png", dpi=220)
    plt.close(figure)

    old_temporal = old_modes["temporal_ego"]
    new_joint = joint_modes["temporal_joint"]
    old_rmse = float(bspline_row["mean_fold_rmse_m"])
    new_rmse = float(coupled_row["mean_fold_rmse_m"])
    result = {
        "status": "complete",
        "identity_rows": identity_rows,
        "model_rows": model_rows,
        "registered_decisions": {
            "joint_valid_frame_change": int(new_joint["valid_two_lane_frames"])
            - int(old_temporal["valid_two_lane_frames"]),
            "joint_continuity_max_change_m": (
                None
                if new_joint["continuity_max_m"] is None
                or old_temporal["continuity_max_m"] is None
                else float(new_joint["continuity_max_m"])
                - float(old_temporal["continuity_max_m"])
            ),
            "coupled_minus_independent_mean_fold_rmse_m": new_rmse - old_rmse,
            "coupled_relative_rmse_change_percent": (
                100.0 * (new_rmse - old_rmse) / old_rmse if old_rmse > 0 else None
            ),
            "coupled_width_positive_by_construction": True,
        },
        "interpretation": [
            "Adopt temporal_joint only if coverage improves without a damaging continuity increase.",
            "Adopt the coupled spline only if held-out consistency is competitive and the plotted curve removes the independent-side bulge.",
            "All error values use CLRNet/IPM-derived held-out points and are not absolute lane accuracy.",
        ],
        "previous_outputs_modified": False,
    }
    (args.output_dir / "RESULT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result["registered_decisions"], ensure_ascii=False))


if __name__ == "__main__":
    main()
