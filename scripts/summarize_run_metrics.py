"""Summarize validation metrics without claiming unavailable ground truth.

The report separates observed-fit consistency, held-out CLRNet/IPM consistency,
coverage, and continuity. None of these is an official lane-position accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def observed_fit_metrics(output_dir: Path) -> dict[str, object]:
    aggregate_rows = read_csv(output_dir / "window_aggregate_points.csv")
    curve_rows = read_csv(output_dir / "window_curve_samples.csv")
    groups: dict[tuple[str, str], tuple[list[dict[str, str]], list[dict[str, str]]]] = {}
    for row in aggregate_rows:
        key = (row["window_id"], row["side"])
        groups.setdefault(key, ([], []))[0].append(row)
    for row in curve_rows:
        key = (row["window_id"], row["side"])
        groups.setdefault(key, ([], []))[1].append(row)
    values = []
    for (window_id, side), (observed, fitted) in sorted(groups.items()):
        observed_points = np.asarray(
            [[float(row["x_common_m"]), float(row["z_common_m"])] for row in observed]
        )
        fitted_points = np.asarray(
            [[float(row["x_common_m"]), float(row["z_common_m"])] for row in fitted]
        )
        values.append(
            {
                "window_id": int(window_id),
                "side": side,
                "observed_point_count": len(observed_points),
                "fitted_sample_count": len(fitted_points),
                "observed_x_range_m": [float(observed_points[:, 0].min()), float(observed_points[:, 0].max())],
                "fitted_x_range_m": [float(fitted_points[:, 0].min()), float(fitted_points[:, 0].max())],
            }
        )
    return {
        "basis": "aggregate CLRNet/IPM observations and fitted window samples are exported separately",
        "warning": "Chamfer is not computed here because the two exports do not share a guaranteed one-to-one support domain. Use Chamfer only with an explicitly declared reference lane set.",
        "window_side_rows": values,
    }


def heldout_metrics(output_dir: Path) -> dict[str, object]:
    rows = read_csv(output_dir / "model_comparison.csv")
    for row in rows:
        for key in ("heldout_count", "heldout_mean_m", "heldout_rmse_m", "heldout_median_m", "heldout_p90_m", "heldout_p95_m", "heldout_max_m"):
            row[key] = float(row[key])
    total_count = sum(row["heldout_count"] for row in rows)
    pooled_rmse = (
        float(np.sqrt(sum(row["heldout_count"] * row["heldout_rmse_m"] ** 2 for row in rows) / total_count))
        if total_count
        else None
    )
    return {
        "basis": "held-out CLRNet/IPM observations omitted from each window fit",
        "warning": "This measures consistency with the detector/projection pipeline, not real-world lane accuracy.",
        "window_side_rows": rows,
        "heldout_point_count": int(total_count),
        "pooled_rmse_m": pooled_rmse,
        "mean_row_median_m": float(np.mean([row["heldout_median_m"] for row in rows])) if rows else None,
        "mean_row_p90_m": float(np.mean([row["heldout_p90_m"] for row in rows])) if rows else None,
        "maximum_row_rmse_m": max((row["heldout_rmse_m"] for row in rows), default=None),
    }


def coverage_metrics(output_dir: Path) -> dict[str, object]:
    rows = read_csv(output_dir / "window_plan.csv")
    result: dict[str, object] = {}
    for side in ("left", "right"):
        values = [float(row[f"{side}_coverage_fraction"]) for row in rows]
        result[side] = {
            "window_count": len(values),
            "mean_frame_coverage_fraction": float(np.mean(values)) if values else None,
            "minimum_frame_coverage_fraction": float(np.min(values)) if values else None,
        }
    return {
        "basis": "selected observed lane frames divided by frames in each window",
        "warning": "Coverage describes availability, not semantic lane identity or accuracy.",
        "sides": result,
    }


def continuity_metrics(output_dir: Path) -> dict[str, object]:
    rows = read_csv(output_dir / "window_continuity.csv")
    failures = [row for row in rows if row["passes_continuity_gate"].lower() != "true"]
    return {
        "basis": "overlap between adjacent fitted windows",
        "warning": "A failed gate starts a separate output segment; it is not a detector accuracy score.",
        "adjacent_window_rows": len(rows),
        "failed_gate_rows": len(failures),
        "failed_rows": failures,
    }


def build_report(output_dir: Path) -> dict[str, object]:
    return {
        "status": "complete",
        "ground_truth_available": False,
        "ground_truth_statement": "No official lane-boundary ground truth was supplied for this run.",
        "observed_fit_consistency": observed_fit_metrics(output_dir),
        "heldout_consistency": heldout_metrics(output_dir),
        "coverage": coverage_metrics(output_dir),
        "continuity": continuity_metrics(output_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.output_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    main()
