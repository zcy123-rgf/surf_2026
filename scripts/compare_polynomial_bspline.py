"""Compare robust degree-1/2/3 polynomials with the existing cubic B-spline.

This focused entry point intentionally stops after model comparison and direct
left/right curve export.  It does not run short-segment anchor extraction or
hierarchical sparse refusion.  All metrics are held-out-frame consistency with
the saved CLRNet/IPM points, not accuracy against official lane-line truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import analyze_lane_curve_hierarchy as analysis  # noqa: E402
from scripts import fit_first5_two_curves as curves  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bin-size-m", type=float, default=0.5)
    parser.add_argument("--smoothing-grid", default="0.0025,0.01,0.04,0.16")
    parser.add_argument("--huber-delta-m", type=float, default=0.2)
    parser.add_argument("--irls-iterations", type=int, default=4)
    parser.add_argument("--curve-samples", type=int, default=500)
    return parser.parse_args()


def plot_models(path: Path, rows: list[dict[str, object]]) -> None:
    labels = [
        str(row["model"])
        .replace("robust ", "")
        .replace(" after z-bin ordering", "")
        for row in rows
    ]
    values = [float(row["mean_fold_rmse_m"]) for row in rows]
    colors = ["0.35"] * (len(rows) - 1) + ["#c62828"]
    figure, axis = plt.subplots(figsize=(9.0, 5.2))
    axis.bar(np.arange(len(rows)), values, color=colors)
    axis.set_xticks(np.arange(len(rows)), labels, rotation=16, ha="right")
    axis.set_ylabel("Held-out-frame RMSE [m]")
    axis.set_title("Polynomial and cubic B-spline comparison (lower is better)")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, object]:
    curves.require_empty_output(args.output_dir)
    if not args.aligned_json.is_file():
        raise FileNotFoundError(args.aligned_json)
    frames = curves.load_aligned_json(args.aligned_json)
    frame_ids = [int(frame["frame_id"]) for frame in frames]
    curves.validate_frames(frames, frame_ids)

    grid = [float(value) for value in args.smoothing_grid.split(",")]
    selected_smoothing, spline_trials, selection = analysis.select_smoothing(
        frames, grid, args
    )
    polynomial_results = [
        analysis.polynomial_lofo(frames, degree, args) for degree in (1, 2, 3)
    ]
    selected_spline = selection["selected"]
    model_rows = [
        {key: value for key, value in result.items() if key != "folds"}
        for result in polynomial_results
    ]
    model_rows.append(
        {
            "model": "robust cubic parametric B-spline after z-bin ordering",
            "count": int(selected_spline["lofo_count"]),
            "mean_m": float(selected_spline["lofo_mean_m"]),
            "rmse_m": float(selected_spline["lofo_rmse_m"]),
            "p95_m": float(selected_spline["lofo_p95_m"]),
            "mean_fold_rmse_m": float(selected_spline["lofo_fold_rmse_mean_m"]),
            "fold_rmse_standard_error_m": float(
                selected_spline["lofo_fold_rmse_standard_error_m"]
            ),
            "requires_single_valued_x_of_z": (
                "not mathematically; current z-bin ordering still assumes "
                "monotonic road progress"
            ),
        }
    )
    fitted_sides = analysis.fit_sides(frames, selected_smoothing, args)

    analysis.write_csv(args.output_dir / "model_lofo.csv", model_rows)
    analysis.write_csv(
        args.output_dir / "polynomial_folds.csv",
        [row for result in polynomial_results for row in result["folds"]],
    )
    analysis.write_csv(args.output_dir / "spline_smoothing_trials.csv", spline_trials)
    curves.export_fit(args.output_dir / "bspline_curve_data", fitted_sides, frames)
    curves.plot_fit(
        args.output_dir / "bspline_left_right_curves.png",
        frames,
        fitted_sides,
        "Direct robust cubic B-spline left/right curves",
        frame_ids[-1],
    )
    plot_models(args.output_dir / "model_comparison.png", model_rows)

    result = {
        "status": "complete",
        "scope": "polynomial/B-spline comparison only; no sparse refusion",
        "input": str(args.aligned_json.resolve()),
        "frames": frame_ids,
        "selected_smoothing_per_point_m2": selected_smoothing,
        "models": model_rows,
        "interpretation": (
            "Held-out-frame consistency against CLRNet/IPM-derived points; "
            "not official KITTI lane-line accuracy."
        ),
    }
    (args.output_dir / "RESULT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
