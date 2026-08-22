"""Fit only robust cubic Frenet B-splines on one ranked range option."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import extended_curve_model_common as common  # noqa: E402
from scripts import run_weekly_lane_hierarchy as weekly  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_arguments(parser)
    parser.add_argument(
        "--smoothing-grid",
        default="0.0025,0.01,0.04,0.16",
        help="Positive B-spline smoothing candidates in m^2 per aggregate point.",
    )
    return parser.parse_args()


def parse_smoothing_grid(value: str) -> list[float]:
    grid = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not grid or len(set(grid)) != len(grid) or any(item <= 0 for item in grid):
        raise ValueError("--smoothing-grid must contain unique positive values.")
    return grid


def plot_curves(
    path: Path,
    data: common.ExtendedCurveInput,
    fits: dict[str, weekly.ScalarSpline],
    samples: int,
) -> None:
    colors = {"left": "#1f77b4", "right": "#d62728"}
    figure, axis = plt.subplots(figsize=(7.2, 9.0))
    for side_index, side in enumerate(weekly.SIDES):
        raw = np.vstack(
            [np.asarray(frame["lanes_xz"][side_index]) for frame in data.frames]
        )
        curve_sd = weekly.sample_fit(fits[side], samples)
        curve_xz = weekly.frenet_to_xz(curve_sd, data.reference_path)
        axis.scatter(
            raw[:, 0], raw[:, 1], s=3, alpha=0.12, color=colors[side]
        )
        axis.plot(
            curve_xz[:, 0],
            curve_xz[:, 1],
            linewidth=2.2,
            color=colors[side],
            label=f"{side} robust cubic B-spline",
        )
    axis.set_xlabel("X right in selected reference frame [m]")
    axis.set_ylabel("Z forward in selected reference frame [m]")
    axis.set_title("Separate robust cubic B-spline left/right curves")
    axis.grid(True, linewidth=0.5, alpha=0.35)
    axis.legend()
    axis.set_aspect("equal", adjustable="datalim")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    grid = parse_smoothing_grid(args.smoothing_grid)
    data = common.load_input(args)

    selected_smoothing, cv_rows = weekly.spline_cross_validation(
        data.groups, grid, args
    )
    fits = weekly.fit_both_sides(data.groups, selected_smoothing, args)
    width_check = weekly.curve_width_check(fits)

    weekly.write_csv(
        args.output_dir / "00_audit" / "frame_validity.csv",
        data.frame_audit.values(),
    )
    weekly.write_csv(
        args.output_dir / "01_evaluation" / "bspline_smoothing_cv.csv",
        cv_rows,
    )
    weekly.export_curves(
        args.output_dir / "02_curves",
        fits,
        data.reference_path,
        args.curve_samples,
    )
    plot_curves(
        args.output_dir / "03_figures" / "bspline_left_right.png",
        data,
        fits,
        args.curve_samples,
    )

    fit_details = {
        side: {
            "smoothing_per_point_m2": fits[side].smoothing_per_point_m2,
            "s_min_m": fits[side].s_min,
            "s_max_m": fits[side].s_max,
            "aggregate_bin_count": len(fits[side].aggregate_sd),
            "median_aggregate_residual_m": float(
                np.median(fits[side].residuals_m)
            ),
            "p95_aggregate_residual_m": float(
                np.percentile(fits[side].residuals_m, 95)
            ),
        }
        for side in weekly.SIDES
    }
    result = {
        "status": "complete",
        "method_scope": "robust cubic B-spline only; no polynomial",
        **common.audit_document(data, args),
        "smoothing_grid_m2_per_point": grid,
        "selected_smoothing_per_point_m2": selected_smoothing,
        "selection_rule": "one-standard-error rule on held-out-frame RMSE",
        "cross_validation": cv_rows,
        "fits": fit_details,
        "left_right_width_check": width_check,
        "result_image": str(
            (args.output_dir / "03_figures" / "bspline_left_right.png").resolve()
        ),
    }
    common.write_json(args.output_dir / "RESULT.json", result)
    common.write_json(
        args.output_dir / "STATUS.json",
        {
            "status": "complete",
            "method": "bspline",
            "frames": [data.start_frame, data.end_frame],
            "valid_frames": len(data.frames),
            "selected_smoothing_per_point_m2": selected_smoothing,
            "previous_outputs_modified": False,
        },
    )
    print(
        f"B-SPLINE FINISHED: frames {data.start_frame}-{data.end_frame}, "
        f"smoothing {selected_smoothing}, output {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()

