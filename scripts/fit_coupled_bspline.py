"""Fit coupled left/right Frenet B-splines through centre and positive width.

The independent B-spline baseline fits the two sides separately.  That leaves
room for one sparse/noisy side to form a local bulge or cross the other curve.
This method instead models a smooth centre offset c(s) and log lane width
log(w(s)); the exported boundaries are c(s)-w(s)/2 and c(s)+w(s)/2.  Width is
therefore positive by construction, while all observations and validation still
come from the saved CLRNet/IPM points.  The reported errors are consistency,
not absolute lane accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from scipy.interpolate import splev, splrep

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import extended_curve_model_common as common  # noqa: E402
from scripts import run_weekly_lane_hierarchy as weekly  # noqa: E402


@dataclass
class CoupledSpline:
    centre_tck: tuple
    log_width_tck: tuple
    s_min: float
    s_max: float
    observed_width_m: np.ndarray
    centre_residual_m: np.ndarray
    log_width_residual: np.ndarray
    smoothing_per_point_m2: float
    width_smoothing_multiplier: float

    def evaluate(self, side: str, progress: np.ndarray) -> np.ndarray:
        centre = np.asarray(splev(progress, self.centre_tck), dtype=np.float64)
        width = np.exp(
            np.asarray(splev(progress, self.log_width_tck), dtype=np.float64)
        )
        if side == "left":
            return centre - 0.5 * width
        if side == "right":
            return centre + 0.5 * width
        raise ValueError(f"Unknown side: {side}")

    def width(self, progress: np.ndarray) -> np.ndarray:
        return np.exp(
            np.asarray(splev(progress, self.log_width_tck), dtype=np.float64)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_arguments(parser)
    parser.add_argument(
        "--smoothing-grid",
        default="0.0025,0.01,0.04,0.16",
        help="Positive centre-spline smoothing candidates in m^2 per point.",
    )
    parser.add_argument(
        "--width-smoothing-multiplier",
        type=float,
        default=4.0,
        help="Extra smoothing applied to log width relative to the centre.",
    )
    parser.add_argument(
        "--log-width-huber-delta",
        type=float,
        default=0.12,
        help="Huber threshold for dimensionless log-width residuals.",
    )
    return parser.parse_args()


def parse_smoothing_grid(value: str) -> list[float]:
    grid = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not grid or len(set(grid)) != len(grid) or any(item <= 0 for item in grid):
        raise ValueError("--smoothing-grid must contain unique positive values.")
    return grid


def paired_centre_width_observations(
    groups: list[dict[str, object]], bin_size_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create equally spaced centre/log-width observations from both sides."""

    left = weekly.aggregate_equal_group_bins(
        weekly.grouped_side_points(groups, "left"), bin_size_m
    )
    right = weekly.aggregate_equal_group_bins(
        weekly.grouped_side_points(groups, "right"), bin_size_m
    )
    lower = max(float(left[0, 0]), float(right[0, 0]))
    upper = min(float(left[-1, 0]), float(right[-1, 0]))
    if upper <= lower:
        raise ValueError("Left and right aggregate supports do not overlap.")
    start = np.ceil(lower / bin_size_m) * bin_size_m
    stop = np.floor(upper / bin_size_m) * bin_size_m
    progress = np.arange(start, stop + 0.5 * bin_size_m, bin_size_m)
    if len(progress) < 6:
        raise ValueError("Fewer than six paired centre/width bins are available.")
    left_d = np.interp(progress, left[:, 0], left[:, 1])
    right_d = np.interp(progress, right[:, 0], right[:, 1])
    width = right_d - left_d
    valid = np.isfinite(width) & (width > 0.0)
    if np.count_nonzero(valid) < 6:
        raise ValueError("Fewer than six positive observed-width bins are available.")
    progress = progress[valid]
    centre = 0.5 * (left_d[valid] + right_d[valid])
    width = width[valid]
    return progress, centre, width


def robust_scalar_spline(
    progress: np.ndarray,
    values: np.ndarray,
    smoothing_per_point: float,
    huber_delta: float,
    irls_iterations: int,
) -> tuple[tuple, np.ndarray]:
    weights = np.ones(len(progress), dtype=np.float64)
    tck = None
    residuals = np.zeros(len(progress), dtype=np.float64)
    for _ in range(max(1, irls_iterations)):
        smoothing = max(smoothing_per_point * len(progress), 1e-12)
        tck = splrep(progress, values, w=weights, k=3, s=smoothing)
        residuals = np.abs(values - np.asarray(splev(progress, tck)))
        weights = np.minimum(1.0, huber_delta / np.maximum(residuals, 1e-9))
        weights = np.maximum(weights, 0.05)
    assert tck is not None
    return tck, residuals


def fit_coupled(
    groups: list[dict[str, object]], smoothing: float, args: argparse.Namespace
) -> CoupledSpline:
    progress, centre, width = paired_centre_width_observations(
        groups, args.bin_size_m
    )
    centre_tck, centre_residuals = robust_scalar_spline(
        progress,
        centre,
        smoothing,
        args.huber_delta_m,
        args.irls_iterations,
    )
    log_width_tck, log_width_residuals = robust_scalar_spline(
        progress,
        np.log(width),
        smoothing * args.width_smoothing_multiplier,
        args.log_width_huber_delta,
        args.irls_iterations,
    )
    return CoupledSpline(
        centre_tck=centre_tck,
        log_width_tck=log_width_tck,
        s_min=float(progress[0]),
        s_max=float(progress[-1]),
        observed_width_m=width,
        centre_residual_m=centre_residuals,
        log_width_residual=log_width_residuals,
        smoothing_per_point_m2=smoothing,
        width_smoothing_multiplier=args.width_smoothing_multiplier,
    )


def cross_validate(
    groups: list[dict[str, object]], grid: list[float], args: argparse.Namespace
) -> tuple[float, list[dict[str, object]]]:
    rows = []
    fold_ids = weekly.fold_indices(len(groups), args.maximum_cv_folds)
    for smoothing in grid:
        fold_rmses = []
        all_errors = []
        failed_folds = 0
        for held_index in fold_ids:
            training = [group for index, group in enumerate(groups) if index != held_index]
            held = groups[held_index]
            try:
                fit = fit_coupled(training, smoothing, args)
            except ValueError:
                failed_folds += 1
                continue
            current = []
            for side_index, side in enumerate(weekly.SIDES):
                points = np.asarray(held["lanes_sd"][side_index], dtype=np.float64)
                supported = points[
                    (points[:, 0] >= fit.s_min) & (points[:, 0] <= fit.s_max)
                ]
                if len(supported):
                    current.append(
                        np.abs(supported[:, 1] - fit.evaluate(side, supported[:, 0]))
                    )
            if current:
                errors = np.concatenate(current)
                all_errors.append(errors)
                fold_rmses.append(float(np.sqrt(np.mean(errors**2))))
        if not fold_rmses:
            raise ValueError("Coupled B-spline cross-validation produced no folds.")
        combined = np.concatenate(all_errors)
        values = np.asarray(fold_rmses, dtype=np.float64)
        rows.append(
            {
                "model": "coupled centre + positive-width Frenet B-spline",
                "smoothing_per_point_m2": smoothing,
                "width_smoothing_multiplier": args.width_smoothing_multiplier,
                "fold_count": len(values),
                "failed_fold_count": failed_folds,
                "mean_m": float(np.mean(combined)),
                "rmse_m": float(np.sqrt(np.mean(combined**2))),
                "median_m": float(np.median(combined)),
                "p90_m": float(np.percentile(combined, 90)),
                "p95_m": float(np.percentile(combined, 95)),
                "max_m": float(np.max(combined)),
                "mean_fold_rmse_m": float(np.mean(values)),
                "fold_rmse_standard_error_m": float(
                    np.std(values, ddof=1) / np.sqrt(len(values))
                    if len(values) > 1
                    else 0.0
                ),
            }
        )
    minimum = min(rows, key=lambda row: float(row["mean_fold_rmse_m"]))
    limit = float(minimum["mean_fold_rmse_m"]) + float(
        minimum["fold_rmse_standard_error_m"]
    )
    eligible = [row for row in rows if float(row["mean_fold_rmse_m"]) <= limit]
    selected = max(eligible, key=lambda row: float(row["smoothing_per_point_m2"]))
    for row in rows:
        row["selected_by_one_standard_error_rule"] = row is selected
        row["one_standard_error_limit_m"] = limit
    return float(selected["smoothing_per_point_m2"]), rows


def sampled_curves(
    fit: CoupledSpline, data: common.ExtendedCurveInput, samples: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    progress = np.linspace(fit.s_min, fit.s_max, samples)
    curves = {}
    for side in weekly.SIDES:
        sd = np.column_stack([progress, fit.evaluate(side, progress)])
        curves[side] = weekly.frenet_to_xz(sd, data.reference_path)
    return progress, curves


def export_curves(
    output_dir: Path,
    fit: CoupledSpline,
    data: common.ExtendedCurveInput,
    samples: int,
) -> None:
    progress, curves = sampled_curves(fit, data, samples)
    for side in weekly.SIDES:
        offsets = fit.evaluate(side, progress)
        weekly.write_csv(
            output_dir / f"{side}_curve.csv",
            [
                {
                    "sample_index": index,
                    "s_path_m": s,
                    "d_right_m": d,
                    "x_right_reference_m": xz[0],
                    "z_forward_reference_m": xz[1],
                }
                for index, (s, d, xz) in enumerate(
                    zip(progress, offsets, curves[side])
                )
            ],
        )


def plot_result(
    path: Path,
    data: common.ExtendedCurveInput,
    fit: CoupledSpline,
    samples: int,
) -> None:
    progress, curves = sampled_curves(fit, data, samples)
    centre_sd = np.column_stack(
        [progress, np.asarray(splev(progress, fit.centre_tck), dtype=np.float64)]
    )
    centre_xz = weekly.frenet_to_xz(centre_sd, data.reference_path)
    colors = {"left": "#1f77b4", "right": "#d62728"}
    figure, axis = plt.subplots(figsize=(7.4, 9.0))
    for side_index, side in enumerate(weekly.SIDES):
        raw = np.vstack(
            [np.asarray(frame["lanes_xz"][side_index]) for frame in data.frames]
        )
        axis.scatter(raw[:, 0], raw[:, 1], s=3, alpha=0.12, color=colors[side])
        axis.plot(
            curves[side][:, 0],
            curves[side][:, 1],
            linewidth=2.2,
            color=colors[side],
            label=f"{side} coupled B-spline",
        )
    axis.plot(
        centre_xz[:, 0],
        centre_xz[:, 1],
        color="0.25",
        linewidth=1.0,
        linestyle="--",
        label="fitted lane centre",
    )
    axis.set_xlabel("X right in selected reference frame [m]")
    axis.set_ylabel("Z forward in selected reference frame [m]")
    axis.set_title("Coupled centre/positive-width robust B-spline")
    axis.grid(True, linewidth=0.5, alpha=0.35)
    axis.legend()
    axis.set_aspect("equal", adjustable="datalim")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.width_smoothing_multiplier <= 0:
        raise ValueError("--width-smoothing-multiplier must be positive.")
    if args.log_width_huber_delta <= 0:
        raise ValueError("--log-width-huber-delta must be positive.")
    grid = parse_smoothing_grid(args.smoothing_grid)
    data = common.load_input(args)
    selected_smoothing, cv_rows = cross_validate(data.groups, grid, args)
    fit = fit_coupled(data.groups, selected_smoothing, args)

    progress = np.linspace(fit.s_min, fit.s_max, args.curve_samples)
    fitted_width = fit.width(progress)
    width_check = {
        "positive_by_construction": True,
        "curves_cross": False,
        "observed_width_median_m": float(np.median(fit.observed_width_m)),
        "observed_width_p10_m": float(np.percentile(fit.observed_width_m, 10)),
        "observed_width_p90_m": float(np.percentile(fit.observed_width_m, 90)),
        "fitted_width_minimum_m": float(np.min(fitted_width)),
        "fitted_width_median_m": float(np.median(fitted_width)),
        "fitted_width_maximum_m": float(np.max(fitted_width)),
        "note": "Width is data-driven and smoothed in log space; no fixed KITTI lane-width truth is assumed.",
    }

    weekly.write_csv(
        args.output_dir / "00_audit" / "frame_validity.csv",
        data.frame_audit.values(),
    )
    weekly.write_csv(
        args.output_dir / "01_evaluation" / "coupled_smoothing_cv.csv", cv_rows
    )
    export_curves(args.output_dir / "02_curves", fit, data, args.curve_samples)
    plot_result(
        args.output_dir / "03_figures" / "coupled_bspline_left_right.png",
        data,
        fit,
        args.curve_samples,
    )

    result = {
        "status": "complete",
        "method_scope": "coupled centre/positive-width robust Frenet B-spline",
        **common.audit_document(data, args),
        "smoothing_grid_m2_per_point": grid,
        "selected_smoothing_per_point_m2": selected_smoothing,
        "width_smoothing_multiplier": args.width_smoothing_multiplier,
        "selection_rule": "one-standard-error rule on held-out-frame RMSE",
        "cross_validation": cv_rows,
        "width_check": width_check,
        "result_image": str(
            (
                args.output_dir
                / "03_figures"
                / "coupled_bspline_left_right.png"
            ).resolve()
        ),
        "warning": "Held-out errors measure consistency with CLRNet/IPM-derived points, not absolute lane accuracy.",
    }
    common.write_json(args.output_dir / "RESULT.json", result)
    common.write_json(
        args.output_dir / "STATUS.json",
        {
            "status": "complete",
            "method": "coupled_bspline",
            "frames": [data.start_frame, data.end_frame],
            "valid_frames": len(data.frames),
            "selected_smoothing_per_point_m2": selected_smoothing,
            "curves_cross": False,
            "previous_outputs_modified": False,
        },
    )
    print(
        f"COUPLED B-SPLINE FINISHED: frames {data.start_frame}-{data.end_frame}, "
        f"valid {len(data.frames)}, smoothing {selected_smoothing}, output {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
