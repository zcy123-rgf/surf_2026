"""Fit support-aware overlapping local Frenet B-splines for a long bend.

Unlike one global B-spline, this model never extrapolates a local curve across
an unsupported interior.  Robust local cubic splines are blended only where
their progress supports overlap.  Candidate window/smoothing settings are
selected by held-out-frame consistency with a registered support-coverage gate.
The metrics remain consistency with CLRNet/IPM-derived points, not accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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


@dataclass
class PiecewiseCurve:
    side: str
    local_fits: list[weekly.ScalarSpline]

    @property
    def s_min(self) -> float:
        return min(fit.s_min for fit in self.local_fits)

    @property
    def s_max(self) -> float:
        return max(fit.s_max for fit in self.local_fits)

    def evaluate(self, progress: np.ndarray) -> np.ndarray:
        values = np.asarray(progress, dtype=np.float64)
        numerator = np.zeros_like(values)
        denominator = np.zeros_like(values)
        for fit in self.local_fits:
            mask = (values >= fit.s_min) & (values <= fit.s_max)
            if not np.any(mask):
                continue
            span = max(fit.s_max - fit.s_min, 1e-9)
            phase = (values[mask] - fit.s_min) / span
            weights = 0.05 + np.sin(np.pi * phase) ** 2
            numerator[mask] += weights * fit.evaluate(values[mask])
            denominator[mask] += weights
        output = np.full_like(values, np.nan)
        supported = denominator > 0.0
        output[supported] = numerator[supported] / denominator[supported]
        return output

    def supported(self, progress: np.ndarray) -> np.ndarray:
        values = np.asarray(progress, dtype=np.float64)
        return np.logical_or.reduce(
            [(values >= fit.s_min) & (values <= fit.s_max) for fit in self.local_fits]
        )


@dataclass
class PiecewisePair:
    curves: dict[str, PiecewiseCurve]
    window_length_m: float
    window_stride_m: float
    smoothing_per_point_m2: float
    attempted_windows: int
    completed_windows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_arguments(parser)
    parser.add_argument(
        "--window-grid",
        default="20:10,30:15,40:20",
        help="Comma-separated LENGTH:STRIDE candidates in path metres.",
    )
    parser.add_argument(
        "--smoothing-grid",
        default="0.0025,0.01,0.04",
        help="Positive local B-spline smoothing values.",
    )
    parser.add_argument("--minimum-local-bins", type=int, default=8)
    parser.add_argument("--minimum-heldout-support-rate", type=float, default=0.85)
    return parser.parse_args()


def parse_window_grid(value: str) -> list[tuple[float, float]]:
    result = []
    for item in value.split(","):
        parts = [float(part.strip()) for part in item.split(":")]
        if len(parts) != 2 or parts[0] <= 0 or parts[1] <= 0 or parts[1] >= parts[0]:
            raise ValueError("Every --window-grid item must be LENGTH:STRIDE with 0 < stride < length.")
        result.append((parts[0], parts[1]))
    if not result or len(set(result)) != len(result):
        raise ValueError("--window-grid must contain unique candidates.")
    return result


def parse_smoothing_grid(value: str) -> list[float]:
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not result or len(set(result)) != len(result) or any(item <= 0 for item in result):
        raise ValueError("--smoothing-grid must contain unique positive values.")
    return result


def progress_extent(groups: list[dict[str, object]]) -> tuple[float, float]:
    extents = []
    for side_index in range(2):
        points = np.vstack(
            [np.asarray(group["lanes_sd"][side_index], dtype=np.float64) for group in groups]
        )
        extents.append((float(np.min(points[:, 0])), float(np.max(points[:, 0]))))
    lower = max(item[0] for item in extents)
    upper = min(item[1] for item in extents)
    if upper <= lower:
        raise ValueError("Left/right progress supports do not overlap.")
    return lower, upper


def progress_windows(
    lower: float, upper: float, length: float, stride: float
) -> list[tuple[float, float]]:
    if upper - lower <= length:
        return [(lower, upper)]
    starts = list(np.arange(lower, upper - length + 1e-9, stride))
    final_start = upper - length
    if not starts or final_start - starts[-1] > 1e-6:
        starts.append(final_start)
    return [(float(start), float(min(start + length, upper))) for start in starts]


def restrict_groups(
    groups: list[dict[str, object]], lower: float, upper: float
) -> list[dict[str, object]]:
    output = []
    for group in groups:
        lanes = []
        for side_index in range(2):
            points = np.asarray(group["lanes_sd"][side_index], dtype=np.float64)
            lanes.append(points[(points[:, 0] >= lower) & (points[:, 0] <= upper)])
        if min(map(len, lanes), default=0) >= 2:
            output.append({"group_id": group["group_id"], "lanes_sd": lanes})
    return output


def fit_piecewise_pair(
    groups: list[dict[str, object]],
    length: float,
    stride: float,
    smoothing: float,
    args: argparse.Namespace,
) -> PiecewisePair:
    lower, upper = progress_extent(groups)
    windows = progress_windows(lower, upper, length, stride)
    side_fits: dict[str, list[weekly.ScalarSpline]] = {side: [] for side in weekly.SIDES}
    completed = 0
    for window_lower, window_upper in windows:
        local = restrict_groups(groups, window_lower, window_upper)
        if len(local) < 2:
            continue
        fits = {}
        try:
            for side in weekly.SIDES:
                aggregate = weekly.aggregate_equal_group_bins(
                    weekly.grouped_side_points(local, side), args.bin_size_m
                )
                if len(aggregate) < args.minimum_local_bins:
                    raise ValueError("too few local bins")
                fits[side] = weekly.fit_scalar_spline(
                    side,
                    weekly.grouped_side_points(local, side),
                    args.bin_size_m,
                    smoothing,
                    args.huber_delta_m,
                    args.irls_iterations,
                )
        except ValueError:
            continue
        for side in weekly.SIDES:
            side_fits[side].append(fits[side])
        completed += 1
    if completed < 2 or any(len(side_fits[side]) < 2 for side in weekly.SIDES):
        raise ValueError("Fewer than two local windows completed for both sides.")
    return PiecewisePair(
        curves={
            side: PiecewiseCurve(side=side, local_fits=side_fits[side])
            for side in weekly.SIDES
        },
        window_length_m=length,
        window_stride_m=stride,
        smoothing_per_point_m2=smoothing,
        attempted_windows=len(windows),
        completed_windows=completed,
    )


def pair_width_check(pair: PiecewisePair, samples: int = 1000) -> dict[str, object]:
    lower = max(pair.curves[side].s_min for side in weekly.SIDES)
    upper = min(pair.curves[side].s_max for side in weekly.SIDES)
    progress = np.linspace(lower, upper, samples)
    left = pair.curves["left"].evaluate(progress)
    right = pair.curves["right"].evaluate(progress)
    supported = np.isfinite(left) & np.isfinite(right)
    if not np.any(supported):
        return {"supported_samples": 0, "curves_cross": None}
    width = right[supported] - left[supported]
    return {
        "supported_samples": int(np.count_nonzero(supported)),
        "sample_count": samples,
        "supported_fraction": float(np.mean(supported)),
        "minimum_right_minus_left_m": float(np.min(width)),
        "median_right_minus_left_m": float(np.median(width)),
        "maximum_right_minus_left_m": float(np.max(width)),
        "curves_cross": bool(np.any(width <= 0.0)),
    }


def evaluate_held_group(
    pair: PiecewisePair, group: dict[str, object]
) -> tuple[np.ndarray, int, int]:
    errors = []
    total = 0
    supported_count = 0
    for side_index, side in enumerate(weekly.SIDES):
        points = np.asarray(group["lanes_sd"][side_index], dtype=np.float64)
        total += len(points)
        predicted = pair.curves[side].evaluate(points[:, 0])
        supported = np.isfinite(predicted)
        supported_count += int(np.count_nonzero(supported))
        if np.any(supported):
            errors.append(np.abs(points[supported, 1] - predicted[supported]))
    return (
        np.concatenate(errors) if errors else np.empty(0, dtype=np.float64),
        supported_count,
        total,
    )


def evaluate_candidate(
    groups: list[dict[str, object]],
    length: float,
    stride: float,
    smoothing: float,
    args: argparse.Namespace,
) -> dict[str, object]:
    fold_rmses = []
    all_errors = []
    supported_count = 0
    total_count = 0
    failed_folds = 0
    for held_index in weekly.fold_indices(len(groups), args.maximum_cv_folds):
        training = [group for index, group in enumerate(groups) if index != held_index]
        try:
            pair = fit_piecewise_pair(training, length, stride, smoothing, args)
        except ValueError:
            failed_folds += 1
            continue
        errors, supported, total = evaluate_held_group(pair, groups[held_index])
        supported_count += supported
        total_count += total
        if len(errors):
            all_errors.append(errors)
            fold_rmses.append(float(np.sqrt(np.mean(errors**2))))
    if not fold_rmses:
        raise ValueError("Piecewise cross-validation produced no supported folds.")
    combined = np.concatenate(all_errors)
    values = np.asarray(fold_rmses, dtype=np.float64)
    final_pair = fit_piecewise_pair(groups, length, stride, smoothing, args)
    width = pair_width_check(final_pair)
    return {
        "model": "overlap-blended local robust Frenet B-splines",
        "window_length_m": length,
        "window_stride_m": stride,
        "smoothing_per_point_m2": smoothing,
        "fold_count": len(values),
        "failed_fold_count": failed_folds,
        "heldout_supported_point_fraction": (
            supported_count / total_count if total_count else 0.0
        ),
        "mean_m": float(np.mean(combined)),
        "rmse_m": float(np.sqrt(np.mean(combined**2))),
        "median_m": float(np.median(combined)),
        "p90_m": float(np.percentile(combined, 90)),
        "p95_m": float(np.percentile(combined, 95)),
        "max_m": float(np.max(combined)),
        "mean_fold_rmse_m": float(np.mean(values)),
        "fold_rmse_standard_error_m": float(
            np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
        ),
        "completed_local_windows": final_pair.completed_windows,
        "attempted_local_windows": final_pair.attempted_windows,
        "curves_cross": width["curves_cross"],
        "width_supported_fraction": width.get("supported_fraction"),
        "_pair": final_pair,
        "_width": width,
    }


def export_curves(
    output_dir: Path,
    pair: PiecewisePair,
    path: weekly.FrenetPath,
    samples: int,
) -> None:
    lower = max(pair.curves[side].s_min for side in weekly.SIDES)
    upper = min(pair.curves[side].s_max for side in weekly.SIDES)
    progress = np.linspace(lower, upper, samples)
    for side in weekly.SIDES:
        offset = pair.curves[side].evaluate(progress)
        supported = np.isfinite(offset)
        sd = np.column_stack([progress[supported], offset[supported]])
        xz = weekly.frenet_to_xz(sd, path)
        weekly.write_csv(
            output_dir / f"{side}_curve.csv",
            [
                {
                    "sample_index": index,
                    "s_path_m": point_sd[0],
                    "d_right_m": point_sd[1],
                    "x_right_reference_m": point_xz[0],
                    "z_forward_reference_m": point_xz[1],
                }
                for index, (point_sd, point_xz) in enumerate(zip(sd, xz))
            ],
        )


def plot_result(
    output: Path,
    data: common.ExtendedCurveInput,
    pair: PiecewisePair,
    samples: int,
) -> None:
    colors = {"left": "#1f77b4", "right": "#d62728"}
    figure, axis = plt.subplots(figsize=(7.4, 9.0))
    lower = max(pair.curves[side].s_min for side in weekly.SIDES)
    upper = min(pair.curves[side].s_max for side in weekly.SIDES)
    progress = np.linspace(lower, upper, samples)
    for side_index, side in enumerate(weekly.SIDES):
        raw = np.vstack(
            [np.asarray(frame["lanes_xz"][side_index]) for frame in data.frames]
        )
        axis.scatter(raw[:, 0], raw[:, 1], s=3, alpha=0.12, color=colors[side])
        offset = pair.curves[side].evaluate(progress)
        supported = np.isfinite(offset)
        # NaN gaps deliberately break the plotted line rather than inventing data.
        curve_sd = np.column_stack([progress, offset])
        curve_xz = np.full_like(curve_sd, np.nan)
        curve_xz[supported] = weekly.frenet_to_xz(
            curve_sd[supported], data.reference_path
        )
        axis.plot(
            curve_xz[:, 0], curve_xz[:, 1], linewidth=2.2, color=colors[side],
            label=f"{side} piecewise B-spline",
        )
    axis.set_xlabel("X right in selected reference frame [m]")
    axis.set_ylabel("Z forward in selected reference frame [m]")
    axis.set_title("Support-aware overlapping local robust B-splines")
    axis.grid(True, linewidth=0.5, alpha=0.35)
    axis.legend()
    axis.set_aspect("equal", adjustable="datalim")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.minimum_local_bins < 6:
        raise ValueError("--minimum-local-bins must be at least six.")
    if not 0.0 < args.minimum_heldout_support_rate <= 1.0:
        raise ValueError("--minimum-heldout-support-rate must be in (0, 1].")
    window_grid = parse_window_grid(args.window_grid)
    smoothing_grid = parse_smoothing_grid(args.smoothing_grid)
    data = common.load_input(args)

    trials = []
    for length, stride in window_grid:
        for smoothing in smoothing_grid:
            try:
                trials.append(
                    evaluate_candidate(data.groups, length, stride, smoothing, args)
                )
            except ValueError as error:
                trials.append(
                    {
                        "model": "overlap-blended local robust Frenet B-splines",
                        "window_length_m": length,
                        "window_stride_m": stride,
                        "smoothing_per_point_m2": smoothing,
                        "status": "failed",
                        "error": str(error),
                        "heldout_supported_point_fraction": 0.0,
                        "curves_cross": None,
                    }
                )
    successful = [row for row in trials if "_pair" in row]
    if not successful:
        raise RuntimeError("Every piecewise candidate failed.")
    eligible = [
        row for row in successful
        if float(row["heldout_supported_point_fraction"])
        >= args.minimum_heldout_support_rate
        and row["curves_cross"] is False
    ]
    pool = eligible or successful
    selected = min(pool, key=lambda row: float(row["mean_fold_rmse_m"]))
    selected_pair = selected["_pair"]
    width = selected["_width"]
    selection_gate_passed = selected in eligible

    serializable_trials = []
    for row in trials:
        serializable_trials.append(
            {key: value for key, value in row.items() if not key.startswith("_")}
        )
    for row in serializable_trials:
        row["selected"] = (
            row.get("window_length_m") == selected["window_length_m"]
            and row.get("window_stride_m") == selected["window_stride_m"]
            and row.get("smoothing_per_point_m2")
            == selected["smoothing_per_point_m2"]
        )

    weekly.write_csv(
        args.output_dir / "00_audit" / "frame_validity.csv",
        data.frame_audit.values(),
    )
    weekly.write_csv(
        args.output_dir / "01_evaluation" / "piecewise_model_trials.csv",
        serializable_trials,
    )
    export_curves(
        args.output_dir / "02_curves",
        selected_pair,
        data.reference_path,
        args.curve_samples,
    )
    plot_result(
        args.output_dir / "03_figures" / "piecewise_bspline_left_right.png",
        data,
        selected_pair,
        args.curve_samples,
    )

    selected_serializable = {
        key: value for key, value in selected.items() if not key.startswith("_")
    }
    result = {
        "status": "complete",
        "method_scope": "support-aware overlapping local robust Frenet B-splines",
        **common.audit_document(data, args),
        "registered_candidates": {
            "window_grid_length_stride_m": window_grid,
            "smoothing_grid_m2_per_point": smoothing_grid,
            "minimum_heldout_support_rate": args.minimum_heldout_support_rate,
            "minimum_local_bins": args.minimum_local_bins,
        },
        "selection_gate_passed": selection_gate_passed,
        "selected_model": selected_serializable,
        "width_check": width,
        "model_trials": serializable_trials,
        "result_image": str(
            (
                args.output_dir
                / "03_figures"
                / "piecewise_bspline_left_right.png"
            ).resolve()
        ),
        "warnings": [
            "NaN support gaps are not bridged or plotted as observed lane.",
            "Held-out errors measure consistency with CLRNet/IPM points, not absolute lane accuracy.",
            "If selection_gate_passed is false, this output is diagnostic and must not be adopted.",
        ],
        "previous_outputs_modified": False,
    }
    common.write_json(args.output_dir / "RESULT.json", result)
    common.write_json(
        args.output_dir / "STATUS.json",
        {
            "status": "complete",
            "valid_frames": len(data.frames),
            "selection_gate_passed": selection_gate_passed,
            "selected_window_length_m": selected["window_length_m"],
            "selected_window_stride_m": selected["window_stride_m"],
            "selected_smoothing_per_point_m2": selected[
                "smoothing_per_point_m2"
            ],
            "previous_outputs_modified": False,
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "selection_gate_passed": selection_gate_passed,
                "selected_mean_fold_rmse_m": selected["mean_fold_rmse_m"],
                "output": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
