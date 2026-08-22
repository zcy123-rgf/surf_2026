"""Fit a long road containing curves with overlapping local lane models.

The input is a selection produced by ``select_hierarchy_150_frames.py`` with
an arbitrary number of 15-frame windows.  Saved CLRNet/IPM points are reused;
CLRNet is not rerun.  Polynomial and cubic B-spline models are fitted in
separate local-window branches.  Adjacent windows are blended only while their
supports overlap and the registered continuity gates are satisfied.  A failed
gate starts a new road segment instead of forcing a single global curve.

All reported errors are consistency diagnostics against CLRNet-derived points,
not lane ground-truth accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
class LocalCurve:
    method: str
    side: str
    s_min: float
    s_max: float
    evaluate_function: Callable[[np.ndarray], np.ndarray]
    derivative_function: Callable[[np.ndarray], np.ndarray]

    def evaluate(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(self.evaluate_function(values), dtype=np.float64)

    def derivative(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(self.derivative_function(values), dtype=np.float64)


@dataclass
class BlendedCurve:
    method: str
    side: str
    curves: list[LocalCurve]
    s_min: float
    s_max: float

    def evaluate(self, values: np.ndarray) -> np.ndarray:
        progress = np.asarray(values, dtype=np.float64)
        numerator = np.zeros_like(progress)
        denominator = np.zeros_like(progress)
        for curve in self.curves:
            mask = (progress >= curve.s_min) & (progress <= curve.s_max)
            if not np.any(mask):
                continue
            phase = (progress[mask] - curve.s_min) / max(
                curve.s_max - curve.s_min, 1e-9
            )
            weights = np.sin(np.pi * phase) ** 2
            numerator[mask] += weights * curve.evaluate(progress[mask])
            denominator[mask] += weights

        missing = denominator <= 1e-12
        if np.any(missing):
            for index in np.flatnonzero(missing):
                value = float(progress.flat[index])
                containing = [
                    curve
                    for curve in self.curves
                    if curve.s_min <= value <= curve.s_max
                ]
                if not containing:
                    raise ValueError(
                        f"{self.method} {self.side} has an uncovered support "
                        f"at s={value:.3f} m."
                    )
                selected = min(
                    containing,
                    key=lambda curve: abs(
                        value - (curve.s_min + curve.s_max) / 2.0
                    ),
                )
                numerator.flat[index] = float(selected.evaluate(np.array([value]))[0])
                denominator.flat[index] = 1.0
        return numerator / denominator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_arguments(parser)
    parser.add_argument("--degrees", default="1,2,3")
    parser.add_argument("--smoothing-grid", default="0.0025,0.01,0.04,0.16")
    parser.add_argument(
        "--maximum-overlap-p95-gap-m",
        type=float,
        default=1.50,
        help=(
            "Heuristic road-segmentation gate.  A larger adjacent-window "
            "B-spline P95 lateral gap starts a new segment."
        ),
    )
    parser.add_argument(
        "--maximum-overlap-p95-tangent-gap-deg",
        type=float,
        default=20.0,
        help=(
            "Heuristic road-segmentation gate.  A larger adjacent-window "
            "B-spline P95 tangent gap starts a new segment."
        ),
    )
    return parser.parse_args()


def parse_int_grid(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("--degrees must contain unique values.")
    if any(item < 1 or item > 5 for item in result):
        raise ValueError("Polynomial degrees must be between one and five.")
    return result


def parse_float_grid(value: str) -> list[float]:
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not result or len(result) != len(set(result)) or any(item <= 0 for item in result):
        raise ValueError("--smoothing-grid must contain unique positive values.")
    return result


def polynomial_curve(
    groups: list[dict[str, object]],
    side: str,
    degree: int,
    args: argparse.Namespace,
) -> LocalCurve:
    aggregate = weekly.aggregate_equal_group_bins(
        weekly.grouped_side_points(groups, side), args.bin_size_m
    )
    coefficients = weekly.robust_polyfit(aggregate, degree, args)
    derivative = np.polyder(coefficients)
    return LocalCurve(
        method="polynomial",
        side=side,
        s_min=float(aggregate[0, 0]),
        s_max=float(aggregate[-1, 0]),
        evaluate_function=lambda values, c=coefficients: np.polyval(c, values),
        derivative_function=lambda values, c=derivative: np.polyval(c, values),
    )


def spline_curve(fit: weekly.ScalarSpline) -> LocalCurve:
    return LocalCurve(
        method="bspline",
        side=fit.side,
        s_min=float(fit.s_min),
        s_max=float(fit.s_max),
        evaluate_function=lambda values, spline=fit: spline.evaluate(values),
        derivative_function=lambda values, spline=fit: np.asarray(
            weekly.splev(values, spline.tck, der=1), dtype=np.float64
        ),
    )


def overlap_diagnostics(first: LocalCurve, second: LocalCurve) -> dict[str, object]:
    lower = max(first.s_min, second.s_min)
    upper = min(first.s_max, second.s_max)
    if upper <= lower:
        return {
            "overlap_available": False,
            "overlap_s_m": 0.0,
            "p95_position_gap_m": None,
            "p95_tangent_gap_deg": None,
        }
    progress = np.linspace(lower, upper, 120)
    position = np.abs(first.evaluate(progress) - second.evaluate(progress))
    first_angle = np.degrees(np.arctan(first.derivative(progress)))
    second_angle = np.degrees(np.arctan(second.derivative(progress)))
    tangent = np.abs(first_angle - second_angle)
    tangent = np.minimum(tangent, 180.0 - tangent)
    return {
        "overlap_available": True,
        "overlap_s_m": float(upper - lower),
        "mean_position_gap_m": float(np.mean(position)),
        "p95_position_gap_m": float(np.percentile(position, 95)),
        "mean_tangent_gap_deg": float(np.mean(tangent)),
        "p95_tangent_gap_deg": float(np.percentile(tangent, 95)),
    }


def segment_windows(
    windows: list[dict[str, object]],
    maximum_position_gap_m: float,
    maximum_tangent_gap_deg: float,
) -> tuple[list[list[dict[str, object]]], list[dict[str, object]]]:
    """Split chronologically ordered windows using B-spline overlap gates."""

    if not windows:
        return [], []
    chains = [[windows[0]]]
    audit = []
    for first, second in zip(windows[:-1], windows[1:]):
        reasons = []
        side_rows = []
        for side in weekly.SIDES:
            metrics = overlap_diagnostics(
                first["curves"]["bspline"][side],
                second["curves"]["bspline"][side],
            )
            if not metrics["overlap_available"]:
                reasons.append(f"{side}:no_support_overlap")
            else:
                if float(metrics["p95_position_gap_m"]) > maximum_position_gap_m:
                    reasons.append(f"{side}:position_gap")
                if float(metrics["p95_tangent_gap_deg"]) > maximum_tangent_gap_deg:
                    reasons.append(f"{side}:tangent_gap")
            side_rows.append(
                {
                    "first_window": int(first["window_index"]),
                    "second_window": int(second["window_index"]),
                    "side": side,
                    **metrics,
                }
            )
        start_new = bool(reasons)
        for row in side_rows:
            row["starts_new_road_segment"] = start_new
            row["break_reasons"] = ";".join(reasons)
            row["registered_p95_position_limit_m"] = maximum_position_gap_m
            row["registered_p95_tangent_limit_deg"] = maximum_tangent_gap_deg
            audit.append(row)
        if start_new:
            chains.append([second])
        else:
            chains[-1].append(second)
    return chains, audit


def blend_chain(
    chain: list[dict[str, object]], method: str
) -> dict[str, BlendedCurve]:
    result = {}
    for side in weekly.SIDES:
        curves = [window["curves"][method][side] for window in chain]
        result[side] = BlendedCurve(
            method=method,
            side=side,
            curves=curves,
            s_min=min(curve.s_min for curve in curves),
            s_max=max(curve.s_max for curve in curves),
        )
    return result


def export_segments(
    output_dir: Path,
    method: str,
    chains: list[list[dict[str, object]]],
    path: weekly.FrenetPath,
    samples: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    segment_rows = []
    curve_rows = []
    for segment_index, chain in enumerate(chains):
        fits = blend_chain(chain, method)
        segment_rows.append(
            {
                "method": method,
                "road_segment_index": segment_index,
                "first_window": int(chain[0]["window_index"]),
                "last_window": int(chain[-1]["window_index"]),
                "start_frame": int(chain[0]["start_frame"]),
                "end_frame": int(chain[-1]["end_frame"]),
                "window_count": len(chain),
                "left_s_min_m": fits["left"].s_min,
                "left_s_max_m": fits["left"].s_max,
                "right_s_min_m": fits["right"].s_min,
                "right_s_max_m": fits["right"].s_max,
            }
        )
        for side in weekly.SIDES:
            fit = fits[side]
            count = max(120, int(samples / max(len(chains), 1)))
            progress = np.linspace(fit.s_min, fit.s_max, count)
            curve_sd = np.column_stack([progress, fit.evaluate(progress)])
            curve_xz = weekly.frenet_to_xz(curve_sd, path)
            for sample_index, (sd, xz) in enumerate(zip(curve_sd, curve_xz)):
                curve_rows.append(
                    {
                        "method": method,
                        "road_segment_index": segment_index,
                        "side": side,
                        "sample_index": sample_index,
                        "s_path_m": float(sd[0]),
                        "d_right_m": float(sd[1]),
                        "x_right_reference_m": float(xz[0]),
                        "z_forward_reference_m": float(xz[1]),
                    }
                )
    weekly.write_csv(output_dir / method / "road_segments.csv", segment_rows)
    weekly.write_csv(output_dir / method / "curve_points.csv", curve_rows)
    return segment_rows, curve_rows


def plot_method(
    output_path: Path,
    method: str,
    data: common.ExtendedCurveInput,
    curve_rows: list[dict[str, object]],
) -> None:
    colors = {"left": "#1f77b4", "right": "#d62728"}
    figure, axis = plt.subplots(figsize=(9.0, 8.0))
    for side_index, side in enumerate(weekly.SIDES):
        raw = np.vstack(
            [np.asarray(frame["lanes_xz"][side_index]) for frame in data.frames]
        )
        axis.scatter(raw[:, 0], raw[:, 1], s=2, alpha=0.07, color=colors[side])
        segments = sorted(
            {int(row["road_segment_index"]) for row in curve_rows if row["side"] == side}
        )
        for segment in segments:
            rows = [
                row
                for row in curve_rows
                if row["side"] == side
                and int(row["road_segment_index"]) == segment
            ]
            axis.plot(
                [float(row["x_right_reference_m"]) for row in rows],
                [float(row["z_forward_reference_m"]) for row in rows],
                linewidth=2.0,
                color=colors[side],
                label=f"{side} {method}" if segment == segments[0] else None,
            )
    axis.set_xlabel("X right in selected reference frame [m]")
    axis.set_ylabel("Z forward in selected reference frame [m]")
    axis.set_title(
        f"Long curved road: overlap-blended local {method} curves"
    )
    axis.grid(True, linewidth=0.5, alpha=0.35)
    axis.legend()
    axis.set_aspect("equal", adjustable="datalim")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.maximum_overlap_p95_gap_m <= 0:
        raise ValueError("maximum overlap position gap must be positive.")
    if args.maximum_overlap_p95_tangent_gap_deg <= 0:
        raise ValueError("maximum overlap tangent gap must be positive.")
    degrees = parse_int_grid(args.degrees)
    smoothing_grid = parse_float_grid(args.smoothing_grid)
    data = common.load_input(args)
    selection = data.selection
    block_size = int(selection["block_size"])
    block_count = int(selection["block_count"])
    block_stride = int(selection["block_stride"])
    if block_size != 15 or block_stride != 10 or block_count < 2:
        raise ValueError(
            "Long-road experiment requires 15-frame windows, stride 10, "
            "and at least two windows."
        )

    polynomial_cv = [
        weekly.polynomial_cross_validation(data.groups, degree, args)
        for degree in degrees
    ]
    best_polynomial = min(
        polynomial_cv, key=lambda row: float(row["mean_fold_rmse_m"])
    )
    best_degree = int(str(best_polynomial["model"]).rsplit(" ", 1)[-1])
    selected_smoothing, spline_cv = weekly.spline_cross_validation(
        data.groups, smoothing_grid, args
    )

    by_id = {int(frame["frame_id"]): frame for frame in data.frames}
    windows = []
    window_rows = []
    minimum_valid = int(selection["minimum_valid_frames_per_block_required"])
    for block in selection["blocks"]:
        window_index = int(block["block_index"])
        start = int(block["start_frame"])
        end = int(block["end_frame"])
        valid_ids = [frame_id for frame_id in range(start, end + 1) if frame_id in by_id]
        if len(valid_ids) < minimum_valid:
            raise ValueError(
                f"Window {window_index} has {len(valid_ids)} valid frames; "
                f"registered minimum is {minimum_valid}."
            )
        groups = [
            {"group_id": frame_id, "lanes_sd": by_id[frame_id]["lanes_sd"]}
            for frame_id in valid_ids
        ]
        spline_fits = weekly.fit_both_sides(groups, selected_smoothing, args)
        curves = {
            "polynomial": {
                side: polynomial_curve(groups, side, best_degree, args)
                for side in weekly.SIDES
            },
            "bspline": {
                side: spline_curve(spline_fits[side]) for side in weekly.SIDES
            },
        }
        windows.append(
            {
                "window_index": window_index,
                "start_frame": start,
                "end_frame": end,
                "valid_ids": valid_ids,
                "curves": curves,
            }
        )
        window_rows.append(
            {
                "window_index": window_index,
                "start_frame": start,
                "end_frame": end,
                "valid_frame_count": len(valid_ids),
                "valid_frame_ids": ";".join(map(str, valid_ids)),
            }
        )

    chains, continuity_rows = segment_windows(
        windows,
        args.maximum_overlap_p95_gap_m,
        args.maximum_overlap_p95_tangent_gap_deg,
    )
    weekly.write_csv(args.output_dir / "00_audit" / "windows.csv", window_rows)
    weekly.write_csv(
        args.output_dir / "00_audit" / "adjacent_window_continuity.csv",
        continuity_rows,
    )
    weekly.write_csv(
        args.output_dir / "01_model_selection" / "polynomial_cv.csv",
        polynomial_cv,
    )
    weekly.write_csv(
        args.output_dir / "01_model_selection" / "bspline_cv.csv",
        spline_cv,
    )

    outputs = {}
    for method in ("polynomial", "bspline"):
        segment_rows, curve_rows = export_segments(
            args.output_dir / "02_piecewise_curves",
            method,
            chains,
            data.reference_path,
            args.curve_samples,
        )
        figure_path = args.output_dir / "03_figures" / f"{method}_long_road.png"
        plot_method(figure_path, method, data, curve_rows)
        outputs[method] = {
            "road_segment_count": len(segment_rows),
            "segments_csv": str(
                (
                    args.output_dir
                    / "02_piecewise_curves"
                    / method
                    / "road_segments.csv"
                ).resolve()
            ),
            "curve_points_csv": str(
                (
                    args.output_dir
                    / "02_piecewise_curves"
                    / method
                    / "curve_points.csv"
                ).resolve()
            ),
            "figure": str(figure_path.resolve()),
        }

    selected = selection["selected"]
    result = {
        "status": "complete",
        "dataset": selection.get("dataset"),
        "selected_frames": [data.start_frame, data.end_frame],
        "reference_frame": data.reference_frame,
        "path_length_m": selected.get("path_length_m"),
        "trajectory_net_heading_change_deg": selected.get(
            "trajectory_net_heading_change_deg"
        ),
        "windows_completed": len(windows),
        "window_size": block_size,
        "window_stride": block_stride,
        "overlap_frames": block_size - block_stride,
        "valid_unique_frames": len(data.frames),
        "road_segment_count": len(chains),
        "selected_polynomial_degree": best_degree,
        "selected_bspline_smoothing_m2_per_point": selected_smoothing,
        "segmentation_gate": {
            "scope": (
                "heuristic adjacent-window continuity gate; not an accuracy metric"
            ),
            "maximum_overlap_p95_gap_m": args.maximum_overlap_p95_gap_m,
            "maximum_overlap_p95_tangent_gap_deg": (
                args.maximum_overlap_p95_tangent_gap_deg
            ),
        },
        "outputs": outputs,
        "warnings": [
            "Road segments are split by registered geometric continuity gates, not semantic lane ground truth.",
            "Metrics and model selection use CLRNet/IPM-derived points, not official KITTI lane ground truth.",
            "The project IPM assumptions remain camera height 1.65 m, pitch 0 degrees and flat road.",
        ],
        "previous_outputs_modified": False,
    }
    common.write_json(args.output_dir / "RESULT.json", result)
    common.write_json(
        args.output_dir / "STATUS.json",
        {
            "status": "complete",
            "selected_frames": result["selected_frames"],
            "windows_completed": len(windows),
            "road_segment_count": len(chains),
            "polynomial_figure": outputs["polynomial"]["figure"],
            "bspline_figure": outputs["bspline"]["figure"],
            "previous_outputs_modified": False,
        },
    )
    print("LONG CURVED ROAD EXPERIMENT FINISHED", flush=True)
    print(f"Frames: {data.start_frame}-{data.end_frame}", flush=True)
    print(f"Windows: {len(windows)}; road segments: {len(chains)}", flush=True)
    print(f"Output: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
