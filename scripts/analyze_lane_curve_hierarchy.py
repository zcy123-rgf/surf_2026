"""Audit direct and segment-wise lane-curve fusion without touching old outputs.

The input is the pose-aligned metric point JSON produced by
``run_full_point_pipeline.py``.  This script answers three separate questions
with separate metrics:

* can all frames be represented by one left and one right smooth curve;
* how do simple x(z) polynomials compare with the existing parametric spline;
* how much information is retained when short-segment curves are represented
  by a small, equal number of arc-length feature points and fused again.

Fidelity to the direct fit is not called accuracy.  Accuracy-like metrics are
only emitted when an explicitly supplied manual pseudo-reference is available.
Every run requires a new or empty output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import fit_first5_two_curves as lane_curve  # noqa: E402


SIDES = lane_curve.SIDES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned-json", type=Path, required=True)
    parser.add_argument("--reference-id", type=int, required=True)
    parser.add_argument("--segment-size", type=int, default=5)
    parser.add_argument("--feature-points-per-segment", type=int, default=8)
    parser.add_argument("--bin-size-m", type=float, default=0.5)
    parser.add_argument(
        "--smoothing-grid", default="0.0025,0.01,0.04,0.16"
    )
    parser.add_argument("--huber-delta-m", type=float, default=0.2)
    parser.add_argument("--irls-iterations", type=int, default=4)
    parser.add_argument("--curve-samples", type=int, default=500)
    parser.add_argument("--manual-json", type=Path)
    parser.add_argument("--calib", type=Path)
    parser.add_argument("--poses", type=Path)
    parser.add_argument("--camera-height", type=float, default=1.65)
    parser.add_argument("--pitch-deg", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_smoothing(
    frames: list[dict[str, object]],
    grid: list[float],
    args: argparse.Namespace,
) -> tuple[float, list[dict[str, object]], dict[str, object]]:
    trials = [
        lane_curve.leave_one_frame_out(
            frames,
            smoothing,
            args.bin_size_m,
            args.huber_delta_m,
            args.irls_iterations,
            args.curve_samples,
        )
        for smoothing in grid
    ]
    minimum = min(trials, key=lambda row: float(row["lofo_fold_rmse_mean_m"]))
    limit = float(minimum["lofo_fold_rmse_mean_m"]) + float(
        minimum["lofo_fold_rmse_standard_error_m"]
    )
    eligible = [
        row for row in trials if float(row["lofo_fold_rmse_mean_m"]) <= limit
    ]
    selected = max(eligible, key=lambda row: float(row["smoothing_per_point_m2"]))
    return float(selected["smoothing_per_point_m2"]), trials, {
        "minimum_mean_fold_rmse_m": float(minimum["lofo_fold_rmse_mean_m"]),
        "one_standard_error_limit_m": limit,
        "selected": {
            key: value for key, value in selected.items() if key != "folds"
        },
    }


def fit_sides(
    frames: list[dict[str, object]],
    smoothing: float,
    args: argparse.Namespace,
) -> dict[str, lane_curve.CurveFit]:
    return {
        side: lane_curve.fit_robust_spline(
            side,
            lane_curve.side_frame_points(frames, side),
            args.bin_size_m,
            smoothing,
            args.huber_delta_m,
            args.irls_iterations,
            args.curve_samples,
        )
        for side in SIDES
    }


def sample_by_arc_length(points: np.ndarray, count: int) -> np.ndarray:
    cumulative = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    )
    if cumulative[-1] <= 0:
        raise ValueError("Cannot sample a zero-length curve.")
    target = np.linspace(0.0, cumulative[-1], count)
    return np.column_stack(
        [np.interp(target, cumulative, points[:, axis]) for axis in range(2)]
    )


def split_segments(
    frames: list[dict[str, object]], segment_size: int
) -> list[list[dict[str, object]]]:
    if segment_size < 5:
        raise ValueError("--segment-size must be at least 5.")
    if len(frames) % segment_size:
        raise ValueError(
            f"{len(frames)} frames are not divisible by segment size {segment_size}."
        )
    return [
        frames[start : start + segment_size]
        for start in range(0, len(frames), segment_size)
    ]


def fit_hierarchy(
    frames: list[dict[str, object]],
    smoothing: float,
    args: argparse.Namespace,
) -> tuple[
    dict[str, lane_curve.CurveFit],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    segments = split_segments(frames, args.segment_size)
    feature_frames: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    for segment_index, segment in enumerate(segments):
        segment_fits = fit_sides(segment, smoothing, args)
        sampled_lanes = []
        for side in SIDES:
            fit = segment_fits[side]
            features = sample_by_arc_length(
                fit.curve_points, args.feature_points_per_segment
            )
            sampled_lanes.append(features)
            segment_rows.append(
                {
                    "segment_index": segment_index,
                    "side": side,
                    "start_frame": int(segment[0]["frame_id"]),
                    "end_frame": int(segment[-1]["frame_id"]),
                    "raw_input_points": int(
                        sum(
                            len(frame["lanes"][SIDES.index(side)])
                            for frame in segment
                        )
                    ),
                    "aggregate_bins": int(len(fit.aggregate_points)),
                    "spline_control_coefficients": int(len(fit.coefficients[0])),
                    "exported_feature_points": int(len(features)),
                }
            )
            for feature_index, point in enumerate(features):
                feature_rows.append(
                    {
                        "segment_index": segment_index,
                        "start_frame": int(segment[0]["frame_id"]),
                        "end_frame": int(segment[-1]["frame_id"]),
                        "side": side,
                        "feature_index": feature_index,
                        "x_right_m": float(point[0]),
                        "z_forward_m": float(point[1]),
                    }
                )
        feature_frames.append(
            {"frame_id": segment_index, "lanes": sampled_lanes}
        )
    hierarchical = fit_sides(feature_frames, smoothing, args)
    return hierarchical, segment_rows, feature_rows


def curve_fidelity(
    direct: dict[str, lane_curve.CurveFit],
    hierarchical: dict[str, lane_curve.CurveFit],
) -> list[dict[str, object]]:
    rows = []
    for side in SIDES:
        first = direct[side].curve_points
        second = hierarchical[side].curve_points
        full_first_to_second = cKDTree(second).query(first, k=1)[0]
        full_second_to_first = cKDTree(first).query(second, k=1)[0]
        common_z_min = max(float(np.min(first[:, 1])), float(np.min(second[:, 1])))
        common_z_max = min(float(np.max(first[:, 1])), float(np.max(second[:, 1])))
        first_common = first[
            (first[:, 1] >= common_z_min) & (first[:, 1] <= common_z_max)
        ]
        second_common = second[
            (second[:, 1] >= common_z_min) & (second[:, 1] <= common_z_max)
        ]
        first_to_second = cKDTree(second_common).query(first_common, k=1)[0]
        second_to_first = cKDTree(first_common).query(second_common, k=1)[0]
        rows.append(
            {
                "side": side,
                "warning": "fidelity to direct fit; not ground-truth accuracy",
                "common_z_min_m": common_z_min,
                "common_z_max_m": common_z_max,
                "symmetric_chamfer_mean_m": float(
                    (np.mean(first_to_second) + np.mean(second_to_first)) / 2.0
                ),
                "symmetric_p95_m": float(
                    max(
                        np.percentile(first_to_second, 95),
                        np.percentile(second_to_first, 95),
                    )
                ),
                "symmetric_hausdorff_max_m": float(
                    max(np.max(first_to_second), np.max(second_to_first))
                ),
                "full_support_symmetric_hausdorff_max_m": float(
                    max(
                        np.max(full_first_to_second),
                        np.max(full_second_to_first),
                    )
                ),
                "full_support_warning": (
                    "includes endpoint support-range differences as well as shape error"
                ),
            }
        )
    return rows


def robust_polynomial(
    points: np.ndarray, degree: int, delta_m: float, iterations: int
) -> np.ndarray:
    z = points[:, 1]
    x = points[:, 0]
    design = np.vander(z, degree + 1)
    weights = np.ones(len(points), dtype=np.float64)
    coefficients = np.zeros(degree + 1, dtype=np.float64)
    for _ in range(max(iterations, 1)):
        root_weight = np.sqrt(weights)
        coefficients = np.linalg.lstsq(
            design * root_weight[:, None], x * root_weight, rcond=None
        )[0]
        residual = np.abs(x - design @ coefficients)
        weights = np.minimum(1.0, delta_m / np.maximum(residual, 1e-9))
        weights = np.maximum(weights, 0.05)
    return coefficients


def polynomial_lofo(
    frames: list[dict[str, object]], degree: int, args: argparse.Namespace
) -> dict[str, object]:
    distances = []
    fold_rows = []
    for held_frame in frames:
        held_id = int(held_frame["frame_id"])
        training = [
            frame for frame in frames if int(frame["frame_id"]) != held_id
        ]
        for side_index, side in enumerate(SIDES):
            aggregate, _ = lane_curve.aggregate_equal_frame_bins(
                lane_curve.side_frame_points(training, side), args.bin_size_m
            )
            coefficients = robust_polynomial(
                aggregate, degree, args.huber_delta_m, args.irls_iterations
            )
            z_min, z_max = np.min(aggregate[:, 1]), np.max(aggregate[:, 1])
            held = np.asarray(held_frame["lanes"][side_index])
            held = held[(held[:, 1] >= z_min) & (held[:, 1] <= z_max)]
            z_samples = np.linspace(z_min, z_max, args.curve_samples)
            predicted = np.column_stack(
                [np.polyval(coefficients, z_samples), z_samples]
            )
            current = cKDTree(predicted).query(held, k=1)[0]
            distances.append(current)
            fold_rows.append(
                {
                    "model": f"polynomial_degree_{degree}",
                    "held_out_frame": held_id,
                    "side": side,
                    "count": int(len(current)),
                    "rmse_m": float(np.sqrt(np.mean(current**2))),
                    "p95_m": float(np.percentile(current, 95)),
                }
            )
    combined = np.concatenate(distances)
    fold_rmse = np.asarray([row["rmse_m"] for row in fold_rows])
    return {
        "model": f"robust polynomial x(z), degree {degree}",
        "count": int(len(combined)),
        "mean_m": float(np.mean(combined)),
        "rmse_m": float(np.sqrt(np.mean(combined**2))),
        "p95_m": float(np.percentile(combined, 95)),
        "mean_fold_rmse_m": float(np.mean(fold_rmse)),
        "fold_rmse_standard_error_m": float(
            np.std(fold_rmse, ddof=1) / np.sqrt(len(fold_rmse))
        ),
        "requires_single_valued_x_of_z": True,
        "folds": fold_rows,
    }


def pose_geometry(record: dict[str, object]) -> dict[str, object]:
    origins = np.asarray(
        [frame["camera_origin_reference_xz_m"] for frame in record["frames"]],
        dtype=np.float64,
    )
    steps = np.diff(origins, axis=0)
    path_length = float(np.sum(np.linalg.norm(steps, axis=1)))
    displacement = float(np.linalg.norm(origins[-1] - origins[0]))
    if len(steps) >= 2:
        headings = np.unwrap(np.arctan2(steps[:, 0], steps[:, 1]))
        heading_change = float(np.degrees(headings[-1] - headings[0]))
        absolute_heading_change = float(np.degrees(np.sum(np.abs(np.diff(headings)))))
    else:
        heading_change = absolute_heading_change = 0.0
    chord = origins[-1] - origins[0]
    if displacement > 0:
        deviation = np.abs(
            (origins[:, 0] - origins[0, 0]) * chord[1]
            - (origins[:, 1] - origins[0, 1]) * chord[0]
        ) / displacement
        maximum_deviation = float(np.max(deviation))
    else:
        maximum_deviation = 0.0
    return {
        "path_length_m": path_length,
        "displacement_m": displacement,
        "signed_heading_change_deg_from_sampled_origins": heading_change,
        "total_absolute_heading_change_deg_from_sampled_origins": absolute_heading_change,
        "maximum_deviation_from_endpoint_chord_m": maximum_deviation,
    }


def manual_metrics(
    fits: dict[str, lane_curve.CurveFit],
    manual_frames: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    thresholds = (0.2, 0.3, 0.5)
    for side_index, side in enumerate(SIDES):
        manual = np.vstack(
            [np.asarray(frame["lanes"][side_index]) for frame in manual_frames]
        )
        predicted = fits[side].curve_points
        z_min = max(np.min(manual[:, 1]), np.min(predicted[:, 1]))
        z_max = min(np.max(manual[:, 1]), np.max(predicted[:, 1]))
        manual = manual[(manual[:, 1] >= z_min) & (manual[:, 1] <= z_max)]
        predicted = predicted[
            (predicted[:, 1] >= z_min) & (predicted[:, 1] <= z_max)
        ]
        pred_to_manual = cKDTree(manual).query(predicted, k=1)[0]
        manual_to_pred = cKDTree(predicted).query(manual, k=1)[0]
        row: dict[str, object] = {
            "side": side,
            "reference_type": "manual pseudo-ground-truth; not official KITTI GT",
            "common_z_min_m": float(z_min),
            "common_z_max_m": float(z_max),
            "symmetric_chamfer_mean_m": float(
                (np.mean(pred_to_manual) + np.mean(manual_to_pred)) / 2.0
            ),
            "symmetric_p95_m": float(
                max(
                    np.percentile(pred_to_manual, 95),
                    np.percentile(manual_to_pred, 95),
                )
            ),
        }
        for threshold in thresholds:
            precision = float(np.mean(pred_to_manual <= threshold))
            recall = float(np.mean(manual_to_pred <= threshold))
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0.0
            )
            suffix = str(threshold).replace(".", "p") + "m"
            row[f"precision_at_{suffix}"] = precision
            row[f"recall_at_{suffix}"] = recall
            row[f"f1_at_{suffix}"] = f1
        rows.append(row)
    return rows


def plot_hierarchy(
    path: Path,
    direct: dict[str, lane_curve.CurveFit],
    hierarchical: dict[str, lane_curve.CurveFit],
    feature_rows: list[dict[str, object]],
) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 10.0))
    for side, color in (("left", "#1f77b4"), ("right", "#d62728")):
        first = direct[side].curve_points
        second = hierarchical[side].curve_points
        features = np.asarray(
            [
                [row["x_right_m"], row["z_forward_m"]]
                for row in feature_rows
                if row["side"] == side
            ],
            dtype=np.float64,
        )
        axis.plot(
            first[:, 0], first[:, 1], color=color, linewidth=3, label=f"{side} direct"
        )
        axis.plot(
            second[:, 0],
            second[:, 1],
            color=color,
            linewidth=2,
            linestyle="--",
            label=f"{side} segment-feature refusion",
        )
        axis.scatter(features[:, 0], features[:, 1], s=16, color=color, alpha=0.55)
    axis.set_xlabel("X right in reference frame [m]")
    axis.set_ylabel("Z forward in reference frame [m]")
    axis.set_title("Direct fit vs short-segment feature-point refusion")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, linewidth=0.5, alpha=0.35)
    axis.legend(fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    lane_curve.require_empty_output(args.output_dir)
    if args.segment_size < 5 or args.feature_points_per_segment < 4:
        raise ValueError("Segment size must be >=5 and feature-point count >=4.")
    if args.manual_json and (not args.calib or not args.poses):
        raise ValueError("Manual evaluation requires --calib and --poses.")
    required = [args.aligned_json]
    required += [path for path in (args.manual_json, args.calib, args.poses) if path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))

    raw_record = json.loads(args.aligned_json.read_text(encoding="utf-8"))
    frames = lane_curve.load_aligned_json(args.aligned_json)
    frame_ids = [int(frame["frame_id"]) for frame in frames]
    lane_curve.validate_frames(frames, frame_ids)
    if frame_ids != list(range(frame_ids[0], frame_ids[-1] + 1)):
        raise ValueError("Aligned input must contain consecutive frames.")

    grid = [float(value) for value in args.smoothing_grid.split(",")]
    selected_smoothing, spline_trials, selection = select_smoothing(
        frames, grid, args
    )
    direct = fit_sides(frames, selected_smoothing, args)
    hierarchical, segment_rows, feature_rows = fit_hierarchy(
        frames, selected_smoothing, args
    )
    fidelity = curve_fidelity(direct, hierarchical)

    polynomial_results = [polynomial_lofo(frames, degree, args) for degree in (1, 2, 3)]
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
                "not mathematically, but current z-bin ordering still assumes monotonic road progress"
            ),
        }
    )

    raw_points = int(
        sum(len(lane) for frame in frames for lane in frame["lanes"])
    )
    feature_points = len(feature_rows)
    compression = {
        "raw_aligned_points": raw_points,
        "exported_segment_feature_points": feature_points,
        "point_count_compression_ratio_raw_over_feature": (
            raw_points / feature_points
        ),
        "feature_point_fraction": feature_points / raw_points,
        "note": (
            "This is geometric sample-count compression, not file-size compression. "
            "Each segment and side contributes the same number of feature points."
        ),
    }

    write_csv(args.output_dir / "01_model_comparison" / "model_lofo.csv", model_rows)
    write_csv(
        args.output_dir / "01_model_comparison" / "polynomial_folds.csv",
        [row for result in polynomial_results for row in result["folds"]],
    )
    write_csv(args.output_dir / "02_short_segments" / "segment_models.csv", segment_rows)
    write_csv(args.output_dir / "02_short_segments" / "feature_points_xz.csv", feature_rows)
    write_csv(
        args.output_dir / "03_hierarchical_refusion" / "fidelity_to_direct.csv",
        fidelity,
    )
    lane_curve.export_fit(
        args.output_dir / "03_hierarchical_refusion" / "direct_curve_data",
        direct,
        frames,
    )
    lane_curve.export_fit(
        args.output_dir / "03_hierarchical_refusion" / "refused_curve_data",
        hierarchical,
        [
            {
                "frame_id": index,
                "lanes": [
                    np.asarray(
                        [
                            [row["x_right_m"], row["z_forward_m"]]
                            for row in feature_rows
                            if row["segment_index"] == index and row["side"] == side
                        ]
                    )
                    for side in SIDES
                ],
            }
            for index in range(len(split_segments(frames, args.segment_size)))
        ],
    )
    plot_hierarchy(
        args.output_dir / "03_hierarchical_refusion" / "direct_vs_refused.png",
        direct,
        hierarchical,
        feature_rows,
    )

    manual_direct = manual_hierarchical = None
    if args.manual_json:
        assert args.calib and args.poses
        calibration = lane_curve.parse_projection(args.calib)
        manual_frames = lane_curve.load_manual_frames(
            args.manual_json,
            calibration,
            args.poses,
            args.reference_id,
            args.camera_height,
            args.pitch_deg,
        )
        manual_direct = manual_metrics(direct, manual_frames)
        manual_hierarchical = manual_metrics(hierarchical, manual_frames)
        write_csv(
            args.output_dir / "04_manual_pseudo_gt" / "direct_curve_metrics.csv",
            manual_direct,
        )
        write_csv(
            args.output_dir / "04_manual_pseudo_gt" / "refused_curve_metrics.csv",
            manual_hierarchical,
        )

    geometry = pose_geometry(raw_record)
    audit = {
        "status": "complete",
        "scope": f"frames {frame_ids[0]:06d}-{frame_ids[-1]:06d}",
        "reference_frame": args.reference_id,
        "input": {
            "aligned_json": str(args.aligned_json.resolve()),
            "aligned_json_sha256": sha256(args.aligned_json),
        },
        "pose_geometry": geometry,
        "direct_two_curve_fit": {
            "model": "robust cubic parametric B-spline",
            "selected_smoothing_per_point_m2": selected_smoothing,
            "selection": selection,
            "fit_metrics": [
                lane_curve.fit_metrics(
                    direct[side], lane_curve.side_frame_points(frames, side)
                )
                for side in SIDES
            ],
            "left_right_check": lane_curve.no_crossing_check(direct),
        },
        "model_comparison": model_rows,
        "hierarchical_segment_refusion": {
            "segment_size_frames": args.segment_size,
            "segment_count": len(split_segments(frames, args.segment_size)),
            "feature_points_per_segment_per_side": args.feature_points_per_segment,
            "compression": compression,
            "fidelity_to_direct_fit": fidelity,
            "warning": "fidelity to the direct fit is not accuracy",
        },
        "manual_pseudo_ground_truth": {
            "direct": manual_direct,
            "hierarchical": manual_hierarchical,
            "warning": "manual labels are not official KITTI ground truth",
        },
        "interpretation_rules": [
            "Low fit or LOFO error only measures consistency with CLRNet-derived points.",
            "Manual pseudo-GT metrics, when present, replace raster overlap for geometric evaluation.",
            "Polynomial x(z) is invalid when a lane is not single-valued in reference-frame z.",
            "The current spline implementation also orders aggregate bins by z; a high-curvature result needs image topology inspection and may require path/Frenet ordering.",
            "Wrong upstream lane identity cannot be repaired by a smooth curve fitter.",
        ],
    }
    audit_dir = args.output_dir / "00_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "STATUS.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "frames": frame_ids,
                "selected_smoothing_per_point_m2": selected_smoothing,
                "raw_points": raw_points,
                "feature_points": feature_points,
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Output: {args.output_dir.resolve()}")
    print(f"Frames: {frame_ids[0]:06d}-{frame_ids[-1]:06d}")
    print(f"Selected smoothing: {selected_smoothing:g}")
    print(f"Raw points: {raw_points}; segment feature points: {feature_points}")
    for row in fidelity:
        print(
            f"{row['side']} hierarchical fidelity: Chamfer="
            f"{row['symmetric_chamfer_mean_m']:.4f} m, "
            f"P95={row['symmetric_p95_m']:.4f} m"
        )


if __name__ == "__main__":
    main()
