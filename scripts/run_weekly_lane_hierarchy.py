"""Run the ten-window lane-curve experiment requested for the weekly meeting.

The input is a completed ``select_hierarchy_150_frames.py`` selection.  This
script does not rerun CLRNet and never reads a previous result as ground truth.
It reuses the saved per-frame CLRNet/IPM points, aligns them with KITTI poses,
fits left and right lanes separately in trajectory Frenet coordinates, reduces
each local curve to a small set of anchors, and fuses the ten local curves.

Accuracy is reported only when an explicit manual pseudo-reference is given.
Without it, curve-fit and refusion errors are labelled internal consistency or
compression fidelity, not accuracy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
from scipy.interpolate import PchipInterpolator, splev, splrep
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import fit_first5_two_curves as lane_io  # noqa: E402
from surf_bev.geometry import (  # noqa: E402
    image_to_ground_ipm,
    load_kitti_poses,
    transform_lane_points_by_pose,
)


SIDES = ("left", "right")


@dataclass
class FrenetPath:
    points_xz: np.ndarray
    cumulative_s: np.ndarray


@dataclass
class ScalarSpline:
    side: str
    smoothing_per_point_m2: float
    tck: tuple
    s_min: float
    s_max: float
    aggregate_sd: np.ndarray
    residuals_m: np.ndarray

    def evaluate(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(splev(values, self.tck), dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--manual-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-height", type=float, default=1.65)
    parser.add_argument("--pitch-deg", type=float, default=0.0)
    parser.add_argument("--bin-size-m", type=float, default=0.50)
    parser.add_argument("--huber-delta-m", type=float, default=0.20)
    parser.add_argument("--irls-iterations", type=int, default=4)
    parser.add_argument("--curve-samples", type=int, default=600)
    parser.add_argument("--maximum-cv-folds", type=int, default=20)
    parser.add_argument("--smoothing-grid", default="0.0025,0.01,0.04,0.16")
    parser.add_argument("--feature-count-grid", default="4,6,8,12")
    parser.add_argument("--feature-p95-limit-m", type=float, default=0.10)
    parser.add_argument("--feature-max-limit-m", type=float, default=0.20)
    return parser.parse_args()


def require_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory must be new or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_reference_path(
    poses_image: list[np.ndarray], frame_ids: list[int], reference_id: int
) -> FrenetPath:
    reference_from_world = np.linalg.inv(poses_image[reference_id])
    points = []
    for frame_id in frame_ids:
        origin_world = poses_image[frame_id] @ np.array([0.0, 0.0, 0.0, 1.0])
        origin_reference = reference_from_world @ origin_world
        points.append(origin_reference[[0, 2]])
    points_xz = np.asarray(points, dtype=np.float64)
    keep = np.concatenate(
        [[True], np.linalg.norm(np.diff(points_xz, axis=0), axis=1) > 1e-5]
    )
    points_xz = points_xz[keep]
    if len(points_xz) < 2:
        raise ValueError("Selected poses do not form a usable reference path.")
    cumulative = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(points_xz, axis=0), axis=1))]
    )
    return FrenetPath(points_xz=points_xz, cumulative_s=cumulative)


def xz_to_frenet(points_xz: np.ndarray, path: FrenetPath) -> np.ndarray:
    """Project reference-frame X/Z points to path progress s and right offset d."""

    points = np.asarray(points_xz, dtype=np.float64).reshape(-1, 2)
    if not len(points):
        return np.empty((0, 2), dtype=np.float64)
    best_distance = np.full(len(points), np.inf, dtype=np.float64)
    best_s = np.zeros(len(points), dtype=np.float64)
    best_d = np.zeros(len(points), dtype=np.float64)
    segment_count = len(path.points_xz) - 1
    for index in range(segment_count):
        start = path.points_xz[index]
        vector = path.points_xz[index + 1] - start
        length = float(np.linalg.norm(vector))
        tangent = vector / length
        raw_t = ((points - start) @ vector) / (length * length)
        if index == 0:
            t = np.minimum(raw_t, 1.0)
        elif index == segment_count - 1:
            t = np.maximum(raw_t, 0.0)
        else:
            t = np.clip(raw_t, 0.0, 1.0)
        projection = start + t[:, None] * vector
        difference = points - projection
        distance = np.linalg.norm(difference, axis=1)
        update = distance < best_distance
        right_normal = np.array([tangent[1], -tangent[0]])
        best_distance[update] = distance[update]
        best_s[update] = path.cumulative_s[index] + t[update] * length
        best_d[update] = difference[update] @ right_normal
    return np.column_stack([best_s, best_d])


def frenet_to_xz(points_sd: np.ndarray, path: FrenetPath) -> np.ndarray:
    points = np.asarray(points_sd, dtype=np.float64).reshape(-1, 2)
    output = np.empty_like(points)
    last = len(path.cumulative_s) - 2
    for row_index, (progress, offset) in enumerate(points):
        if progress <= path.cumulative_s[0]:
            segment = 0
        elif progress >= path.cumulative_s[-1]:
            segment = last
        else:
            segment = int(np.searchsorted(path.cumulative_s, progress) - 1)
        vector = path.points_xz[segment + 1] - path.points_xz[segment]
        length = float(np.linalg.norm(vector))
        tangent = vector / length
        t = (progress - path.cumulative_s[segment]) / length
        center = path.points_xz[segment] + t * vector
        right_normal = np.array([tangent[1], -tangent[0]])
        output[row_index] = center + offset * right_normal
    return output


def load_aligned_frames(
    selection: dict[str, object],
    poses_image: list[np.ndarray],
    reference_id: int,
    path: FrenetPath,
    camera_height: float,
    pitch_deg: float,
) -> tuple[list[dict[str, object]], dict[int, dict[str, object]], Path]:
    selected = selection["selected"]
    points_path = Path(str(selected["selected_lane_points_json"]))
    if not points_path.is_file():
        raise FileNotFoundError(f"Saved CLRNet/IPM points are missing: {points_path}")
    document = load_json(points_path)
    start = int(selected["start_frame"])
    end = int(selected["end_frame"])
    by_id = {int(row["frame_id"]): row for row in document["frames"]}
    frames = []
    audit: dict[int, dict[str, object]] = {}
    reference_pose = poses_image[reference_id]
    for frame_id in range(start, end + 1):
        source = by_id.get(frame_id)
        reason = None
        lanes_sd: list[np.ndarray] = []
        lanes_xz: list[np.ndarray] = []
        if source is None:
            reason = "missing_saved_detection"
        elif not bool(source.get("eligible_for_two_curve_fit", False)):
            reason = "failed_saved_two_lane_metric_gate"
        else:
            local_lanes = source.get("selected_lanes_local_ground_xz_m", [])
            if len(local_lanes) != 2:
                reason = "not_exactly_two_selected_lanes"
            else:
                for lane in local_lanes:
                    aligned = transform_lane_points_by_pose(
                        np.asarray(lane, dtype=np.float64),
                        poses_image[frame_id],
                        reference_pose,
                        camera_height=camera_height,
                        pitch_deg=pitch_deg,
                    )
                    sd = xz_to_frenet(aligned, path)
                    valid = (
                        np.isfinite(sd).all(axis=1)
                        & (np.abs(sd[:, 1]) <= 15.0)
                        & (sd[:, 0] >= path.cumulative_s[0] - 25.0)
                        & (sd[:, 0] <= path.cumulative_s[-1] + 55.0)
                    )
                    lanes_xz.append(aligned[valid])
                    lanes_sd.append(sd[valid])
                if min(map(len, lanes_sd), default=0) < 4:
                    reason = "too_few_points_after_pose_frenet_filter"
        valid_frame = reason is None
        audit[frame_id] = {
            "frame_id": frame_id,
            "valid": valid_frame,
            "reason": reason,
            "left_points": len(lanes_sd[0]) if len(lanes_sd) == 2 else 0,
            "right_points": len(lanes_sd[1]) if len(lanes_sd) == 2 else 0,
        }
        if valid_frame:
            frames.append(
                {
                    "frame_id": frame_id,
                    "lanes_sd": lanes_sd,
                    "lanes_xz": lanes_xz,
                }
            )
    return frames, audit, points_path


def grouped_side_points(
    groups: list[dict[str, object]], side: str
) -> list[tuple[int, np.ndarray]]:
    side_index = SIDES.index(side)
    return [
        (int(group["group_id"]), np.asarray(group["lanes_sd"][side_index]))
        for group in groups
    ]


def aggregate_equal_group_bins(
    grouped_points: list[tuple[int, np.ndarray]], bin_size_m: float
) -> np.ndarray:
    nonempty = [points for _, points in grouped_points if len(points)]
    if not nonempty:
        raise ValueError("No points are available for aggregation.")
    s_min = min(float(np.min(points[:, 0])) for points in nonempty)
    s_max = max(float(np.max(points[:, 0])) for points in nonempty)
    lower = math.floor(s_min / bin_size_m) * bin_size_m
    upper = math.ceil(s_max / bin_size_m) * bin_size_m
    edges = np.arange(lower, upper + 1.5 * bin_size_m, bin_size_m)
    rows = []
    for start, end in zip(edges[:-1], edges[1:]):
        votes = []
        for _, points in grouped_points:
            selected = points[(points[:, 0] >= start) & (points[:, 0] < end)]
            if len(selected):
                votes.append(np.median(selected, axis=0))
        if votes:
            vote_array = np.asarray(votes, dtype=np.float64)
            rows.append(np.median(vote_array, axis=0))
    aggregate = np.asarray(rows, dtype=np.float64)
    if len(aggregate) < 6:
        raise ValueError(f"Only {len(aggregate)} aggregate Frenet bins are available.")
    order = np.argsort(aggregate[:, 0])
    return aggregate[order]


def fit_scalar_spline(
    side: str,
    grouped_points: list[tuple[int, np.ndarray]],
    bin_size_m: float,
    smoothing_per_point_m2: float,
    huber_delta_m: float,
    irls_iterations: int,
) -> ScalarSpline:
    aggregate = aggregate_equal_group_bins(grouped_points, bin_size_m)
    progress, offsets = aggregate[:, 0], aggregate[:, 1]
    weights = np.ones(len(aggregate), dtype=np.float64)
    tck = None
    residuals = np.zeros(len(aggregate), dtype=np.float64)
    for _ in range(max(1, irls_iterations)):
        smoothing = max(smoothing_per_point_m2 * len(aggregate), 1e-12)
        tck = splrep(progress, offsets, w=weights, k=3, s=smoothing)
        residuals = np.abs(offsets - splev(progress, tck))
        weights = np.minimum(1.0, huber_delta_m / np.maximum(residuals, 1e-9))
        weights = np.maximum(weights, 0.05)
    assert tck is not None
    return ScalarSpline(
        side=side,
        smoothing_per_point_m2=smoothing_per_point_m2,
        tck=tck,
        s_min=float(progress[0]),
        s_max=float(progress[-1]),
        aggregate_sd=aggregate,
        residuals_m=residuals,
    )


def fold_indices(count: int, maximum: int) -> list[int]:
    if count <= maximum:
        return list(range(count))
    return sorted(set(np.linspace(0, count - 1, maximum).round().astype(int).tolist()))


def spline_cross_validation(
    groups: list[dict[str, object]], grid: list[float], args: argparse.Namespace
) -> tuple[float, list[dict[str, object]]]:
    rows = []
    for smoothing in grid:
        fold_errors = []
        for held_index in fold_indices(len(groups), args.maximum_cv_folds):
            training = [group for index, group in enumerate(groups) if index != held_index]
            held = groups[held_index]
            side_errors = []
            for side_index, side in enumerate(SIDES):
                try:
                    fit = fit_scalar_spline(
                        side,
                        grouped_side_points(training, side),
                        args.bin_size_m,
                        smoothing,
                        args.huber_delta_m,
                        args.irls_iterations,
                    )
                except ValueError:
                    continue
                points = np.asarray(held["lanes_sd"][side_index])
                points = points[
                    (points[:, 0] >= fit.s_min) & (points[:, 0] <= fit.s_max)
                ]
                if len(points):
                    side_errors.append(np.abs(points[:, 1] - fit.evaluate(points[:, 0])))
            if side_errors:
                errors = np.concatenate(side_errors)
                fold_errors.append(float(np.sqrt(np.mean(errors**2))))
        if not fold_errors:
            raise ValueError("Spline cross-validation produced no supported folds.")
        values = np.asarray(fold_errors)
        rows.append(
            {
                "model": "robust cubic Frenet B-spline d(s)",
                "smoothing_per_point_m2": smoothing,
                "fold_count": len(values),
                "mean_fold_rmse_m": float(np.mean(values)),
                "fold_rmse_standard_error_m": float(
                    np.std(values, ddof=1) / np.sqrt(len(values))
                    if len(values) > 1
                    else 0.0
                ),
                "p95_fold_rmse_m": float(np.percentile(values, 95)),
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


def robust_polyfit(points_sd: np.ndarray, degree: int, args: argparse.Namespace) -> np.ndarray:
    if len(points_sd) < degree + 2:
        raise ValueError("Too few points for polynomial fitting.")
    progress, offsets = points_sd[:, 0], points_sd[:, 1]
    weights = np.ones(len(points_sd), dtype=np.float64)
    coefficients = np.polyfit(progress, offsets, degree, w=weights)
    for _ in range(max(1, args.irls_iterations)):
        residuals = np.abs(offsets - np.polyval(coefficients, progress))
        weights = np.minimum(1.0, args.huber_delta_m / np.maximum(residuals, 1e-9))
        weights = np.maximum(weights, 0.05)
        coefficients = np.polyfit(progress, offsets, degree, w=weights)
    return coefficients


def polynomial_cross_validation(
    groups: list[dict[str, object]], degree: int, args: argparse.Namespace
) -> dict[str, object]:
    folds = []
    all_errors = []
    for held_index in fold_indices(len(groups), args.maximum_cv_folds):
        training = [group for index, group in enumerate(groups) if index != held_index]
        held = groups[held_index]
        current_fold = []
        for side_index, side in enumerate(SIDES):
            aggregate = aggregate_equal_group_bins(
                grouped_side_points(training, side), args.bin_size_m
            )
            coefficients = robust_polyfit(aggregate, degree, args)
            points = np.asarray(held["lanes_sd"][side_index])
            supported = points[
                (points[:, 0] >= aggregate[0, 0])
                & (points[:, 0] <= aggregate[-1, 0])
            ]
            if len(supported):
                current_fold.append(
                    np.abs(supported[:, 1] - np.polyval(coefficients, supported[:, 0]))
                )
        if current_fold:
            errors = np.concatenate(current_fold)
            all_errors.append(errors)
            folds.append(float(np.sqrt(np.mean(errors**2))))
    combined = np.concatenate(all_errors)
    fold_values = np.asarray(folds)
    return {
        "model": f"robust Frenet polynomial d(s), degree {degree}",
        "fold_count": len(folds),
        "mean_error_m": float(np.mean(combined)),
        "rmse_m": float(np.sqrt(np.mean(combined**2))),
        "p95_m": float(np.percentile(combined, 95)),
        "mean_fold_rmse_m": float(np.mean(fold_values)),
        "fold_rmse_standard_error_m": float(
            np.std(fold_values, ddof=1) / np.sqrt(len(fold_values))
            if len(fold_values) > 1
            else 0.0
        ),
    }


def fit_both_sides(
    groups: list[dict[str, object]], smoothing: float, args: argparse.Namespace
) -> dict[str, ScalarSpline]:
    return {
        side: fit_scalar_spline(
            side,
            grouped_side_points(groups, side),
            args.bin_size_m,
            smoothing,
            args.huber_delta_m,
            args.irls_iterations,
        )
        for side in SIDES
    }


def sample_fit(fit: ScalarSpline, count: int) -> np.ndarray:
    progress = np.linspace(fit.s_min, fit.s_max, count)
    return np.column_stack([progress, fit.evaluate(progress)])


def sparse_reconstruction_trial(
    fit: ScalarSpline, count: int, path: FrenetPath, dense_count: int
) -> tuple[dict[str, object], np.ndarray]:
    anchors = sample_fit(fit, count)
    dense_s = np.linspace(fit.s_min, fit.s_max, dense_count)
    original_sd = np.column_stack([dense_s, fit.evaluate(dense_s)])
    reconstructed_d = PchipInterpolator(anchors[:, 0], anchors[:, 1])(dense_s)
    reconstructed_sd = np.column_stack([dense_s, reconstructed_d])
    original_xz = frenet_to_xz(original_sd, path)
    reconstructed_xz = frenet_to_xz(reconstructed_sd, path)
    errors = np.linalg.norm(original_xz - reconstructed_xz, axis=1)
    return (
        {
            "feature_points_per_window_per_side": count,
            "mean_reconstruction_error_m": float(np.mean(errors)),
            "rmse_reconstruction_error_m": float(np.sqrt(np.mean(errors**2))),
            "p95_reconstruction_error_m": float(np.percentile(errors, 95)),
            "max_reconstruction_error_m": float(np.max(errors)),
        },
        anchors,
    )


def choose_feature_count(
    window_fits: list[dict[str, object]],
    counts: list[int],
    path: FrenetPath,
    args: argparse.Namespace,
) -> tuple[int, list[dict[str, object]]]:
    summary = []
    for count in counts:
        errors = []
        for window in window_fits:
            for side in SIDES:
                trial, _ = sparse_reconstruction_trial(
                    window["fits"][side], count, path, args.curve_samples
                )
                errors.append(trial)
        p95_values = [float(row["p95_reconstruction_error_m"]) for row in errors]
        max_values = [float(row["max_reconstruction_error_m"]) for row in errors]
        summary.append(
            {
                "feature_points_per_window_per_side": count,
                "curve_count": len(errors),
                "mean_curve_p95_m": float(np.mean(p95_values)),
                "worst_curve_p95_m": float(np.max(p95_values)),
                "worst_curve_max_m": float(np.max(max_values)),
                "meets_registered_fidelity_gate": bool(
                    np.max(p95_values) <= args.feature_p95_limit_m
                    and np.max(max_values) <= args.feature_max_limit_m
                ),
            }
        )
    passing = [row for row in summary if row["meets_registered_fidelity_gate"]]
    if passing:
        selected = min(passing, key=lambda row: int(row["feature_points_per_window_per_side"]))
        rule = "fewest points meeting both registered fidelity limits"
    else:
        selected = min(
            summary,
            key=lambda row: (
                float(row["worst_curve_p95_m"]),
                int(row["feature_points_per_window_per_side"]),
            ),
        )
        rule = "no candidate met both limits; selected lowest worst-curve P95"
    for row in summary:
        row["selected"] = row is selected
        row["selection_rule"] = rule
        row["registered_p95_limit_m"] = args.feature_p95_limit_m
        row["registered_max_limit_m"] = args.feature_max_limit_m
    return int(selected["feature_points_per_window_per_side"]), summary


def common_curve_metrics(
    first: ScalarSpline, second: ScalarSpline, path: FrenetPath, samples: int
) -> dict[str, float]:
    lower = max(first.s_min, second.s_min)
    upper = min(first.s_max, second.s_max)
    if upper <= lower:
        raise ValueError("Curves have no common Frenet support.")
    progress = np.linspace(lower, upper, samples)
    first_xz = frenet_to_xz(
        np.column_stack([progress, first.evaluate(progress)]), path
    )
    second_xz = frenet_to_xz(
        np.column_stack([progress, second.evaluate(progress)]), path
    )
    pointwise = np.linalg.norm(first_xz - second_xz, axis=1)
    first_to_second = cKDTree(second_xz).query(first_xz, k=1)[0]
    second_to_first = cKDTree(first_xz).query(second_xz, k=1)[0]
    return {
        "common_s_min_m": lower,
        "common_s_max_m": upper,
        "pointwise_mean_m": float(np.mean(pointwise)),
        "pointwise_p95_m": float(np.percentile(pointwise, 95)),
        "pointwise_max_m": float(np.max(pointwise)),
        "symmetric_chamfer_mean_m": float(
            (np.mean(first_to_second) + np.mean(second_to_first)) / 2.0
        ),
        "symmetric_p95_m": float(
            max(np.percentile(first_to_second, 95), np.percentile(second_to_first, 95))
        ),
    }


def adjacent_window_continuity(
    windows: list[dict[str, object]], path: FrenetPath
) -> list[dict[str, object]]:
    rows = []
    for first, second in zip(windows[:-1], windows[1:]):
        for side in SIDES:
            first_fit = first["fits"][side]
            second_fit = second["fits"][side]
            lower = max(first_fit.s_min, second_fit.s_min)
            upper = min(first_fit.s_max, second_fit.s_max)
            if upper <= lower:
                rows.append(
                    {
                        "first_window": first["window_index"],
                        "second_window": second["window_index"],
                        "side": side,
                        "overlap_available": False,
                    }
                )
                continue
            progress = np.linspace(lower, upper, 100)
            first_d = first_fit.evaluate(progress)
            second_d = second_fit.evaluate(progress)
            position = np.abs(first_d - second_d)
            first_slope = np.asarray(splev(progress, first_fit.tck, der=1))
            second_slope = np.asarray(splev(progress, second_fit.tck, der=1))
            angle = np.abs(np.degrees(np.arctan(first_slope) - np.arctan(second_slope)))
            angle = np.minimum(angle, 180.0 - angle)
            rows.append(
                {
                    "first_window": first["window_index"],
                    "second_window": second["window_index"],
                    "side": side,
                    "overlap_available": True,
                    "overlap_s_m": upper - lower,
                    "mean_position_gap_m": float(np.mean(position)),
                    "p95_position_gap_m": float(np.percentile(position, 95)),
                    "mean_tangent_gap_deg": float(np.mean(angle)),
                    "p95_tangent_gap_deg": float(np.percentile(angle, 95)),
                }
            )
    return rows


def curve_width_check(fits: dict[str, ScalarSpline]) -> dict[str, object]:
    lower = max(fits["left"].s_min, fits["right"].s_min)
    upper = min(fits["left"].s_max, fits["right"].s_max)
    progress = np.linspace(lower, upper, 500)
    width = fits["right"].evaluate(progress) - fits["left"].evaluate(progress)
    return {
        "common_s_min_m": lower,
        "common_s_max_m": upper,
        "minimum_right_minus_left_m": float(np.min(width)),
        "median_right_minus_left_m": float(np.median(width)),
        "maximum_right_minus_left_m": float(np.max(width)),
        "curves_cross": bool(np.any(width <= 0.0)),
    }


def manual_metrics(
    manual_json: Path,
    calibration: dict[str, np.ndarray],
    poses_path: Path,
    reference_id: int,
    camera_height: float,
    pitch_deg: float,
    final_fits: dict[str, ScalarSpline],
    path: FrenetPath,
) -> list[dict[str, object]]:
    # Do not use the old short-range metric_filter here: a 105-frame experiment
    # legitimately contains annotations far behind the final reference camera.
    record = load_json(manual_json)
    poses_cam0 = load_kitti_poses(poses_path)
    poses_image = [pose @ calibration["cam0_from_image"] for pose in poses_cam0]
    reference_pose = poses_image[reference_id]
    frames = []
    for item in record["frames_xy"]:
        frame_id = int(item["frame_id"])
        aligned_lanes = []
        for side in SIDES:
            image_points = np.asarray(item[side], dtype=np.float64).reshape(-1, 2)
            ground = image_to_ground_ipm(
                image_points,
                calibration["K"],
                camera_height=camera_height,
                pitch_deg=pitch_deg,
                z_range=(3.0, 50.0),
            )
            aligned_lanes.append(
                transform_lane_points_by_pose(
                    ground,
                    poses_image[frame_id],
                    reference_pose,
                    camera_height=camera_height,
                    pitch_deg=pitch_deg,
                )
            )
        frames.append({"frame_id": frame_id, "lanes": aligned_lanes})
    rows = []
    for side_index, side in enumerate(SIDES):
        manual_xz = np.vstack([frame["lanes"][side_index] for frame in frames])
        manual_sd = xz_to_frenet(manual_xz, path)
        fit = final_fits[side]
        lower = max(fit.s_min, float(np.min(manual_sd[:, 0])))
        upper = min(fit.s_max, float(np.max(manual_sd[:, 0])))
        manual_xz = manual_xz[
            (manual_sd[:, 0] >= lower) & (manual_sd[:, 0] <= upper)
        ]
        progress = np.linspace(lower, upper, 1000)
        predicted_xz = frenet_to_xz(
            np.column_stack([progress, fit.evaluate(progress)]), path
        )
        predicted_to_manual = cKDTree(manual_xz).query(predicted_xz, k=1)[0]
        manual_to_predicted = cKDTree(predicted_xz).query(manual_xz, k=1)[0]
        row: dict[str, object] = {
            "side": side,
            "reference_type": "manual pseudo-ground-truth; not official KITTI GT",
            "symmetric_chamfer_mean_m": float(
                (np.mean(predicted_to_manual) + np.mean(manual_to_predicted)) / 2.0
            ),
            "symmetric_p95_m": float(
                max(
                    np.percentile(predicted_to_manual, 95),
                    np.percentile(manual_to_predicted, 95),
                )
            ),
        }
        for threshold in (0.2, 0.3, 0.5):
            precision = float(np.mean(predicted_to_manual <= threshold))
            recall = float(np.mean(manual_to_predicted <= threshold))
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            suffix = str(threshold).replace(".", "p") + "m"
            row[f"precision_at_{suffix}"] = precision
            row[f"recall_at_{suffix}"] = recall
            row[f"f1_at_{suffix}"] = f1
        rows.append(row)
    return rows


def export_curves(
    directory: Path,
    fits: dict[str, ScalarSpline],
    path: FrenetPath,
    samples: int,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for side in SIDES:
        points_sd = sample_fit(fits[side], samples)
        points_xz = frenet_to_xz(points_sd, path)
        write_csv(
            directory / f"{side}_curve.csv",
            [
                {
                    "s_path_m": sd[0],
                    "d_right_m": sd[1],
                    "x_right_reference_m": xz[0],
                    "z_forward_reference_m": xz[1],
                }
                for sd, xz in zip(points_sd, points_xz)
            ],
        )


def plot_final(
    path_out: Path,
    direct: dict[str, ScalarSpline],
    final: dict[str, ScalarSpline],
    anchors: list[dict[str, object]],
    path: FrenetPath,
) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 10.0))
    colors = {"left": "#1f77b4", "right": "#d62728"}
    axis.plot(
        path.points_xz[:, 0], path.points_xz[:, 1], color="0.65", linewidth=1.5,
        label="camera path"
    )
    for side in SIDES:
        direct_xz = frenet_to_xz(sample_fit(direct[side], 800), path)
        final_xz = frenet_to_xz(sample_fit(final[side], 800), path)
        selected = np.asarray(
            [[row["x_right_reference_m"], row["z_forward_reference_m"]]
             for row in anchors if row["side"] == side]
        )
        axis.plot(
            direct_xz[:, 0], direct_xz[:, 1], color=colors[side], linewidth=1.4,
            alpha=0.55, label=f"{side} direct raw-point fit"
        )
        axis.plot(
            final_xz[:, 0], final_xz[:, 1], color=colors[side], linewidth=3.0,
            label=f"{side} 10-window sparse refusion"
        )
        axis.scatter(selected[:, 0], selected[:, 1], s=12, color=colors[side], alpha=0.45)
    axis.set_xlabel("X right in final reference frame [m]")
    axis.set_ylabel("Z forward in final reference frame [m]")
    axis.set_title("Ten overlapping 15-frame windows: final left/right lane curves")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, linewidth=0.5, alpha=0.35)
    axis.legend(fontsize=8)
    figure.tight_layout()
    path_out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path_out, dpi=240)
    plt.close(figure)


def plot_models(path_out: Path, rows: list[dict[str, object]]) -> None:
    labels = [str(row["model"]).replace("robust Frenet ", "") for row in rows]
    values = [float(row["mean_fold_rmse_m"]) for row in rows]
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.bar(range(len(rows)), values, color="0.25")
    axis.set_xticks(range(len(rows)), labels, rotation=18, ha="right")
    axis.set_ylabel("held-out frame RMSE [m]")
    axis.set_title("Curve model comparison (lower is better)")
    axis.grid(True, axis="y", linewidth=0.5, alpha=0.35)
    figure.tight_layout()
    path_out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path_out, dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    require_empty_output(args.output_dir)
    required = [args.selection_json, args.poses, args.calib]
    if args.manual_json:
        required.append(args.manual_json)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs: " + ", ".join(missing))
    smoothing_grid = [float(value) for value in args.smoothing_grid.split(",")]
    feature_grid = [int(value) for value in args.feature_count_grid.split(",")]
    if any(value <= 0 for value in smoothing_grid):
        raise ValueError("Smoothing candidates must be positive.")
    if any(value < 4 for value in feature_grid):
        raise ValueError("Feature-point candidates must be at least four.")

    selection = load_json(args.selection_json)
    if selection.get("status") != "selected":
        raise ValueError(
            "The hierarchy selection did not meet the registered coverage gate: "
            f"{selection.get('status')}"
        )
    if int(selection["block_count"]) != 10 or int(selection["block_size"]) != 15:
        raise ValueError("This experiment requires exactly ten 15-frame windows.")

    calibration = lane_io.parse_projection(args.calib)
    poses_cam0 = load_kitti_poses(args.poses)
    poses_image = [pose @ calibration["cam0_from_image"] for pose in poses_cam0]
    selected = selection["selected"]
    start, end = int(selected["start_frame"]), int(selected["end_frame"])
    reference_id = end
    frame_ids = list(range(start, end + 1))
    reference_path = build_reference_path(poses_image, frame_ids, reference_id)
    frames, frame_audit, points_path = load_aligned_frames(
        selection,
        poses_image,
        reference_id,
        reference_path,
        args.camera_height,
        args.pitch_deg,
    )
    by_id = {int(frame["frame_id"]): frame for frame in frames}
    groups = [
        {"group_id": int(frame["frame_id"]), "lanes_sd": frame["lanes_sd"]}
        for frame in frames
    ]
    if len(groups) < 10:
        raise ValueError("Fewer than ten valid frames remain after metric alignment.")

    selected_smoothing, spline_cv_rows = spline_cross_validation(
        groups, smoothing_grid, args
    )
    direct_fits = fit_both_sides(groups, selected_smoothing, args)
    model_rows = [polynomial_cross_validation(groups, degree, args) for degree in (1, 2, 3)]
    selected_spline_row = next(
        row for row in spline_cv_rows if row["selected_by_one_standard_error_rule"]
    )
    model_rows.append(selected_spline_row)
    best_model = min(model_rows, key=lambda row: float(row["mean_fold_rmse_m"]))
    for row in model_rows:
        row["lowest_observed_mean_fold_rmse"] = row is best_model

    windows: list[dict[str, object]] = []
    window_rows = []
    for block in selection["blocks"]:
        window_index = int(block["block_index"])
        window_start = int(block["start_frame"])
        window_end = int(block["end_frame"])
        valid_ids = [frame_id for frame_id in range(window_start, window_end + 1) if frame_id in by_id]
        if len(valid_ids) < int(selection["minimum_valid_frames_per_block_required"]):
            raise ValueError(
                f"Window {window_index} has only {len(valid_ids)} valid frames."
            )
        window_groups = [
            {"group_id": frame_id, "lanes_sd": by_id[frame_id]["lanes_sd"]}
            for frame_id in valid_ids
        ]
        fits = fit_both_sides(window_groups, selected_smoothing, args)
        windows.append(
            {
                "window_index": window_index,
                "start_frame": window_start,
                "end_frame": window_end,
                "valid_ids": valid_ids,
                "fits": fits,
            }
        )
        window_rows.append(
            {
                "window_index": window_index,
                "start_frame": window_start,
                "end_frame": window_end,
                "valid_frame_count": len(valid_ids),
                "valid_frame_ids": ";".join(map(str, valid_ids)),
                "left_s_min_m": fits["left"].s_min,
                "left_s_max_m": fits["left"].s_max,
                "right_s_min_m": fits["right"].s_min,
                "right_s_max_m": fits["right"].s_max,
            }
        )

    selected_feature_count, compression_trials = choose_feature_count(
        windows, feature_grid, reference_path, args
    )
    anchor_rows = []
    anchor_groups = []
    reconstruction_rows = []
    for window in windows:
        sampled_lanes = []
        for side in SIDES:
            trial, anchors = sparse_reconstruction_trial(
                window["fits"][side],
                selected_feature_count,
                reference_path,
                args.curve_samples,
            )
            trial.update({"window_index": window["window_index"], "side": side})
            reconstruction_rows.append(trial)
            sampled_lanes.append(anchors)
            anchors_xz = frenet_to_xz(anchors, reference_path)
            for anchor_index, (sd, xz) in enumerate(zip(anchors, anchors_xz)):
                anchor_rows.append(
                    {
                        "window_index": window["window_index"],
                        "side": side,
                        "anchor_index": anchor_index,
                        "s_path_m": sd[0],
                        "d_right_m": sd[1],
                        "x_right_reference_m": xz[0],
                        "z_forward_reference_m": xz[1],
                    }
                )
        anchor_groups.append(
            {
                "group_id": int(window["window_index"]),
                "lanes_sd": sampled_lanes,
            }
        )

    refusion_smoothing, refusion_cv_rows = spline_cross_validation(
        anchor_groups, smoothing_grid, args
    )
    final_fits = fit_both_sides(anchor_groups, refusion_smoothing, args)
    fidelity_rows = []
    for side in SIDES:
        fidelity_rows.append(
            {
                "side": side,
                "metric_scope": "hierarchical sparse refusion fidelity to direct fit; not accuracy",
                **common_curve_metrics(
                    direct_fits[side], final_fits[side], reference_path, args.curve_samples
                ),
            }
        )
    continuity_rows = adjacent_window_continuity(windows, reference_path)
    width_check = curve_width_check(final_fits)

    manual_rows = None
    manual_status = "pending_manual_annotation"
    if args.manual_json:
        manual_rows = manual_metrics(
            args.manual_json,
            calibration,
            args.poses,
            reference_id,
            args.camera_height,
            args.pitch_deg,
            final_fits,
            reference_path,
        )
        manual_status = "complete_manual_pseudo_ground_truth_evaluation"

    raw_point_count = int(
        sum(len(lane) for frame in frames for lane in frame["lanes_sd"])
    )
    anchor_count = len(anchor_rows)
    compression_ratio = raw_point_count / anchor_count

    write_csv(args.output_dir / "00_audit" / "frame_validity.csv", frame_audit.values())
    write_csv(args.output_dir / "01_q1_two_curves" / "window_summary.csv", window_rows)
    write_csv(args.output_dir / "02_q2_model_comparison" / "model_comparison.csv", model_rows)
    write_csv(args.output_dir / "02_q2_model_comparison" / "spline_smoothing_cv.csv", spline_cv_rows)
    write_csv(args.output_dir / "03_q3_sparse_refusion" / "compression_trials.csv", compression_trials)
    write_csv(args.output_dir / "03_q3_sparse_refusion" / "per_curve_reconstruction.csv", reconstruction_rows)
    write_csv(args.output_dir / "03_q3_sparse_refusion" / "sparse_anchors.csv", anchor_rows)
    write_csv(args.output_dir / "03_q3_sparse_refusion" / "adjacent_window_continuity.csv", continuity_rows)
    write_csv(args.output_dir / "03_q3_sparse_refusion" / "refusion_smoothing_cv.csv", refusion_cv_rows)
    write_csv(args.output_dir / "03_q3_sparse_refusion" / "fidelity_to_direct.csv", fidelity_rows)
    if manual_rows is not None:
        write_csv(args.output_dir / "04_q4_manual_evaluation" / "manual_metrics.csv", manual_rows)
    else:
        pending = {
            "status": manual_status,
            "reason": "Accuracy cannot be measured before manual image annotations exist.",
            "annotation_package": selection.get("package_zip"),
            "planned_metrics": [
                "symmetric Chamfer mean and P95 in metres",
                "precision, recall and F1 at fixed 0.2/0.3/0.5 m tolerances",
            ],
            "warning": "Manual labels are pseudo-ground-truth, not official KITTI truth.",
        }
        pending_path = args.output_dir / "04_q4_manual_evaluation" / "PENDING.json"
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        pending_path.write_text(json.dumps(pending, ensure_ascii=False, indent=2), encoding="utf-8")

    export_curves(
        args.output_dir / "01_q1_two_curves" / "direct_curve_data",
        direct_fits,
        reference_path,
        args.curve_samples,
    )
    export_curves(
        args.output_dir / "01_q1_two_curves" / "final_hierarchical_curve_data",
        final_fits,
        reference_path,
        args.curve_samples,
    )
    plot_final(
        args.output_dir / "01_q1_two_curves" / "final_two_curves.png",
        direct_fits,
        final_fits,
        anchor_rows,
        reference_path,
    )
    plot_models(
        args.output_dir / "02_q2_model_comparison" / "model_comparison.png",
        model_rows,
    )

    result = {
        "status": "complete",
        "dataset": "KITTI Odometry Sequence 00",
        "scope": f"frames {start:06d}-{end:06d}",
        "reference_frame": reference_id,
        "coordinate_system": {
            "curve_fit": "Frenet s/d: s follows the selected camera path, d is positive right",
            "export": "reference image-camera ground X/Z metres; X right, Z forward",
            "pose_formula": "p_ref = inv(T_world_ref) @ T_world_src @ p_src",
        },
        "inputs": {
            "selection_json": str(args.selection_json.resolve()),
            "selection_sha256": sha256(args.selection_json),
            "selected_lane_points_json": str(points_path.resolve()),
            "selected_lane_points_sha256": sha256(points_path),
            "poses": str(args.poses.resolve()),
            "poses_sha256": sha256(args.poses),
            "calib": str(args.calib.resolve()),
            "calib_sha256": sha256(args.calib),
            "manual_json": str(args.manual_json.resolve()) if args.manual_json else None,
        },
        "question_1": {
            "method": "pose alignment; per-frame equal-vote bins; robust cubic Frenet B-spline; left/right never pooled",
            "ten_windows_completed": len(windows),
            "valid_unique_frames": len(frames),
            "selected_smoothing_per_point_m2": selected_smoothing,
            "left_right_width_check": width_check,
            "result_image": str((args.output_dir / "01_q1_two_curves" / "final_two_curves.png").resolve()),
        },
        "question_2": {
            "method": "held-out-frame comparison of degree 1/2/3 Frenet polynomials and robust cubic Frenet B-spline",
            "best_observed_model": best_model,
            "model_rows": model_rows,
            "warning": "Model selection error measures agreement with CLRNet-derived points, not lane accuracy.",
        },
        "question_3": {
            "method": "ten overlapping 15-frame local fits; PCHIP reconstruction from sparse samples; equal-window robust global refit",
            "selected_feature_points_per_window_per_side": selected_feature_count,
            "raw_aligned_points": raw_point_count,
            "sparse_anchor_points": anchor_count,
            "raw_point_to_anchor_count_ratio": compression_ratio,
            "refusion_smoothing_per_point_m2": refusion_smoothing,
            "fidelity_to_direct_fit": fidelity_rows,
            "left_right_width_check": width_check,
            "warning": "Compression fidelity to the direct fit is not accuracy.",
        },
        "question_4": {
            "status": manual_status,
            "manual_metrics": manual_rows,
            "replacement_for_pixel_overlap": "metric curve distance plus precision/recall/F1 at fixed metre tolerances",
            "warning": "Manual annotations are pseudo-ground-truth, not official KITTI ground truth.",
        },
        "registered_settings": {
            "windows": "10 x 15 frames, stride 10, overlap 5 frames",
            "smoothing_grid_m2_per_point": smoothing_grid,
            "feature_count_grid": feature_grid,
            "feature_p95_limit_m": args.feature_p95_limit_m,
            "feature_max_limit_m": args.feature_max_limit_m,
            "manual_tolerances_m": [0.2, 0.3, 0.5],
        },
        "limitations": [
            "Outermost CLRNet candidates are not guaranteed to be ego-lane boundaries.",
            "The IPM inherits the project assumptions camera height 1.65 m, pitch 0 degrees and flat road.",
            "KITTI pose aligns observations but does not correct detection or IPM bias.",
            "No accuracy claim is allowed until the manual pseudo-reference is supplied.",
        ],
    }
    audit_dir = args.output_dir / "00_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "RESULTS.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "STATUS.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "previous_outputs_modified": False,
                "frames": [start, end],
                "valid_unique_frames": len(frames),
                "windows_completed": len(windows),
                "selected_feature_points_per_window_per_side": selected_feature_count,
                "manual_evaluation_status": manual_status,
                "results_json": str((audit_dir / "RESULTS.json").resolve()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    best_name = str(best_model["model"])
    summary_lines = [
        "# 本周四个问题：自动实验结果",
        "",
        f"- 数据：KITTI Odometry Sequence 00，帧 {start:06d}--{end:06d}。",
        f"- 实际有效帧：{len(frames)}；完成 10 个 15 帧窗口（步长 10、重叠 5 帧）。",
        "- 所有坐标、曲线和指标均由本次运行生成；旧输出未修改。",
        "",
        "## 1. 多帧点怎样形成左右两条平滑曲线",
        "",
        "先用 KITTI pose 把各帧米制点对齐，再转到沿车辆轨迹的 Frenet s/d 坐标；左右分开、每帧等权，使用 Huber-IRLS 鲁棒三次 B 样条。",
        f"结果：10 个窗口全部完成；最终两曲线交叉={width_check['curves_cross']}，最小左右间距={width_check['minimum_right_minus_left_m']:.3f} m。",
        "",
        "## 2. 弯道和更多帧时多项式是否合适",
        "",
        "统一在 Frenet d(s) 中比较 1/2/3 次多项式和三次 B 样条，依据留出帧 RMSE，不用训练拟合误差选模型。",
        f"结果：本段观测到的最低平均留出帧 RMSE 模型为 `{best_name}`，RMSE={float(best_model['mean_fold_rmse_m']):.4f} m。该数值是 CLRNet 点的一致性，不是车道真值准确率。",
        "",
        "## 3. 每段少量点怎样继续融合",
        "",
        "每个 15 帧窗口的左右曲线分别试 4/6/8/12 个等 s 锚点，用 PCHIP 从锚点反向重建；在预先登记的 P95<=0.10 m、最大误差<=0.20 m 门槛下选最少点，再按窗口等权做总融合。",
        f"结果：选择每段每侧 {selected_feature_count} 点；原始对齐点/锚点={compression_ratio:.1f} 倍；最终稀疏再融合对直接融合的平均 Chamfer 为 "
        + ", ".join(f"{row['side']} {row['symmetric_chamfer_mean_m']:.4f} m" for row in fidelity_rows)
        + "。这是压缩保真度，不是准确率。",
        "",
        "## 4. 怎样替换像素重叠率",
        "",
        "改用米制双向曲线距离（Chamfer 均值/P95）和固定 0.2/0.3/0.5 m 容差下的 Precision、Recall、F1。",
        (
            "结果：已完成人工伪真值评估，详见 04_q4_manual_evaluation/manual_metrics.csv。"
            if manual_rows is not None
            else "结果：指标代码和 30 帧标注包已准备好，但没有人工标注前必须标记为 pending，不能报告准确率。"
        ),
        "",
        "## 组会必须同时说明的限制",
        "",
        "- 当前左右身份继承自每帧最外侧 CLRNet 候选，仍需看图核验。",
        "- IPM 仍使用项目假设：相机高 1.65 m、俯仰 0°、平坦路面。",
        "- pose 只负责坐标对齐，不能消除检测误差或 IPM 系统偏差。",
        "- 人工标注只能称人工参考/伪真值，不能称 KITTI 官方真值。",
    ]
    (args.output_dir / "MEETING_SUMMARY.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    print(f"Output: {args.output_dir.resolve()}")
    print(f"Frames: {start:06d}-{end:06d}; valid={len(frames)}")
    print(f"Windows completed: {len(windows)}")
    print(f"Best observed model: {best_name}")
    print(f"Feature points/window/side: {selected_feature_count}")
    print(f"Manual evaluation: {manual_status}")


if __name__ == "__main__":
    main()
