"""Fit the first five pose-aligned lane observations as two smooth curves.

This experiment is deliberately separate from the legacy RANSAC pipeline.  It
does not reject points merely to increase raster overlap.  The left and right
lane identities supplied by the upstream pipeline are kept separate, each
frame is given equal influence in longitudinal bins, and one robust cubic
parametric B-spline is fitted per side.

Every run requires a new or empty output directory.  Numerical curve samples,
residuals, parameter-selection results and provenance are saved alongside the
figures so that the result can be audited without reading pixels from a plot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
from scipy.interpolate import splev, splprep
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from surf_bev.geometry import (  # noqa: E402
    image_to_ground_ipm,
    load_kitti_poses,
    transform_lane_points_by_pose,
)


SIDES = ("left", "right")
SIDE_COLORS = {"left": "#1f77b4", "right": "#d62728"}
FRAME_COLORS = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#9d755d"]
FUSION_X_RANGE = (-15.0, 15.0)
FUSION_Z_RANGE = (-10.0, 50.0)


@dataclass
class CurveFit:
    side: str
    smoothing_per_point_m2: float
    aggregate_points: np.ndarray
    aggregate_support_frames: np.ndarray
    curve_points: np.ndarray
    parameter_samples: np.ndarray
    knots: np.ndarray
    coefficients: list[np.ndarray]
    degree: int
    irls_iterations: int
    huber_delta_m: float
    aggregate_residuals_m: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit five pose-aligned lane point sets as left/right splines."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--aligned-json",
        type=Path,
        help="aligned_lane_points.json from run_full_point_pipeline.py",
    )
    source.add_argument(
        "--clrnet-json",
        type=Path,
        help="Saved CLRNet image-point JSON; requires --calib and --poses.",
    )
    parser.add_argument("--calib", type=Path)
    parser.add_argument("--poses", type=Path)
    parser.add_argument("--manual-json", type=Path)
    parser.add_argument("--reference-id", type=int, default=4)
    parser.add_argument("--camera-height", type=float, default=1.65)
    parser.add_argument("--pitch-deg", type=float, default=0.0)
    parser.add_argument("--bin-size-m", type=float, default=0.50)
    parser.add_argument(
        "--smoothing-grid",
        default="0.0025,0.01,0.04,0.16",
        help="Candidate weighted squared-error allowance per aggregate point (m^2).",
    )
    parser.add_argument("--huber-delta-m", type=float, default=0.20)
    parser.add_argument("--irls-iterations", type=int, default=4)
    parser.add_argument("--curve-samples", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {path}. Choose a new directory."
        )
    path.mkdir(parents=True, exist_ok=True)


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


def parse_projection(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        values[key.strip()] = np.asarray(
            [float(item) for item in raw.split()], dtype=np.float64
        )
    key = "P2" if "P2" in values else "P0"
    projection = values[key].reshape(3, 4)
    intrinsic = projection[:, :3]
    image_from_cam0 = np.eye(4, dtype=np.float64)
    image_from_cam0[:3, 3] = np.linalg.solve(intrinsic, projection[:, 3])
    return {"K": intrinsic, "cam0_from_image": np.linalg.inv(image_from_cam0)}


def bottom_x(lane: np.ndarray) -> float:
    threshold = np.quantile(lane[:, 1], 0.85)
    return float(np.median(lane[lane[:, 1] >= threshold, 0]))


def load_clrnet_image_lanes(path: Path) -> list[dict[str, object]]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if "items" in record:
        items = record["items"]
        candidate_key = "lanes_image_xy_px"
    elif isinstance(record.get("detector"), dict):
        items = record["detector"]["frames"]
        candidate_key = "candidate_lanes_image_xy_px"
    else:
        raise ValueError("Unsupported CLRNet JSON schema.")

    frames: list[dict[str, object]] = []
    for default_frame_id, item in enumerate(items):
        if item.get("selected"):
            selected_by_side = {entry["side"]: entry for entry in item["selected"]}
            lanes = [
                np.asarray(selected_by_side[side]["image_xy_px"], dtype=np.float64)
                for side in SIDES
            ]
            lane_indices = [int(selected_by_side[side]["lane_index"]) for side in SIDES]
            selection_mode = "upstream selected lanes"
        else:
            candidates = [
                np.asarray(lane, dtype=np.float64).reshape(-1, 2)
                for lane in item[candidate_key]
            ]
            if len(candidates) < 2:
                raise ValueError(f"Frame {default_frame_id} has fewer than two lanes.")
            ordered = sorted(enumerate(candidates), key=lambda pair: bottom_x(pair[1]))
            chosen = (ordered[0], ordered[-1])
            lane_indices = [int(pair[0]) for pair in chosen]
            lanes = [pair[1] for pair in chosen]
            selection_mode = "outermost-by-bottom-x compatibility selection"
        frames.append(
            {
                "frame_id": int(item.get("frame_id", default_frame_id)),
                "lanes": lanes,
                "lane_indices": lane_indices,
                "selection_mode": selection_mode,
            }
        )
    return frames


def metric_filter(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    valid = (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= FUSION_X_RANGE[0])
        & (points[:, 0] <= FUSION_X_RANGE[1])
        & (points[:, 1] >= FUSION_Z_RANGE[0])
        & (points[:, 1] <= FUSION_Z_RANGE[1])
    )
    return points[valid]


def align_image_lane_frames(
    frames: list[dict[str, object]],
    calibration: dict[str, np.ndarray],
    poses_path: Path,
    reference_id: int,
    camera_height: float,
    pitch_deg: float,
) -> list[dict[str, object]]:
    poses_cam0 = load_kitti_poses(poses_path)
    poses = [pose @ calibration["cam0_from_image"] for pose in poses_cam0]
    reference_pose = poses[reference_id]
    output: list[dict[str, object]] = []
    for frame in frames:
        frame_id = int(frame["frame_id"])
        aligned_lanes = []
        for lane in frame["lanes"]:
            ground = image_to_ground_ipm(
                lane,
                calibration["K"],
                camera_height=camera_height,
                pitch_deg=pitch_deg,
                z_range=(3.0, 50.0),
            )
            aligned = transform_lane_points_by_pose(
                ground,
                poses[frame_id],
                reference_pose,
                camera_height=camera_height,
                pitch_deg=pitch_deg,
            )
            aligned_lanes.append(metric_filter(aligned))
        output.append(
            {
                "frame_id": frame_id,
                "lanes": aligned_lanes,
                "lane_indices": frame.get("lane_indices"),
                "selection_mode": frame.get("selection_mode"),
            }
        )
    return output


def load_aligned_json(path: Path) -> list[dict[str, object]]:
    record = json.loads(path.read_text(encoding="utf-8"))
    frames = []
    for item in record["frames"]:
        lanes = [
            metric_filter(np.asarray(lane, dtype=np.float64))
            for lane in item["lanes_reference_xz_m"]
        ]
        if len(lanes) != 2:
            raise ValueError(
                f"Frame {item['frame_id']} has {len(lanes)} selected lanes, expected 2."
            )
        frames.append({"frame_id": int(item["frame_id"]), "lanes": lanes})
    return frames


def load_manual_frames(
    path: Path,
    calibration: dict[str, np.ndarray],
    poses_path: Path,
    reference_id: int,
    camera_height: float,
    pitch_deg: float,
) -> list[dict[str, object]]:
    record = json.loads(path.read_text(encoding="utf-8"))
    frames = [
        {
            "frame_id": int(item["frame_id"]),
            "lanes": [
                np.asarray(item[side], dtype=np.float64).reshape(-1, 2)
                for side in SIDES
            ],
        }
        for item in record["frames_xy"]
    ]
    return align_image_lane_frames(
        frames,
        calibration,
        poses_path,
        reference_id,
        camera_height,
        pitch_deg,
    )


def validate_frames(frames: list[dict[str, object]]) -> None:
    frame_ids = [int(frame["frame_id"]) for frame in frames]
    if len(frames) != 5 or frame_ids != [0, 1, 2, 3, 4]:
        raise ValueError(
            f"This first experiment requires frames 0..4 exactly; received {frame_ids}."
        )
    for frame in frames:
        if len(frame["lanes"]) != 2:
            raise ValueError(f"Frame {frame['frame_id']} does not contain two lanes.")
        if any(len(lane) < 4 for lane in frame["lanes"]):
            raise ValueError(f"Frame {frame['frame_id']} has too few lane points.")


def side_frame_points(
    frames: list[dict[str, object]], side: str
) -> list[tuple[int, np.ndarray]]:
    side_index = SIDES.index(side)
    return [
        (int(frame["frame_id"]), np.asarray(frame["lanes"][side_index]))
        for frame in frames
    ]


def aggregate_equal_frame_bins(
    frame_points: list[tuple[int, np.ndarray]], bin_size_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate with one median vote per frame in each longitudinal bin."""

    nonempty = [points for _, points in frame_points if len(points)]
    z_min = min(float(np.min(points[:, 1])) for points in nonempty)
    z_max = max(float(np.max(points[:, 1])) for points in nonempty)
    lower = np.floor(z_min / bin_size_m) * bin_size_m
    upper = np.ceil(z_max / bin_size_m) * bin_size_m
    edges = np.arange(lower, upper + 1.5 * bin_size_m, bin_size_m)
    rows = []
    supports = []
    for start, end in zip(edges[:-1], edges[1:]):
        per_frame = []
        for _, points in frame_points:
            mask = (points[:, 1] >= start) & (points[:, 1] < end)
            if np.any(mask):
                per_frame.append(np.median(points[mask], axis=0))
        if per_frame:
            per_frame_array = np.asarray(per_frame, dtype=np.float64)
            rows.append(np.median(per_frame_array, axis=0))
            supports.append(len(per_frame))
    centers = np.asarray(rows, dtype=np.float64)
    order = np.argsort(centers[:, 1])
    return centers[order], np.asarray(supports, dtype=np.int64)[order]


def chord_parameter(points: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(distances)])
    if cumulative[-1] <= 0:
        raise ValueError("Cannot parameterize a zero-length lane curve.")
    return cumulative / cumulative[-1]


def fit_robust_spline(
    side: str,
    frame_points: list[tuple[int, np.ndarray]],
    bin_size_m: float,
    smoothing_per_point_m2: float,
    huber_delta_m: float,
    irls_iterations: int,
    curve_samples: int,
) -> CurveFit:
    aggregate, supports = aggregate_equal_frame_bins(frame_points, bin_size_m)
    if len(aggregate) < 6:
        raise ValueError(f"Only {len(aggregate)} aggregate points are available for {side}.")
    u = chord_parameter(aggregate)
    weights = np.ones(len(aggregate), dtype=np.float64)
    tck = None
    residuals = np.zeros(len(aggregate), dtype=np.float64)
    completed_iterations = 0
    for iteration in range(max(1, irls_iterations)):
        smoothing = max(smoothing_per_point_m2 * len(aggregate), 1e-12)
        tck, _ = splprep(
            [aggregate[:, 0], aggregate[:, 1]],
            u=u,
            w=weights,
            k=3,
            s=smoothing,
        )
        fitted = np.column_stack(splev(u, tck))
        residuals = np.linalg.norm(aggregate - fitted, axis=1)
        weights = np.minimum(1.0, huber_delta_m / np.maximum(residuals, 1e-9))
        weights = np.maximum(weights, 0.05)
        completed_iterations = iteration + 1
    assert tck is not None
    parameter_samples = np.linspace(0.0, 1.0, curve_samples)
    curve = np.column_stack(splev(parameter_samples, tck))
    return CurveFit(
        side=side,
        smoothing_per_point_m2=smoothing_per_point_m2,
        aggregate_points=aggregate,
        aggregate_support_frames=supports,
        curve_points=curve,
        parameter_samples=parameter_samples,
        knots=np.asarray(tck[0], dtype=np.float64),
        coefficients=[np.asarray(values, dtype=np.float64) for values in tck[1]],
        degree=int(tck[2]),
        irls_iterations=completed_iterations,
        huber_delta_m=huber_delta_m,
        aggregate_residuals_m=residuals,
    )


def distances_to_curve(points: np.ndarray, curve: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty(0, dtype=np.float64)
    return cKDTree(curve).query(points, k=1)[0]


def summarize_distances(prefix: str, distances: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_count": int(len(distances)),
        f"{prefix}_mean_m": float(np.mean(distances)),
        f"{prefix}_median_m": float(np.median(distances)),
        f"{prefix}_rmse_m": float(np.sqrt(np.mean(distances**2))),
        f"{prefix}_p95_m": float(np.percentile(distances, 95)),
    }


def leave_one_frame_out(
    frames: list[dict[str, object]],
    smoothing: float,
    bin_size_m: float,
    huber_delta_m: float,
    irls_iterations: int,
    curve_samples: int,
) -> dict[str, object]:
    side_distances: dict[str, list[np.ndarray]] = {side: [] for side in SIDES}
    fold_rows = []
    for held_out in range(5):
        train_frames = [frame for frame in frames if int(frame["frame_id"]) != held_out]
        held_frame = next(frame for frame in frames if int(frame["frame_id"]) == held_out)
        for side_index, side in enumerate(SIDES):
            fit = fit_robust_spline(
                side,
                side_frame_points(train_frames, side),
                bin_size_m,
                smoothing,
                huber_delta_m,
                irls_iterations,
                curve_samples,
            )
            held_points = np.asarray(held_frame["lanes"][side_index])
            z_min, z_max = np.min(fit.curve_points[:, 1]), np.max(fit.curve_points[:, 1])
            supported = held_points[
                (held_points[:, 1] >= z_min) & (held_points[:, 1] <= z_max)
            ]
            distances = distances_to_curve(supported, fit.curve_points)
            side_distances[side].append(distances)
            fold_rows.append(
                {
                    "held_out_frame": held_out,
                    "side": side,
                    **summarize_distances("distance", distances),
                }
            )
    combined = np.concatenate(
        [distances for values in side_distances.values() for distances in values]
    )
    fold_rmses = np.asarray(
        [float(row["distance_rmse_m"]) for row in fold_rows], dtype=np.float64
    )
    summary: dict[str, object] = {
        "smoothing_per_point_m2": smoothing,
        **summarize_distances("lofo", combined),
        "lofo_fold_rmse_mean_m": float(np.mean(fold_rmses)),
        "lofo_fold_rmse_standard_error_m": float(
            np.std(fold_rmses, ddof=1) / np.sqrt(len(fold_rmses))
        ),
        "folds": fold_rows,
    }
    for side in SIDES:
        summary.update(
            summarize_distances(
                f"lofo_{side}", np.concatenate(side_distances[side])
            )
        )
    return summary


def curve_length(curve: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=1)))


def fit_metrics(
    fit: CurveFit, frame_points: list[tuple[int, np.ndarray]]
) -> dict[str, object]:
    raw = np.vstack([points for _, points in frame_points])
    distances = distances_to_curve(raw, fit.curve_points)
    return {
        "side": fit.side,
        "input_frames": len(frame_points),
        "input_points": int(len(raw)),
        "aggregate_bin_points": int(len(fit.aggregate_points)),
        "minimum_supporting_frames_per_bin": int(
            np.min(fit.aggregate_support_frames)
        ),
        "median_supporting_frames_per_bin": float(
            np.median(fit.aggregate_support_frames)
        ),
        "maximum_supporting_frames_per_bin": int(
            np.max(fit.aggregate_support_frames)
        ),
        "curve_length_m": curve_length(fit.curve_points),
        "curve_z_min_m": float(np.min(fit.curve_points[:, 1])),
        "curve_z_max_m": float(np.max(fit.curve_points[:, 1])),
        **summarize_distances("raw_to_curve", distances),
        **summarize_distances(
            "aggregate_to_curve", fit.aggregate_residuals_m
        ),
    }


def evaluate_manual_reference(
    fits: dict[str, CurveFit], manual_frames: list[dict[str, object]]
) -> list[dict[str, object]]:
    rows = []
    for side_index, side in enumerate(SIDES):
        manual = np.vstack(
            [np.asarray(frame["lanes"][side_index]) for frame in manual_frames]
        )
        curve = fits[side].curve_points
        common_z_min = max(float(np.min(curve[:, 1])), float(np.min(manual[:, 1])))
        common_z_max = min(float(np.max(curve[:, 1])), float(np.max(manual[:, 1])))
        curve_supported = curve[
            (curve[:, 1] >= common_z_min) & (curve[:, 1] <= common_z_max)
        ]
        manual_supported = manual[
            (manual[:, 1] >= common_z_min) & (manual[:, 1] <= common_z_max)
        ]
        predicted_to_manual = distances_to_curve(curve_supported, manual_supported)
        manual_to_predicted = distances_to_curve(manual_supported, curve_supported)
        rows.append(
            {
                "side": side,
                "warning": "manual pseudo-reference, not KITTI ground truth",
                "common_z_min_m": common_z_min,
                "common_z_max_m": common_z_max,
                **summarize_distances(
                    "predicted_to_manual", predicted_to_manual
                ),
                **summarize_distances(
                    "manual_to_predicted", manual_to_predicted
                ),
                "symmetric_chamfer_mean_m": float(
                    (np.mean(predicted_to_manual) + np.mean(manual_to_predicted))
                    / 2.0
                ),
                "precision_within_0p5m": float(
                    np.mean(predicted_to_manual <= 0.5)
                ),
                "recall_within_0p5m": float(
                    np.mean(manual_to_predicted <= 0.5)
                ),
            }
        )
    return rows


def no_crossing_check(fits: dict[str, CurveFit]) -> dict[str, object]:
    left = fits["left"].curve_points
    right = fits["right"].curve_points
    z_min = max(np.min(left[:, 1]), np.min(right[:, 1]))
    z_max = min(np.max(left[:, 1]), np.max(right[:, 1]))
    z = np.linspace(z_min, z_max, 300)
    left_order = np.argsort(left[:, 1])
    right_order = np.argsort(right[:, 1])
    left_x = np.interp(z, left[left_order, 1], left[left_order, 0])
    right_x = np.interp(z, right[right_order, 1], right[right_order, 0])
    widths = right_x - left_x
    return {
        "common_z_min_m": float(z_min),
        "common_z_max_m": float(z_max),
        "minimum_right_minus_left_m": float(np.min(widths)),
        "median_right_minus_left_m": float(np.median(widths)),
        "curves_cross": bool(np.any(widths <= 0.0)),
        "note": "X is rightward in the frame-4 reference camera coordinate system.",
    }


def plot_fit(
    path: Path,
    frames: list[dict[str, object]],
    fits: dict[str, CurveFit],
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 10.0))
    for frame_index, frame in enumerate(frames):
        for side_index, side in enumerate(SIDES):
            points = np.asarray(frame["lanes"][side_index])
            axis.scatter(
                points[:, 0],
                points[:, 1],
                s=8,
                alpha=0.38,
                color=FRAME_COLORS[frame_index],
                label=f"frame {frame['frame_id']}" if side_index == 0 else None,
            )
    for side in SIDES:
        fit = fits[side]
        axis.plot(
            fit.curve_points[:, 0],
            fit.curve_points[:, 1],
            linewidth=3.0,
            color=SIDE_COLORS[side],
            label=f"{side} robust cubic B-spline",
        )
    axis.set_title(title)
    axis.set_xlabel("X right in frame 000004 reference [m]")
    axis.set_ylabel("Z forward in frame 000004 reference [m]")
    axis.set_xlim(FUSION_X_RANGE)
    axis.set_ylim(FUSION_Z_RANGE)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, linewidth=0.5, alpha=0.35)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def export_fit(
    output_dir: Path,
    fits: dict[str, CurveFit],
    frames: list[dict[str, object]],
) -> None:
    for side in SIDES:
        fit = fits[side]
        write_csv(
            output_dir / f"{side}_curve_xz.csv",
            [
                {
                    "sample_index": index,
                    "parameter_u": float(u),
                    "x_right_m": float(point[0]),
                    "z_forward_m": float(point[1]),
                }
                for index, (u, point) in enumerate(
                    zip(fit.parameter_samples, fit.curve_points)
                )
            ],
        )
        write_csv(
            output_dir / f"{side}_aggregate_bins.csv",
            [
                {
                    "bin_point_index": index,
                    "x_right_m": float(point[0]),
                    "z_forward_m": float(point[1]),
                    "supporting_frames": int(support),
                    "fit_residual_m": float(residual),
                }
                for index, (point, support, residual) in enumerate(
                    zip(
                        fit.aggregate_points,
                        fit.aggregate_support_frames,
                        fit.aggregate_residuals_m,
                    )
                )
            ],
        )
    write_csv(
        output_dir / "aligned_input_points_xz.csv",
        [
            {
                "frame_id": int(frame["frame_id"]),
                "side": side,
                "point_index": point_index,
                "x_right_m": float(point[0]),
                "z_forward_m": float(point[1]),
            }
            for frame in frames
            for side_index, side in enumerate(SIDES)
            for point_index, point in enumerate(frame["lanes"][side_index])
        ],
    )


def model_document(fits: dict[str, CurveFit]) -> dict[str, object]:
    return {
        "model": "robust cubic parametric B-spline",
        "coordinate_system": (
            "metric X/Z in reference image camera frame 4; X right, Z forward"
        ),
        "curves": {
            side: {
                "degree": fit.degree,
                "knots": fit.knots.tolist(),
                "coefficients_x_z": [values.tolist() for values in fit.coefficients],
                "valid_parameter_range": [0.0, 1.0],
                "smoothing_per_point_m2": fit.smoothing_per_point_m2,
                "huber_delta_m": fit.huber_delta_m,
                "irls_iterations": fit.irls_iterations,
                "no_extrapolation": True,
            }
            for side, fit in fits.items()
        },
    }


def main() -> None:
    args = parse_args()
    required = [args.aligned_json or args.clrnet_json]
    if args.clrnet_json is not None:
        if args.calib is None or args.poses is None:
            raise ValueError("--clrnet-json requires --calib and --poses.")
        required.extend([args.calib, args.poses])
    if args.manual_json is not None:
        if args.calib is None or args.poses is None:
            raise ValueError("--manual-json requires --calib and --poses.")
        required.extend([args.manual_json, args.calib, args.poses])
    missing = [str(path) for path in required if path is None or not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))
    smoothing_grid = [float(value) for value in args.smoothing_grid.split(",")]
    if any(value <= 0 for value in smoothing_grid):
        raise ValueError("Every smoothing-grid value must be positive.")
    require_empty_output(args.output_dir)

    calibration = parse_projection(args.calib) if args.calib else None
    if args.aligned_json is not None:
        frames = load_aligned_json(args.aligned_json)
        source_mode = "upstream pose-aligned metric points"
    else:
        assert args.clrnet_json is not None and calibration is not None and args.poses
        image_frames = load_clrnet_image_lanes(args.clrnet_json)
        frames = align_image_lane_frames(
            image_frames,
            calibration,
            args.poses,
            args.reference_id,
            args.camera_height,
            args.pitch_deg,
        )
        source_mode = "saved CLRNet image points reconstructed to metric coordinates"
    validate_frames(frames)

    trial_summaries = []
    for smoothing in smoothing_grid:
        cv = leave_one_frame_out(
            frames,
            smoothing,
            args.bin_size_m,
            args.huber_delta_m,
            args.irls_iterations,
            args.curve_samples,
        )
        trial_summaries.append(cv)
        trial_name = f"s_{str(smoothing).replace('.', 'p')}"
        trial_dir = args.output_dir / "02_parameter_trials" / trial_name
        trial_fits = {
            side: fit_robust_spline(
                side,
                side_frame_points(frames, side),
                args.bin_size_m,
                smoothing,
                args.huber_delta_m,
                args.irls_iterations,
                args.curve_samples,
            )
            for side in SIDES
        }
        plot_fit(
            trial_dir / "two_curves.png",
            frames,
            trial_fits,
            f"First five frames: smoothing={smoothing:g} m^2/point",
        )
        write_csv(trial_dir / "leave_one_frame_out_folds.csv", cv["folds"])
        write_csv(
            trial_dir / "fit_metrics.csv",
            [
                fit_metrics(trial_fits[side], side_frame_points(frames, side))
                for side in SIDES
            ],
        )
        export_fit(trial_dir / "curve_data", trial_fits, frames)

    # Use the standard one-standard-error model-selection rule: locate the
    # candidate with the smallest mean held-out fold RMSE, then choose the
    # smoothest candidate whose mean remains within one standard error of that
    # minimum.  This avoids choosing a visibly wiggly curve for a numerically
    # negligible held-out improvement.
    minimum_cv = min(
        trial_summaries, key=lambda row: float(row["lofo_fold_rmse_mean_m"])
    )
    cv_limit = float(minimum_cv["lofo_fold_rmse_mean_m"]) + float(
        minimum_cv["lofo_fold_rmse_standard_error_m"]
    )
    eligible = [
        row
        for row in trial_summaries
        if float(row["lofo_fold_rmse_mean_m"]) <= cv_limit
    ]
    selected = max(eligible, key=lambda row: float(row["smoothing_per_point_m2"]))
    selected_smoothing = float(selected["smoothing_per_point_m2"])
    fits = {
        side: fit_robust_spline(
            side,
            side_frame_points(frames, side),
            args.bin_size_m,
            selected_smoothing,
            args.huber_delta_m,
            args.irls_iterations,
            args.curve_samples,
        )
        for side in SIDES
    }
    selected_dir = args.output_dir / "03_selected_result"
    plot_fit(
        selected_dir / "five_frames_two_smooth_curves.png",
        frames,
        fits,
        "KITTI 00 frames 000000-000004: two pose-aligned smooth curves",
    )
    export_fit(selected_dir / "curve_data", fits, frames)
    metrics = [
        fit_metrics(fits[side], side_frame_points(frames, side)) for side in SIDES
    ]
    write_csv(selected_dir / "fit_metrics.csv", metrics)
    (selected_dir / "curve_model.json").write_text(
        json.dumps(model_document(fits), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    crossing = no_crossing_check(fits)

    manual_rows = None
    if args.manual_json is not None:
        assert calibration is not None and args.poses is not None
        manual_frames = load_manual_frames(
            args.manual_json,
            calibration,
            args.poses,
            args.reference_id,
            args.camera_height,
            args.pitch_deg,
        )
        manual_rows = evaluate_manual_reference(fits, manual_frames)
        write_csv(
            args.output_dir
            / "04_manual_pseudo_reference_evaluation"
            / "geometry_metrics.csv",
            manual_rows,
        )

    selection_rows = [
        {key: value for key, value in summary.items() if key != "folds"}
        for summary in trial_summaries
    ]
    write_csv(args.output_dir / "00_audit" / "smoothing_selection.csv", selection_rows)
    inputs = {}
    for label, path in (
        ("aligned_json", args.aligned_json),
        ("clrnet_json", args.clrnet_json),
        ("calib", args.calib),
        ("poses", args.poses),
        ("manual_json", args.manual_json),
    ):
        if path is not None:
            inputs[label] = {"path": str(path.resolve()), "sha256": sha256(path)}
    audit = {
        "status": "complete",
        "scope": "KITTI Odometry Sequence 00 frames 000000-000004 only",
        "source_mode": source_mode,
        "inputs": inputs,
        "coordinate_system": {
            "reference_frame": args.reference_id,
            "x": "right, metres",
            "z": "forward, metres",
            "pose_formula": "p_ref = inv(T_world_ref) @ T_world_src @ p_src",
        },
        "method": {
            "identity_handling": "left and right lanes are never pooled together",
            "aggregation": (
                f"{args.bin_size_m:g} m longitudinal bins; one median vote per frame, "
                "then median across frames"
            ),
            "curve": "degree-3 parametric B-spline",
            "robustness": (
                f"Huber-style IRLS weights, delta={args.huber_delta_m:g} m, "
                f"iterations={args.irls_iterations}"
            ),
            "smoothing_selection": (
                "one-standard-error rule on leave-one-frame-out fold RMSE: choose "
                "the smoothest candidate within one standard error of the minimum; "
                "manual pseudo-reference is not used for parameter selection"
            ),
            "minimum_cv_fold_rmse_mean_m": float(
                minimum_cv["lofo_fold_rmse_mean_m"]
            ),
            "one_standard_error_limit_m": cv_limit,
            "candidate_smoothing_per_point_m2": smoothing_grid,
            "selected_smoothing_per_point_m2": selected_smoothing,
            "extrapolation": "disabled; each curve is output only on observed support",
        },
        "selected_leave_one_frame_out": {
            key: value for key, value in selected.items() if key != "folds"
        },
        "fit_metrics": metrics,
        "left_right_check": crossing,
        "manual_pseudo_reference_metrics": manual_rows,
        "limitations": [
            "Manual reference points are pseudo-labels, not KITTI lane ground truth.",
            "The fit inherits any wrong upstream left/right lane selection.",
            "Flat-road IPM uses the project assumptions supplied to this run.",
            "Five mostly straight frames do not validate curved-road performance.",
        ],
    }
    (args.output_dir / "00_audit" / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "STATUS.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "selected_smoothing_per_point_m2": selected_smoothing,
                "curves_cross": crossing["curves_cross"],
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Output: {args.output_dir.resolve()}")
    print(f"Selected smoothing: {selected_smoothing:g} m^2 per aggregate point")
    print(f"Curves cross: {crossing['curves_cross']}")
    for row in metrics:
        print(
            f"{row['side']}: {row['input_points']} points, "
            f"raw-to-curve RMSE={row['raw_to_curve_rmse_m']:.4f} m, "
            f"P95={row['raw_to_curve_p95_m']:.4f} m"
        )


if __name__ == "__main__":
    main()
