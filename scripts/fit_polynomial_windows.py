"""Fit a long lane section with local robust polynomials in metric X/Z.

The input lane observations stay in their native metric ground-plane X/Z
representation.  Each short window is first aligned to its own KITTI pose,
then fitted independently on the left and right.  Every window uses the same
robust low-degree parametric polynomial.  Its dimensionless parameter only
orders samples along a curve; all fitted geometry remains metric X/Z.

Every sampled window curve is transformed to one requested common camera
reference frame.  Overlapping samples are combined into one polyline per side,
and both numerical continuity diagnostics and complete provenance are saved.
No existing result directory is overwritten.
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
from typing import Iterable, Sequence

import matplotlib
import numpy as np
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.calibration import parse_projection  # noqa: E402
from surf_bev.geometry import (  # noqa: E402
    load_kitti_poses,
    transform_lane_points_by_pose,
)


SIDES = ("left", "right")
SIDE_COLOURS = {"left": "#1f77b4", "right": "#d62728"}
MODEL_COLOURS = {"parametric_polynomial": "#2b6cb0"}


@dataclass(frozen=True)
class WindowSpec:
    window_id: int
    start_frame: int
    end_frame: int
    reference_frame: int
    terminal_anchor: bool


@dataclass
class ParametricFit:
    model: str
    degree: int
    aggregate_xz: np.ndarray
    aggregate_support_frames: np.ndarray
    parameter: np.ndarray
    curve_xz: np.ndarray
    residuals_m: np.ndarray
    coefficients: dict[str, object]
    huber_delta_m: float
    irls_iterations: int


@dataclass
class WindowResult:
    spec: WindowSpec
    classification: str
    net_heading_change_deg: float
    total_absolute_heading_change_deg: float
    trajectory_path_length_m: float
    side: str
    selected_model: str
    valid_frame_ids: list[int]
    fit_local: ParametricFit
    curve_common_xz: np.ndarray
    aggregate_common_xz: np.ndarray
    sample_route_keys: np.ndarray | None = None
    model_selection_reason: str = ""
    segment_id: int = -1


@dataclass
class BlendedSegment:
    side: str
    segment_id: int
    parent_window_segment_id: int
    ordering_keys: np.ndarray
    points_xz: np.ndarray
    supports: np.ndarray
    start_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Final polynomial-only left/right lane fitting in pose-aligned metric X/Z."
        )
    )
    parser.add_argument("--selected-lane-points", type=Path, required=True)
    parser.add_argument("--scan-json", type=Path)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--sequence-id", default="01")
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--reference-frame", type=int)
    parser.add_argument("--core-end-frame", type=int)
    parser.add_argument("--window-length", type=int, default=15)
    parser.add_argument("--window-stride", type=int, default=10)
    parser.add_argument("--straight-max-curvature-1-per-m", type=float, default=0.004)
    parser.add_argument("--curve-min-curvature-1-per-m", type=float, default=0.005)
    parser.add_argument("--curvature-persistence-frames", type=int, default=3)
    parser.add_argument("--curvature-smoothing-window", type=int, default=11)
    parser.add_argument("--curvature-majority-fraction", type=float, default=0.60)
    parser.add_argument("--polynomial-degree", type=int, default=2)
    parser.add_argument("--bin-size-m", type=float, default=0.50)
    parser.add_argument("--huber-delta-m", type=float, default=0.20)
    parser.add_argument("--irls-iterations", type=int, default=4)
    parser.add_argument("--curve-samples", type=int, default=240)
    parser.add_argument("--minimum-valid-frames-per-side", type=int, default=8)
    parser.add_argument("--maximum-missing-run-frames", type=int, default=3)
    parser.add_argument("--maximum-cv-folds", type=int, default=8)
    parser.add_argument("--blend-key-bin-width", type=float, default=0.25)
    parser.add_argument("--blend-maximum-route-distance-m", type=float, default=12.0)
    parser.add_argument("--continuity-p95-gate-m", type=float, default=0.75)
    parser.add_argument("--continuity-angle-gate-deg", type=float, default=30.0)
    parser.add_argument("--maximum-internal-order-gap-frames", type=float, default=3.0)
    parser.add_argument("--maximum-internal-node-gap-m", type=float, default=5.0)
    parser.add_argument("--camera-height", type=float)
    parser.add_argument("--pitch-deg", type=float)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    records = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in records:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def require_new_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {path}. Choose a new directory."
        )
    path.mkdir(parents=True, exist_ok=True)


def validate_args(args: argparse.Namespace) -> None:
    missing = [
        str(path)
        for path in (args.selected_lane_points, args.poses, args.calib)
        if not path.is_file()
    ]
    if args.scan_json is not None and not args.scan_json.is_file():
        missing.append(str(args.scan_json))
    if missing:
        raise FileNotFoundError("Missing input files: " + ", ".join(missing))
    if args.start_frame < 0 or args.end_frame < args.start_frame:
        raise ValueError("Frame range must be increasing and non-negative.")
    if args.window_length < 5 or args.window_stride < 1:
        raise ValueError("window-length must be >=5 and window-stride >=1.")
    if args.end_frame - args.start_frame + 1 < args.window_length:
        raise ValueError("Frame range is shorter than one full window.")
    if args.core_end_frame is not None and not (
        args.start_frame <= args.core_end_frame <= args.end_frame
    ):
        raise ValueError("core-end-frame must fall inside the requested range.")
    if not 1 <= args.polynomial_degree <= 3:
        raise ValueError("polynomial-degree must be between 1 and 3.")
    if not 0 < args.straight_max_curvature_1_per_m < args.curve_min_curvature_1_per_m:
        raise ValueError(
            "Curvature thresholds must satisfy 0 < straight maximum < curve minimum."
        )
    if args.curvature_persistence_frames < 1:
        raise ValueError("curvature-persistence-frames must be at least one.")
    if args.curvature_smoothing_window < 3 or args.curvature_smoothing_window % 2 == 0:
        raise ValueError("curvature-smoothing-window must be an odd number >=3.")
    if not 0.5 <= args.curvature_majority_fraction <= 1.0:
        raise ValueError("curvature-majority-fraction must be in [0.5, 1.0].")
    if min(
        args.bin_size_m,
        args.huber_delta_m,
        args.blend_key_bin_width,
        args.blend_maximum_route_distance_m,
        args.continuity_p95_gate_m,
        args.continuity_angle_gate_deg,
        args.maximum_internal_order_gap_frames,
        args.maximum_internal_node_gap_m,
    ) <= 0:
        raise ValueError("Distance, bin and smoothing parameters must be positive.")
    if args.irls_iterations < 1 or args.curve_samples < 30:
        raise ValueError("irls-iterations must be >=1 and curve-samples >=30.")
    if args.minimum_valid_frames_per_side < 2:
        raise ValueError("minimum-valid-frames-per-side must be at least two.")
    if args.minimum_valid_frames_per_side > args.window_length:
        raise ValueError("minimum-valid-frames-per-side cannot exceed window-length.")
    if args.maximum_missing_run_frames < 0:
        raise ValueError("maximum-missing-run-frames cannot be negative.")
    if args.maximum_cv_folds < 0:
        raise ValueError("maximum-cv-folds cannot be negative.")
    require_new_output(args.output_dir)


def build_windows(
    start_frame: int, end_frame: int, length: int, stride: int
) -> list[WindowSpec]:
    last_start = end_frame - length + 1
    starts = list(range(start_frame, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return [
        WindowSpec(
            window_id=index,
            start_frame=start,
            end_frame=start + length - 1,
            reference_frame=start + length - 1,
            terminal_anchor=(index == len(starts) - 1 and start != start_frame + index * stride),
        )
        for index, start in enumerate(starts)
    ]


def camera_positions_xz(
    poses_image: Sequence[np.ndarray],
    frame_ids: Sequence[int],
    reference_frame: int,
) -> np.ndarray:
    reference_from_world = np.linalg.inv(poses_image[reference_frame])
    output = []
    for frame_id in frame_ids:
        origin_world = poses_image[frame_id] @ np.array([0.0, 0.0, 0.0, 1.0])
        output.append((reference_from_world @ origin_world)[[0, 2]])
    return np.asarray(output, dtype=np.float64)


def trajectory_turn_statistics(points_xz: np.ndarray) -> dict[str, float]:
    """Match the scan-stage robust position-trajectory turn calculation."""

    points = np.asarray(points_xz, dtype=np.float64).reshape(-1, 2)
    steps = np.diff(points, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    keep = lengths > 1e-4
    steps = steps[keep]
    lengths = lengths[keep]
    if len(steps) < 2:
        return {
            "path_length_m": float(lengths.sum()),
            "net_heading_change_deg": 0.0,
            "total_absolute_heading_change_deg": 0.0,
            "trajectory_net_heading_change_deg": 0.0,
        }
    headings = np.unwrap(np.arctan2(steps[:, 0], steps[:, 1]))
    changes = np.diff(headings)
    baseline = max(1, min(10, (len(points) - 1) // 3))
    start_vector = points[baseline] - points[0]
    end_vector = points[-1] - points[-1 - baseline]
    start_heading = float(np.arctan2(start_vector[0], start_vector[1]))
    end_heading = float(np.arctan2(end_vector[0], end_vector[1]))
    trajectory_turn = float(
        np.arctan2(
            np.sin(end_heading - start_heading),
            np.cos(end_heading - start_heading),
        )
    )
    return {
        "path_length_m": float(lengths.sum()),
        "net_heading_change_deg": float(np.degrees(headings[-1] - headings[0])),
        "total_absolute_heading_change_deg": float(
            np.degrees(np.sum(np.abs(changes)))
        ),
        "trajectory_net_heading_change_deg": float(np.degrees(trajectory_turn)),
    }


def absolute_pose_curvature(points_xz: np.ndarray, smoothing_window: int) -> np.ndarray:
    """Compute smoothed planar absolute curvature for an ordered pose path."""

    points = np.asarray(points_xz, dtype=np.float64).reshape(-1, 2)
    if len(points) < 3:
        return np.zeros(len(points), dtype=np.float64)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    valid = np.diff(arc) > 1e-8
    if not np.all(valid):
        keep = np.concatenate([[True], valid])
        points = points[keep]
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(steps)])
    if len(points) < 3 or np.any(np.diff(arc) <= 1e-8):
        return np.zeros(len(points_xz), dtype=np.float64)
    window = min(smoothing_window, len(points) if len(points) % 2 else len(points) - 1)
    if window >= 3:
        x = savgol_filter(points[:, 0], window, 2, mode="interp")
        z = savgol_filter(points[:, 1], window, 2, mode="interp")
    else:
        x, z = points[:, 0], points[:, 1]
    dx = np.gradient(x, arc)
    dz = np.gradient(z, arc)
    ddx = np.gradient(dx, arc)
    ddz = np.gradient(dz, arc)
    speed_sq = np.maximum(dx * dx + dz * dz, 1e-12)
    curvature = np.abs((dx * ddz - dz * ddx) / np.power(speed_sq, 1.5))
    if len(curvature) == len(points_xz):
        return curvature
    output = np.zeros(len(points_xz), dtype=np.float64)
    output[np.flatnonzero(keep)] = curvature
    return output


def persistent_curvature_labels(
    curvature: np.ndarray,
    straight_max: float,
    curve_min: float,
    persistence_frames: int,
) -> np.ndarray:
    """Keep only persistent high/low curvature runs; mark the rest transition."""

    values = np.asarray(curvature, dtype=np.float64)
    raw = np.full(len(values), "transition", dtype=object)
    raw[values <= straight_max] = "straight"
    raw[values >= curve_min] = "curve"
    labels = np.full(len(values), "transition", dtype=object)
    start = 0
    while start < len(raw):
        end = start + 1
        while end < len(raw) and raw[end] == raw[start]:
            end += 1
        if raw[start] != "transition" and end - start >= persistence_frames:
            labels[start:end] = raw[start]
        start = end
    return labels


def classify_curvature_window(
    curvature: np.ndarray,
    straight_max: float,
    curve_min: float,
    persistence_frames: int,
    majority_fraction: float,
) -> tuple[str, dict[str, float]]:
    labels = persistent_curvature_labels(
        curvature, straight_max, curve_min, persistence_frames
    )
    fractions = {
        name: float(np.mean(labels == name)) if len(labels) else 0.0
        for name in ("straight", "curve", "transition")
    }
    if fractions["curve"] >= majority_fraction:
        classification = "curve"
    elif fractions["straight"] >= majority_fraction:
        classification = "straight"
    else:
        classification = "transition"
    return classification, fractions


def load_selected_frames(
    document: dict[str, object], start_frame: int, end_frame: int
) -> dict[int, dict[str, object]]:
    frames = document.get("frames")
    if not isinstance(frames, list):
        raise ValueError("Selected-lane-points JSON does not contain a frames list.")
    by_id: dict[int, dict[str, object]] = {}
    for raw in frames:
        if not isinstance(raw, dict) or "frame_id" not in raw:
            continue
        frame_id = int(raw["frame_id"])
        if start_frame <= frame_id <= end_frame:
            by_id[frame_id] = raw
    if not by_id:
        raise ValueError("No selected lane records fall inside the requested range.")
    return by_id


def valid_local_lane(
    frame: dict[str, object], side_index: int
) -> np.ndarray | None:
    lanes = frame.get("selected_lanes_local_ground_xz_m")
    if isinstance(lanes, dict):
        raw_lane = lanes.get(SIDES[side_index])
    elif isinstance(lanes, list) and side_index < len(lanes):
        raw_lane = lanes[side_index]
    else:
        return None
    if raw_lane is None:
        return None
    raw_array = np.asarray(raw_lane, dtype=np.float64)
    if raw_array.size == 0 or raw_array.size % 2:
        return None
    points = raw_array.reshape(-1, 2)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 4:
        return None
    order = np.argsort(points[:, 1], kind="stable")
    return points[order]


def align_window_observations(
    spec: WindowSpec,
    side_index: int,
    selected_by_id: dict[int, dict[str, object]],
    poses_image: Sequence[np.ndarray],
    camera_height: float,
    pitch_deg: float,
) -> list[tuple[int, np.ndarray]]:
    observations: list[tuple[int, np.ndarray]] = []
    reference_pose = poses_image[spec.reference_frame]
    for frame_id in range(spec.start_frame, spec.end_frame + 1):
        record = selected_by_id.get(frame_id)
        if record is None:
            continue
        points = valid_local_lane(record, side_index)
        if points is None:
            continue
        aligned = transform_lane_points_by_pose(
            points,
            poses_image[frame_id],
            reference_pose,
            camera_height=camera_height,
            pitch_deg=pitch_deg,
        )
        aligned = aligned[np.isfinite(aligned).all(axis=1)]
        if len(aligned) >= 4:
            observations.append((frame_id, aligned))
    return observations


def longest_missing_run(
    start_frame: int, end_frame: int, valid_frame_ids: Sequence[int]
) -> int:
    valid = set(int(frame_id) for frame_id in valid_frame_ids)
    longest = 0
    current = 0
    for frame_id in range(start_frame, end_frame + 1):
        if frame_id in valid:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def short_internal_gaps(
    start_frame: int,
    end_frame: int,
    valid_frame_ids: Sequence[int],
    maximum_gap: int,
) -> list[int]:
    """Return only missing frames bracketed by observations on both sides."""

    valid = set(int(frame_id) for frame_id in valid_frame_ids)
    gaps: list[int] = []
    frame = start_frame
    while frame <= end_frame:
        if frame in valid:
            frame += 1
            continue
        gap_start = frame
        while frame <= end_frame and frame not in valid:
            frame += 1
        gap_end = frame - 1
        gap_length = gap_end - gap_start + 1
        bracketed = gap_start > start_frame and gap_end < end_frame
        if bracketed and gap_length <= maximum_gap:
            gaps.extend(range(gap_start, gap_end + 1))
    return gaps


def polyline_order_key(points_xz: np.ndarray, trajectory_xz: np.ndarray) -> np.ndarray:
    """Order X/Z points by nearest progress on the local pose polyline.

    The returned scalar is used only to order and bin observations.  No fitted
    coordinate is replaced: model targets remain the original X and Z values.
    """

    points = np.asarray(points_xz, dtype=np.float64).reshape(-1, 2)
    trajectory = np.asarray(trajectory_xz, dtype=np.float64).reshape(-1, 2)
    steps = np.diff(trajectory, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    keep = lengths > 1e-6
    if not np.any(keep):
        raise ValueError("Window trajectory has no usable motion segment.")
    starts = trajectory[:-1][keep]
    steps = steps[keep]
    lengths = lengths[keep]
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    best_distance = np.full(len(points), np.inf, dtype=np.float64)
    best_order = np.zeros(len(points), dtype=np.float64)
    last = len(steps) - 1
    for index, (start, step, length) in enumerate(zip(starts, steps, lengths)):
        raw = ((points - start) @ step) / (length * length)
        if last == 0:
            fraction = raw
        elif index == 0:
            fraction = np.minimum(raw, 1.0)
        elif index == last:
            fraction = np.maximum(raw, 0.0)
        else:
            fraction = np.clip(raw, 0.0, 1.0)
        projection = start + fraction[:, None] * step
        distances = np.linalg.norm(points - projection, axis=1)
        update = distances < best_distance
        best_distance[update] = distances[update]
        best_order[update] = cumulative[index] + fraction[update] * length
    return best_order


def aggregate_equal_frame_bins_xz(
    observations: Sequence[tuple[int, np.ndarray]],
    trajectory_xz: np.ndarray,
    bin_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create one median vote per frame and local ordering bin."""

    if not observations:
        raise ValueError("No lane observations are available for aggregation.")
    projected = [
        polyline_order_key(points, trajectory_xz) for _, points in observations
    ]
    lower = math.floor(min(float(values.min()) for values in projected) / bin_size_m)
    upper = math.ceil(max(float(values.max()) for values in projected) / bin_size_m)
    edges = np.arange(
        lower * bin_size_m,
        (upper + 1.5) * bin_size_m,
        bin_size_m,
        dtype=np.float64,
    )
    rows: list[np.ndarray] = []
    supports: list[int] = []
    order_values: list[float] = []
    for bin_start, bin_end in zip(edges[:-1], edges[1:]):
        votes = []
        for (_, points), values in zip(observations, projected):
            mask = (values >= bin_start) & (values < bin_end)
            if np.any(mask):
                votes.append(np.median(points[mask], axis=0))
        if votes:
            vote_array = np.asarray(votes, dtype=np.float64)
            rows.append(np.median(vote_array, axis=0))
            supports.append(len(votes))
            order_values.append(0.5 * (bin_start + bin_end))
    if len(rows) < 5:
        raise ValueError(f"Only {len(rows)} aggregate X/Z bins are available.")
    aggregate = np.asarray(rows, dtype=np.float64)
    support_array = np.asarray(supports, dtype=np.int64)
    order = np.argsort(np.asarray(order_values), kind="stable")
    aggregate = aggregate[order]
    support_array = support_array[order]

    # Repeated or nearly repeated aggregate points destabilise a parametric fit.
    step = np.linalg.norm(np.diff(aggregate, axis=0), axis=1)
    keep = np.concatenate([[True], step > 1e-4])
    aggregate = aggregate[keep]
    support_array = support_array[keep]
    if len(aggregate) < 5:
        raise ValueError("Too few distinct aggregate X/Z points remain.")
    return aggregate, support_array


def huber_weights(residuals_m: np.ndarray, delta_m: float) -> np.ndarray:
    residuals = np.asarray(residuals_m, dtype=np.float64)
    return np.maximum(
        np.minimum(1.0, delta_m / np.maximum(residuals, 1e-9)), 0.05
    )


def residual_summary(residuals_m: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(residuals_m, dtype=np.float64)
    if not len(values):
        return {
            "count": 0,
            "mean_m": float("nan"),
            "rmse_m": float("nan"),
            "median_m": float("nan"),
            "p90_m": float("nan"),
            "p95_m": float("nan"),
            "max_m": float("nan"),
        }
    return {
        "count": int(len(values)),
        "mean_m": float(np.mean(values)),
        "rmse_m": float(np.sqrt(np.mean(values**2))),
        "median_m": float(np.median(values)),
        "p90_m": float(np.percentile(values, 90)),
        "p95_m": float(np.percentile(values, 95)),
        "max_m": float(np.max(values)),
    }


def fit_parametric_polynomial(
    aggregate_xz: np.ndarray,
    support_frames: np.ndarray,
    degree: int,
    huber_delta_m: float,
    irls_iterations: int,
    curve_samples: int,
) -> ParametricFit:
    # Aggregate rows come from equal-width ordering bins.  Their row index is
    # therefore a stable, dimensionless ordering parameter and is deliberately
    # not changed by one large geometric outlier.
    parameter = np.linspace(0.0, 1.0, len(aggregate_xz))
    design = np.vander(parameter, N=degree + 1, increasing=True)
    weights = np.ones(len(parameter), dtype=np.float64)
    coefficients = np.zeros((degree + 1, 2), dtype=np.float64)
    residuals = np.zeros(len(parameter), dtype=np.float64)
    for _ in range(max(1, irls_iterations)):
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_values = aggregate_xz * np.sqrt(weights)[:, None]
        coefficients = np.linalg.lstsq(
            weighted_design, weighted_values, rcond=None
        )[0]
        predicted = design @ coefficients
        residuals = np.linalg.norm(aggregate_xz - predicted, axis=1)
        weights = huber_weights(residuals, huber_delta_m)
    sample_parameter = np.linspace(0.0, 1.0, curve_samples)
    curve = np.vander(
        sample_parameter, N=degree + 1, increasing=True
    ) @ coefficients
    return ParametricFit(
        model="parametric_polynomial",
        degree=degree,
        aggregate_xz=aggregate_xz,
        aggregate_support_frames=support_frames,
        parameter=sample_parameter,
        curve_xz=curve,
        residuals_m=residuals,
        coefficients={
            "x_in_increasing_power_order": coefficients[:, 0].tolist(),
            "z_in_increasing_power_order": coefficients[:, 1].tolist(),
        },
        huber_delta_m=huber_delta_m,
        irls_iterations=irls_iterations,
    )


def fit_model(
    model: str,
    aggregate_xz: np.ndarray,
    support_frames: np.ndarray,
    args: argparse.Namespace,
) -> ParametricFit:
    if model == "parametric_polynomial":
        return fit_parametric_polynomial(
            aggregate_xz,
            support_frames,
            args.polynomial_degree,
            args.huber_delta_m,
            args.irls_iterations,
            args.curve_samples,
        )
    raise ValueError(f"Unsupported model: {model}")


def points_to_curve_distances(points_xz: np.ndarray, curve_xz: np.ndarray) -> np.ndarray:
    if not len(points_xz) or not len(curve_xz):
        return np.empty(0, dtype=np.float64)
    return cKDTree(np.asarray(curve_xz)).query(np.asarray(points_xz), k=1)[0]


def cross_validate_model(
    model: str,
    observations: Sequence[tuple[int, np.ndarray]],
    trajectory_xz: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, float | int]:
    if args.maximum_cv_folds == 0 or len(observations) < 4:
        return {"heldout_fold_count": 0, **residual_summary(np.empty(0))}
    fold_count = min(args.maximum_cv_folds, len(observations))
    fold_indices = np.unique(
        np.linspace(0, len(observations) - 1, fold_count).round().astype(int)
    )
    distances: list[np.ndarray] = []
    successful = 0
    for held_index in fold_indices:
        training = [
            row for index, row in enumerate(observations) if index != held_index
        ]
        try:
            aggregate, supports = aggregate_equal_frame_bins_xz(
                training, trajectory_xz, args.bin_size_m
            )
            fit = fit_model(model, aggregate, supports, args)
        except (ValueError, TypeError, RuntimeError):
            continue
        distances.append(
            points_to_curve_distances(observations[held_index][1], fit.curve_xz)
        )
        successful += 1
    combined = np.concatenate(distances) if distances else np.empty(0)
    return {"heldout_fold_count": successful, **residual_summary(combined)}


def transform_xz_between_references(
    points_xz: np.ndarray,
    source_reference: int,
    target_reference: int,
    poses_image: Sequence[np.ndarray],
    camera_height: float,
    pitch_deg: float,
) -> np.ndarray:
    if source_reference == target_reference:
        return np.asarray(points_xz, dtype=np.float64).copy()
    return transform_lane_points_by_pose(
        np.asarray(points_xz, dtype=np.float64),
        poses_image[source_reference],
        poses_image[target_reference],
        camera_height=camera_height,
        pitch_deg=pitch_deg,
    )


def route_tangents(route_xz: np.ndarray) -> np.ndarray:
    tangents = np.empty_like(route_xz)
    tangents[0] = route_xz[1] - route_xz[0]
    tangents[-1] = route_xz[-1] - route_xz[-2]
    if len(route_xz) > 2:
        tangents[1:-1] = route_xz[2:] - route_xz[:-2]
    norms = np.linalg.norm(tangents, axis=1)
    valid = norms > 1e-8
    tangents[valid] /= norms[valid, None]
    tangents[~valid] = np.array([0.0, 1.0])
    return tangents


def assign_route_keys(
    curve_xz: np.ndarray,
    route_xz: np.ndarray,
    route_tree: cKDTree,
    tangents: np.ndarray,
    window_start_index: int,
    window_end_index: int,
    maximum_distance_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    distances, nearest = route_tree.query(curve_xz, k=1)
    steps = np.linalg.norm(np.diff(route_xz, axis=0), axis=1)
    typical_step = max(float(np.median(steps[steps > 1e-5])), 1e-3)
    local = np.einsum(
        "ij,ij->i", curve_xz - route_xz[nearest], tangents[nearest]
    )
    fraction = np.clip(local / typical_step, -0.49, 0.49)
    keys = nearest.astype(np.float64) + fraction
    keep = (
        (distances <= maximum_distance_m)
        & (nearest >= window_start_index)
        & (nearest <= window_end_index)
    )
    return keys, keep


def collapse_keyed_samples(
    keys: np.ndarray, points_xz: np.ndarray, bin_width: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(keys, kind="stable")
    keys = np.asarray(keys)[order]
    points = np.asarray(points_xz)[order]
    bin_ids = np.floor(keys / bin_width).astype(np.int64)
    output_keys = []
    output_points = []
    support = []
    for bin_id in np.unique(bin_ids):
        selected = bin_ids == bin_id
        output_keys.append(float(np.median(keys[selected])))
        output_points.append(np.median(points[selected], axis=0))
        support.append(int(np.sum(selected)))
    return (
        np.asarray(output_keys, dtype=np.float64),
        np.asarray(output_points, dtype=np.float64),
        np.asarray(support, dtype=np.int64),
    )


def interpolate_keyed_curve(
    keys: np.ndarray, points_xz: np.ndarray, samples: np.ndarray
) -> np.ndarray:
    compact_keys, compact_points, _ = collapse_keyed_samples(
        keys, points_xz, max(float(np.ptp(keys)) / max(len(keys), 1), 1e-6)
    )
    unique_keys, unique_indices = np.unique(compact_keys, return_index=True)
    compact_points = compact_points[unique_indices]
    if len(unique_keys) < 2:
        raise ValueError("Too few unique route keys for interpolation.")
    return np.column_stack(
        [
            np.interp(samples, unique_keys, compact_points[:, axis])
            for axis in range(2)
        ]
    )


def vector_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 1e-9 or second_norm <= 1e-9:
        return float("nan")
    cosine = np.clip(float(first @ second) / (first_norm * second_norm), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def pair_continuity(
    first: WindowResult, second: WindowResult
) -> dict[str, float | int | str]:
    assert first.sample_route_keys is not None
    assert second.sample_route_keys is not None
    low = max(float(first.sample_route_keys.min()), float(second.sample_route_keys.min()))
    high = min(float(first.sample_route_keys.max()), float(second.sample_route_keys.max()))
    base: dict[str, float | int | str] = {
        "side": first.side,
        "first_window_id": first.spec.window_id,
        "second_window_id": second.spec.window_id,
        "first_model": first.selected_model,
        "second_model": second.selected_model,
        "overlap_key_start": low,
        "overlap_key_end": high,
    }
    if high <= low:
        return {
            **base,
            "overlap_sample_count": 0,
            "mean_gap_m": float("nan"),
            "p95_gap_m": float("nan"),
            "max_gap_m": float("nan"),
            "midpoint_tangent_angle_deg": float("nan"),
        }
    query = np.linspace(low, high, 60)
    first_curve = interpolate_keyed_curve(
        first.sample_route_keys, first.curve_common_xz, query
    )
    second_curve = interpolate_keyed_curve(
        second.sample_route_keys, second.curve_common_xz, query
    )
    gaps = np.linalg.norm(first_curve - second_curve, axis=1)
    middle = len(query) // 2
    first_tangent = first_curve[min(middle + 1, len(query) - 1)] - first_curve[
        max(middle - 1, 0)
    ]
    second_tangent = second_curve[min(middle + 1, len(query) - 1)] - second_curve[
        max(middle - 1, 0)
    ]
    return {
        **base,
        "overlap_sample_count": len(query),
        "mean_gap_m": float(np.mean(gaps)),
        "p95_gap_m": float(np.percentile(gaps, 95)),
        "max_gap_m": float(np.max(gaps)),
        "midpoint_tangent_angle_deg": vector_angle_deg(first_tangent, second_tangent),
    }


def blend_side(
    results: Sequence[WindowResult], bin_width: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = [
        result
        for result in results
        if result.sample_route_keys is not None and len(result.sample_route_keys)
    ]
    if not valid:
        return (
            np.empty(0, dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            np.empty(0, dtype=np.int64),
        )
    keys = np.concatenate([result.sample_route_keys for result in valid])
    points = np.vstack([result.curve_common_xz for result in valid])
    return collapse_keyed_samples(keys, points, bin_width)


def split_blended_support(
    keys: np.ndarray,
    points_xz: np.ndarray,
    supports: np.ndarray,
    maximum_order_gap_frames: float,
    maximum_node_gap_m: float,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
    """Break a blended curve wherever the internal support has a real gap."""

    if not len(keys):
        return []
    order_gaps = np.diff(keys)
    spatial_gaps = np.linalg.norm(np.diff(points_xz, axis=0), axis=1)
    break_mask = (order_gaps > maximum_order_gap_frames) | (
        spatial_gaps > maximum_node_gap_m
    )
    break_indices = (np.flatnonzero(break_mask) + 1).tolist()
    boundaries = [0, *break_indices, len(keys)]
    pieces = []
    for piece_index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if end <= start:
            continue
        if piece_index == 0:
            reason = "window_continuity_segment_start"
        else:
            boundary = start - 1
            reasons = []
            if order_gaps[boundary] > maximum_order_gap_frames:
                reasons.append("internal_order_support_gap")
            if spatial_gaps[boundary] > maximum_node_gap_m:
                reasons.append("internal_spatial_node_gap")
            reason = ";".join(reasons)
        pieces.append((keys[start:end], points_xz[start:end], supports[start:end], reason))
    return pieces


def split_continuity_segments(
    results: Sequence[WindowResult],
    p95_gate_m: float,
    angle_gate_deg: float,
) -> tuple[list[list[WindowResult]], list[dict[str, object]]]:
    """Split rather than visually connect unsupported or inconsistent gaps."""

    ordered = sorted(results, key=lambda result: result.spec.window_id)
    if not ordered:
        return [], []
    segments: list[list[WindowResult]] = [[ordered[0]]]
    ordered[0].segment_id = 0
    rows: list[dict[str, object]] = []
    for first, second in zip(ordered[:-1], ordered[1:]):
        diagnostic = pair_continuity(first, second)
        reasons: list[str] = []
        if second.spec.window_id != first.spec.window_id + 1:
            reasons.append("nonconsecutive_window_ids")
        if int(diagnostic["overlap_sample_count"]) == 0:
            reasons.append("no_supported_overlap")
        p95 = float(diagnostic["p95_gap_m"])
        angle = float(diagnostic["midpoint_tangent_angle_deg"])
        if not np.isfinite(p95) or p95 > p95_gate_m:
            reasons.append("overlap_p95_gap_failed")
        if not np.isfinite(angle) or angle > angle_gate_deg:
            reasons.append("overlap_tangent_angle_failed")
        passes = not reasons
        if passes:
            second.segment_id = first.segment_id
            segments[-1].append(second)
        else:
            second.segment_id = first.segment_id + 1
            segments.append([second])
        rows.append(
            {
                **diagnostic,
                "p95_gate_m": p95_gate_m,
                "angle_gate_deg": angle_gate_deg,
                "passes_continuity_gate": passes,
                "break_reason": ";".join(reasons),
                "first_segment_id": first.segment_id,
                "second_segment_id": second.segment_id,
            }
        )
    return segments, rows


def lane_width_diagnostics(
    left_keys: np.ndarray,
    left_xz: np.ndarray,
    right_keys: np.ndarray,
    right_xz: np.ndarray,
) -> dict[str, float | int]:
    if len(left_keys) < 2 or len(right_keys) < 2:
        return {"count": 0}
    low = max(float(left_keys.min()), float(right_keys.min()))
    high = min(float(left_keys.max()), float(right_keys.max()))
    if high <= low:
        return {"count": 0}
    query = np.linspace(low, high, 300)
    left = interpolate_keyed_curve(left_keys, left_xz, query)
    right = interpolate_keyed_curve(right_keys, right_xz, query)
    widths = np.linalg.norm(left - right, axis=1)
    return {
        "count": int(len(widths)),
        "median_m": float(np.median(widths)),
        "p05_m": float(np.percentile(widths, 5)),
        "p95_m": float(np.percentile(widths, 95)),
        "minimum_m": float(np.min(widths)),
        "maximum_m": float(np.max(widths)),
    }


def node_gap_diagnostics(points_xz: np.ndarray) -> dict[str, float | int]:
    if len(points_xz) < 2:
        return {"count": 0}
    gaps = np.linalg.norm(np.diff(points_xz, axis=0), axis=1)
    return {
        "count": int(len(gaps)),
        "median_m": float(np.median(gaps)),
        "p95_m": float(np.percentile(gaps, 95)),
        "maximum_m": float(np.max(gaps)),
    }


def plot_results(
    path: Path,
    route_xz: np.ndarray,
    window_results: Sequence[WindowResult],
    blended_segments: dict[str, list[BlendedSegment]],
    window_rows: Sequence[dict[str, object]],
    comparison_rows: Sequence[dict[str, object]],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    ax = axes[0]
    ax.plot(route_xz[:, 0], route_xz[:, 1], color="#222222", lw=1.3, label="camera path")
    for result in window_results:
        colour = MODEL_COLOURS[result.selected_model]
        ax.plot(
            result.curve_common_xz[:, 0],
            result.curve_common_xz[:, 1],
            color=colour,
            alpha=0.28,
            lw=1.0,
        )
    for side in SIDES:
        first_label = True
        for segment in blended_segments[side]:
            points = segment.points_xz
            if not len(points):
                continue
            ax.plot(
                points[:, 0],
                points[:, 1],
                color=SIDE_COLOURS[side],
                lw=2.6,
                label=f"{side} blended lane" if first_label else None,
            )
            first_label = False
    ax.set_title("Pose-aligned window curves and blended lanes")
    ax.set_xlabel("X right in common reference [m]")
    ax.set_ylabel("Z forward in common reference [m]")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    class_colour = {"straight": "#4a5568", "transition": "#805ad5", "curve": "#dd6b20"}
    for row in window_rows:
        ax.barh(
            int(row["window_id"]),
            int(row["end_frame"]) - int(row["start_frame"]) + 1,
            left=int(row["start_frame"]),
            color=class_colour[str(row["classification"])],
            alpha=0.8,
        )
        ax.text(
            int(row["start_frame"]) + 0.5,
            int(row["window_id"]),
            str(row["selected_model_short"]),
            va="center",
            ha="left",
            fontsize=7,
            color="white",
        )
    ax.set_title("Window classification and selected model")
    ax.set_xlabel("frame")
    ax.set_ylabel("window id")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)

    ax = axes[2]
    plotted = False
    for model, marker in (("parametric_polynomial", "o"),):
        rows = [
            row
            for row in comparison_rows
            if row["model"] == model
            and np.isfinite(float(row.get("heldout_rmse_m", float("nan"))))
        ]
        if not rows:
            continue
        x = [int(row["window_id"]) + (-0.08 if row["side"] == "left" else 0.08) for row in rows]
        y = [float(row["heldout_rmse_m"]) for row in rows]
        ax.scatter(
            x,
            y,
            marker=marker,
            s=28,
            alpha=0.75,
            color=MODEL_COLOURS[model],
            label=model.replace("parametric_", ""),
        )
        plotted = True
    ax.set_title("Held-out-frame consistency")
    ax.set_xlabel("window id (left/right offset)")
    ax.set_ylabel("RMSE [m]")
    ax.grid(alpha=0.25)
    if plotted:
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=8)

    figure.suptitle("Final polynomial windows in metric Cartesian X/Z", fontsize=14)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, object]:
    validate_args(args)
    selected_document = load_json(args.selected_lane_points)
    selected_by_id = load_selected_frames(
        selected_document, args.start_frame, args.end_frame
    )
    camera_height = float(
        args.camera_height
        if args.camera_height is not None
        else selected_document.get("camera_height_m", 1.65)
    )
    pitch_deg = float(
        args.pitch_deg
        if args.pitch_deg is not None
        else selected_document.get("pitch_deg", 0.0)
    )

    calibration = parse_projection(args.calib)
    poses_cam0 = load_kitti_poses(args.poses)
    if args.end_frame >= len(poses_cam0):
        raise ValueError(
            f"End frame {args.end_frame} exceeds pose rows {len(poses_cam0)}."
        )
    poses_image = [pose @ calibration["cam0_from_image"] for pose in poses_cam0]
    common_reference = (
        args.reference_frame if args.reference_frame is not None else args.end_frame
    )
    if not args.start_frame <= common_reference <= args.end_frame:
        raise ValueError("reference-frame must fall inside the requested range.")

    frame_ids = list(range(args.start_frame, args.end_frame + 1))
    route_common = camera_positions_xz(
        poses_image, frame_ids, common_reference
    )
    route_curvature = absolute_pose_curvature(
        route_common, args.curvature_smoothing_window
    )
    route_tree = cKDTree(route_common)
    tangents = route_tangents(route_common)
    windows = build_windows(
        args.start_frame, args.end_frame, args.window_length, args.window_stride
    )

    window_results: list[WindowResult] = []
    window_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []

    for spec in windows:
        local_frame_ids = list(range(spec.start_frame, spec.end_frame + 1))
        trajectory_local = camera_positions_xz(
            poses_image, local_frame_ids, spec.reference_frame
        )
        turn = trajectory_turn_statistics(trajectory_local)
        local_start = spec.start_frame - args.start_frame
        local_end = spec.end_frame - args.start_frame + 1
        window_curvature = route_curvature[local_start:local_end]
        classification, curvature_fractions = classify_curvature_window(
            window_curvature,
            args.straight_max_curvature_1_per_m,
            args.curve_min_curvature_1_per_m,
            args.curvature_persistence_frames,
            args.curvature_majority_fraction,
        )
        fixed_model = "parametric_polynomial"
        window_row: dict[str, object] = {
            "window_id": spec.window_id,
            "start_frame": spec.start_frame,
            "end_frame": spec.end_frame,
            "reference_frame": spec.reference_frame,
            "terminal_anchor": spec.terminal_anchor,
            "experiment_section": (
                "unspecified"
                if args.core_end_frame is None
                else (
                    "core"
                    if spec.end_frame <= args.core_end_frame
                    else (
                        "exit_extension"
                        if spec.start_frame > args.core_end_frame
                        else "core_to_exit_boundary"
                    )
                )
            ),
            "classification": classification,
            "classification_value": "persistent absolute pose curvature majority",
            "curvature_min_1_per_m": float(np.min(window_curvature)),
            "curvature_median_1_per_m": float(np.median(window_curvature)),
            "curvature_p90_1_per_m": float(np.percentile(window_curvature, 90)),
            "curvature_max_1_per_m": float(np.max(window_curvature)),
            "curvature_straight_frame_fraction": curvature_fractions["straight"],
            "curvature_curve_frame_fraction": curvature_fractions["curve"],
            "curvature_transition_frame_fraction": curvature_fractions["transition"],
            "position_step_net_heading_change_deg_diagnostic_only": turn[
                "net_heading_change_deg"
            ],
            "trajectory_net_heading_change_deg": turn[
                "trajectory_net_heading_change_deg"
            ],
            "total_absolute_heading_change_deg_diagnostic_only": turn[
                "total_absolute_heading_change_deg"
            ],
            "trajectory_path_length_m": turn["path_length_m"],
            "selected_model": fixed_model,
            "selected_model_short": f"poly{args.polynomial_degree}",
        }
        window_rows.append(window_row)

        for side_index, side in enumerate(SIDES):
            observations = align_window_observations(
                spec,
                side_index,
                selected_by_id,
                poses_image,
                camera_height,
                pitch_deg,
            )
            valid_ids = [frame_id for frame_id, _ in observations]
            missing_run = longest_missing_run(
                spec.start_frame, spec.end_frame, valid_ids
            )
            interpolated_ids = short_internal_gaps(
                spec.start_frame,
                spec.end_frame,
                valid_ids,
                args.maximum_missing_run_frames,
            )
            window_row[f"{side}_valid_frame_count"] = len(observations)
            window_row[f"{side}_coverage_fraction"] = (
                len(observations) / args.window_length
            )
            window_row[f"{side}_longest_missing_run_frames"] = missing_run
            window_row[f"{side}_short_gap_interpolated_frames"] = len(
                interpolated_ids
            )
            for frame_id in interpolated_ids:
                gap_rows.append(
                    {
                        "window_id": spec.window_id,
                        "side": side,
                        "frame_id": frame_id,
                        "method": "window_polynomial_between_bracketing_observations",
                        "source_observations": "bracketed",
                        "maximum_gap_frames": args.maximum_missing_run_frames,
                    }
                )
            coverage_reasons = []
            if len(observations) < args.minimum_valid_frames_per_side:
                coverage_reasons.append("below_minimum_valid_frames")
            if missing_run > args.maximum_missing_run_frames:
                coverage_reasons.append("internal_missing_run_exceeds_gate")
            if coverage_reasons:
                skipped_rows.append(
                    {
                        "window_id": spec.window_id,
                        "start_frame": spec.start_frame,
                        "end_frame": spec.end_frame,
                        "side": side,
                        "reason": ";".join(coverage_reasons),
                        "valid_frame_count": len(observations),
                        "coverage_fraction": len(observations) / args.window_length,
                        "longest_missing_run_frames": missing_run,
                        "minimum_valid_frames_gate": args.minimum_valid_frames_per_side,
                        "maximum_missing_run_gate": args.maximum_missing_run_frames,
                    }
                )
                continue
            try:
                aggregate, supports = aggregate_equal_frame_bins_xz(
                    observations, trajectory_local, args.bin_size_m
                )
            except ValueError as error:
                skipped_rows.append(
                    {
                        "window_id": spec.window_id,
                        "start_frame": spec.start_frame,
                        "end_frame": spec.end_frame,
                        "side": side,
                        "reason": str(error),
                        "valid_frame_count": len(observations),
                    }
                )
                continue

            candidate_fits: dict[str, ParametricFit] = {}
            heldout_by_model: dict[str, dict[str, float | int]] = {}
            side_comparison_rows: list[dict[str, object]] = []
            for model in ("parametric_polynomial",):
                try:
                    candidate = fit_model(model, aggregate, supports, args)
                    candidate_fits[model] = candidate
                    training = residual_summary(candidate.residuals_m)
                    heldout = cross_validate_model(
                        model, observations, trajectory_local, args
                    )
                    heldout_by_model[model] = heldout
                    comparison_row: dict[str, object] = {
                            "window_id": spec.window_id,
                            "start_frame": spec.start_frame,
                            "end_frame": spec.end_frame,
                            "classification": classification,
                            "side": side,
                            "model": model,
                            "selected_for_output": False,
                            "valid_frame_count": len(observations),
                            "aggregate_point_count": len(aggregate),
                            **{f"training_{key}": value for key, value in training.items()},
                            **{
                                f"heldout_{key}": value
                                for key, value in heldout.items()
                            },
                        }
                    comparison_rows.append(comparison_row)
                    side_comparison_rows.append(comparison_row)
                except (ValueError, TypeError, RuntimeError) as error:
                    skipped_rows.append(
                        {
                            "window_id": spec.window_id,
                            "start_frame": spec.start_frame,
                            "end_frame": spec.end_frame,
                            "side": side,
                            "model": model,
                            "reason": f"model_fit_failed: {error}",
                            "valid_frame_count": len(observations),
                        }
                    )
            if not candidate_fits:
                skipped_rows.append(
                    {
                        "window_id": spec.window_id,
                        "side": side,
                        "reason": "both_candidate_models_failed",
                        "valid_frame_count": len(observations),
                    }
                )
                continue
            selected_model = fixed_model
            selection_reason = (
                "fixed polynomial-only policy; window classification is diagnostic"
            )
            window_row[f"{side}_selected_model"] = selected_model
            window_row[f"{side}_selection_reason"] = selection_reason
            for row in side_comparison_rows:
                row["selected_for_output"] = row["model"] == selected_model
                row["selection_reason"] = selection_reason

            fit = candidate_fits.get(selected_model)
            if fit is None:
                skipped_rows.append(
                    {
                        "window_id": spec.window_id,
                        "side": side,
                        "model": selected_model,
                        "reason": "registered_model_failed_even_though_comparison_model_was_available",
                        "valid_frame_count": len(observations),
                    }
                )
                continue
            curve_common = transform_xz_between_references(
                fit.curve_xz,
                spec.reference_frame,
                common_reference,
                poses_image,
                camera_height,
                pitch_deg,
            )
            aggregate_common = transform_xz_between_references(
                fit.aggregate_xz,
                spec.reference_frame,
                common_reference,
                poses_image,
                camera_height,
                pitch_deg,
            )
            result = WindowResult(
                spec=spec,
                classification=classification,
                net_heading_change_deg=turn["net_heading_change_deg"],
                total_absolute_heading_change_deg=turn[
                    "total_absolute_heading_change_deg"
                ],
                trajectory_path_length_m=turn["path_length_m"],
                side=side,
                selected_model=selected_model,
                valid_frame_ids=[frame_id for frame_id, _ in observations],
                fit_local=fit,
                curve_common_xz=curve_common,
                aggregate_common_xz=aggregate_common,
                model_selection_reason=selection_reason,
            )
            start_index = spec.start_frame - args.start_frame
            end_index = spec.end_frame - args.start_frame
            keys, keep = assign_route_keys(
                curve_common,
                route_common,
                route_tree,
                tangents,
                start_index,
                end_index,
                args.blend_maximum_route_distance_m,
            )
            result.curve_common_xz = curve_common[keep]
            result.sample_route_keys = keys[keep]
            if len(result.curve_common_xz) < 2:
                skipped_rows.append(
                    {
                        "window_id": spec.window_id,
                        "side": side,
                        "model": selected_model,
                        "reason": "too_few_samples_after_route_support_trim",
                        "valid_frame_count": len(observations),
                    }
                )
                continue
            window_results.append(result)

        left_model = window_row.get("left_selected_model")
        right_model = window_row.get("right_selected_model")
        if left_model is not None and left_model == right_model:
            window_row["selected_model"] = left_model
            window_row["selected_model_short"] = (
                f"poly{args.polynomial_degree}"
                if left_model == "parametric_polynomial"
                else "polynomial"
            )
        elif left_model is not None or right_model is not None:
            window_row["selected_model"] = "side_specific"
            window_row["selected_model_short"] = "L/R differ"

    by_side = {
        side: sorted(
            [result for result in window_results if result.side == side],
            key=lambda result: result.spec.window_id,
        )
        for side in SIDES
    }
    continuity_rows: list[dict[str, object]] = []
    segments_by_side: dict[str, list[list[WindowResult]]] = {}
    for side in SIDES:
        segments, rows = split_continuity_segments(
            by_side[side],
            args.continuity_p95_gate_m,
            args.continuity_angle_gate_deg,
        )
        segments_by_side[side] = segments
        continuity_rows.extend(rows)
    blended_segments: dict[str, list[BlendedSegment]] = {
        side: [] for side in SIDES
    }
    for side in SIDES:
        next_output_segment_id = 0
        for segment in segments_by_side[side]:
            keys, points, supports = blend_side(
                segment, args.blend_key_bin_width
            )
            pieces = split_blended_support(
                keys,
                points,
                supports,
                args.maximum_internal_order_gap_frames,
                args.maximum_internal_node_gap_m,
            )
            for piece_keys, piece_points, piece_supports, start_reason in pieces:
                blended_segments[side].append(
                    BlendedSegment(
                        side=side,
                        segment_id=next_output_segment_id,
                        parent_window_segment_id=segment[0].segment_id,
                        ordering_keys=piece_keys,
                        points_xz=piece_points,
                        supports=piece_supports,
                        start_reason=start_reason,
                    )
                )
                next_output_segment_id += 1

    sample_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    model_documents: list[dict[str, object]] = []
    for result in window_results:
        assert result.sample_route_keys is not None
        for sample_id, (key, point) in enumerate(
            zip(result.sample_route_keys, result.curve_common_xz)
        ):
            sample_rows.append(
                {
                    "window_id": result.spec.window_id,
                    "start_frame": result.spec.start_frame,
                    "end_frame": result.spec.end_frame,
                    "side": result.side,
                    "segment_id": result.segment_id,
                    "classification": result.classification,
                    "model": result.selected_model,
                    "sample_id": sample_id,
                    "ordering_key": float(key),
                    "x_common_m": float(point[0]),
                    "z_common_m": float(point[1]),
                }
            )
        for point_id, (point, support) in enumerate(
            zip(
                result.aggregate_common_xz,
                result.fit_local.aggregate_support_frames,
            )
        ):
            aggregate_rows.append(
                {
                    "window_id": result.spec.window_id,
                    "side": result.side,
                    "segment_id": result.segment_id,
                    "point_id": point_id,
                    "supporting_frame_count": int(support),
                    "x_common_m": float(point[0]),
                    "z_common_m": float(point[1]),
                }
            )
        model_documents.append(
            {
                "window_id": result.spec.window_id,
                "frames": [result.spec.start_frame, result.spec.end_frame],
                "window_reference_frame": result.spec.reference_frame,
                "side": result.side,
                "segment_id": result.segment_id,
                "classification": result.classification,
                "selected_model": result.selected_model,
                "model_selection_reason": result.model_selection_reason,
                "degree": result.fit_local.degree,
                "valid_frame_ids": result.valid_frame_ids,
                "dimensionless_parameter_role": "sample ordering only",
                "coefficients": result.fit_local.coefficients,
                "training_residuals_m": residual_summary(
                    result.fit_local.residuals_m
                ),
            }
        )

    blended_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    blended_diagnostics: dict[str, object] = {}
    for side in SIDES:
        side_diagnostics = []
        for output_segment in blended_segments[side]:
            segment_id = output_segment.segment_id
            keys = output_segment.ordering_keys
            points = output_segment.points_xz
            supports = output_segment.supports
            source_segment = next(
                segment
                for segment in segments_by_side[side]
                if segment[0].segment_id
                == output_segment.parent_window_segment_id
            )
            segment_rows.append(
                {
                    "side": side,
                    "segment_id": segment_id,
                    "parent_window_segment_id": output_segment.parent_window_segment_id,
                    "segment_start_reason": output_segment.start_reason,
                    "first_window_id": source_segment[0].spec.window_id,
                    "last_window_id": source_segment[-1].spec.window_id,
                    "first_frame": source_segment[0].spec.start_frame,
                    "last_frame": source_segment[-1].spec.end_frame,
                    "window_count": len(source_segment),
                    "blended_node_count": len(points),
                    "ordering_key_start": (
                        float(keys.min()) if len(keys) else float("nan")
                    ),
                    "ordering_key_end": (
                        float(keys.max()) if len(keys) else float("nan")
                    ),
                }
            )
            for node_id, (key, point, support) in enumerate(
                zip(keys, points, supports)
            ):
                blended_rows.append(
                    {
                        "side": side,
                        "segment_id": segment_id,
                        "node_id": node_id,
                        "ordering_key": float(key),
                        "contributing_sample_count": int(support),
                        "x_common_m": float(point[0]),
                        "z_common_m": float(point[1]),
                    }
                )
            side_diagnostics.append(
                {
                    "segment_id": segment_id,
                    "node_count": int(len(points)),
                    "node_gap": node_gap_diagnostics(points),
                }
            )
        blended_diagnostics[side] = side_diagnostics

    width_diagnostics: list[dict[str, object]] = []
    for left_segment in blended_segments["left"]:
        for right_segment in blended_segments["right"]:
            left_id = left_segment.segment_id
            left_keys = left_segment.ordering_keys
            left_points = left_segment.points_xz
            right_id = right_segment.segment_id
            right_keys = right_segment.ordering_keys
            right_points = right_segment.points_xz
            low = max(
                float(left_keys.min()) if len(left_keys) else float("inf"),
                float(right_keys.min()) if len(right_keys) else float("inf"),
            )
            high = min(
                float(left_keys.max()) if len(left_keys) else float("-inf"),
                float(right_keys.max()) if len(right_keys) else float("-inf"),
            )
            if high <= low:
                continue
            width_diagnostics.append(
                {
                    "left_segment_id": left_id,
                    "right_segment_id": right_id,
                    **lane_width_diagnostics(
                        left_keys, left_points, right_keys, right_points
                    ),
                }
            )

    trajectory_rows = [
        {
            "frame_id": frame_id,
            "x_common_m": float(point[0]),
            "z_common_m": float(point[1]),
        }
        for frame_id, point in zip(frame_ids, route_common)
    ]
    write_csv(args.output_dir / "window_plan.csv", window_rows)
    write_csv(args.output_dir / "model_comparison.csv", comparison_rows)
    write_csv(args.output_dir / "short_gap_interpolations.csv", gap_rows)
    write_csv(args.output_dir / "window_continuity.csv", continuity_rows)
    write_csv(args.output_dir / "window_curve_samples.csv", sample_rows)
    write_csv(args.output_dir / "window_aggregate_points.csv", aggregate_rows)
    write_csv(args.output_dir / "blended_lane_nodes.csv", blended_rows)
    write_csv(args.output_dir / "output_segments.csv", segment_rows)
    write_csv(args.output_dir / "trajectory_common_xz.csv", trajectory_rows)
    write_csv(args.output_dir / "skipped_items.csv", skipped_rows)
    write_json(args.output_dir / "window_models.json", model_documents)

    plot_results(
        args.output_dir / "polynomial_windows_xz_overview.png",
        route_common,
        window_results,
        blended_segments,
        window_rows,
        comparison_rows,
    )

    warnings = [
        "The lane observations come from CLRNet and the configured planar IPM; they are not official lane-boundary ground truth.",
        "Upstream left/right selection is preserved, but a stable project-side label does not prove semantic lane identity.",
        "Window classes use persistent absolute pose curvature thresholds chosen for this experiment; they are not KITTI annotations.",
        "Pose curvature describes the camera/vehicle trajectory and does not prove painted lane-boundary identity.",
        "Held-out errors measure consistency with omitted CLRNet/IPM observations, not real-world lane-position accuracy.",
        "The camera height, pitch and flat-road projection assumptions remain part of the measurement model.",
        "Curve parameters are dimensionless ordering variables only; all saved geometry is metric Cartesian X/Z.",
        "Overlap blending combines observed fitted samples; gaps with no supported window are not claimed as measured lane markings.",
        "Only bracketed internal gaps up to the registered limit are reported as polynomial interpolation; no unbounded extrapolation is performed.",
        "A failed continuity gate starts a new segment_id; plotting software must not connect different segments.",
        "A window side below the registered valid-frame coverage or missing-run gate is skipped rather than presented as a continuous fit.",
    ]
    expected_window_side_fits = len(windows) * len(SIDES)
    if len(window_results) == expected_window_side_fits:
        status = "complete"
    elif window_results:
        status = "complete_with_skips"
    else:
        status = "no_fitted_windows"
    result_document: dict[str, object] = {
        "status": status,
        "dataset": f"KITTI Odometry Sequence {args.sequence_id}",
        "requested_frames": [args.start_frame, args.end_frame],
        "experiment_sections": (
            None
            if args.core_end_frame is None
            else {
                "core_frames": [args.start_frame, args.core_end_frame],
                "exit_extension_frames": (
                    [args.core_end_frame + 1, args.end_frame]
                    if args.core_end_frame < args.end_frame
                    else None
                ),
            }
        ),
        "common_reference_frame": common_reference,
        "coordinate_system": (
            "metric Cartesian X/Z in the selected common image-camera reference; "
            "X is right and Z is forward"
        ),
        "geometry_policy": (
            "local IPM X/Z -> KITTI pose alignment -> local parametric X(q), Z(q) "
            "polynomial fitting -> common-reference X/Z"
        ),
        "parameter_policy": "q is a dimensionless ordering variable, not a spatial coordinate",
        "observation_ordering_policy": (
            "nearest cumulative progress on the local pose polyline is used only "
            "as an aggregation key; saved and fitted coordinates remain X/Z"
        ),
        "windowing": {
            "length_frames": args.window_length,
            "stride_frames": args.window_stride,
            "window_count": len(windows),
            "terminal_window_added_for_end_coverage": bool(windows[-1].terminal_anchor),
            "minimum_valid_frames_per_side": args.minimum_valid_frames_per_side,
            "maximum_missing_run_frames": args.maximum_missing_run_frames,
        },
        "classification": {
            "measurement": "persistent absolute pose curvature majority inside each window",
            "straight_max_curvature_1_per_m_inclusive": args.straight_max_curvature_1_per_m,
            "curve_min_curvature_1_per_m_inclusive": args.curve_min_curvature_1_per_m,
            "transition_interval_1_per_m": [
                args.straight_max_curvature_1_per_m,
                args.curve_min_curvature_1_per_m,
            ],
            "persistence_frames": args.curvature_persistence_frames,
            "majority_fraction": args.curvature_majority_fraction,
            "heading_change_retained_as_diagnostic_only": True,
        },
        "model_policy": {
            "straight": f"robust parametric polynomial, degree {args.polynomial_degree}",
            "transition": "same robust polynomial; classification is diagnostic only",
            "curve": "same robust parametric polynomial",
            "left_right_are_never_pooled": True,
            "only_one_model_is_fitted": True,
        },
        "counts": {
            "expected_window_side_fits": expected_window_side_fits,
            "window_side_fits": len(window_results),
            "skipped_items": len(skipped_rows),
            "left_output_segments": len(blended_segments["left"]),
            "right_output_segments": len(blended_segments["right"]),
            "left_blended_nodes": sum(
                len(item.points_xz) for item in blended_segments["left"]
            ),
            "right_blended_nodes": sum(
                len(item.points_xz) for item in blended_segments["right"]
            ),
        },
        "continuity": {
            "adjacent_window_rows": len(continuity_rows),
            "p95_gate_m": args.continuity_p95_gate_m,
            "tangent_angle_gate_deg": args.continuity_angle_gate_deg,
            "maximum_internal_order_gap_frames": (
                args.maximum_internal_order_gap_frames
            ),
            "maximum_internal_node_gap_m": args.maximum_internal_node_gap_m,
            "failed_gate_rows": sum(
                not bool(row["passes_continuity_gate"])
                for row in continuity_rows
            ),
            "failed_gates_start_new_output_segments": True,
            "blended_lane_nodes": blended_diagnostics,
            "left_right_separation": width_diagnostics,
        },
        "provenance": {
            "selected_lane_points": str(args.selected_lane_points.resolve()),
            "selected_lane_points_sha256": sha256(args.selected_lane_points),
            "scan_json": (
                str(args.scan_json.resolve()) if args.scan_json is not None else None
            ),
            "scan_json_sha256": (
                sha256(args.scan_json) if args.scan_json is not None else None
            ),
            "poses": str(args.poses.resolve()),
            "poses_sha256": sha256(args.poses),
            "calib": str(args.calib.resolve()),
            "calib_sha256": sha256(args.calib),
        },
        "warnings": warnings,
        "outputs": {
            "overview_png": "polynomial_windows_xz_overview.png",
            "window_plan_csv": "window_plan.csv",
            "model_comparison_csv": "model_comparison.csv",
            "short_gap_interpolations_csv": "short_gap_interpolations.csv",
            "window_continuity_csv": "window_continuity.csv",
            "window_curve_samples_csv": "window_curve_samples.csv",
            "window_aggregate_points_csv": "window_aggregate_points.csv",
            "blended_lane_nodes_csv": "blended_lane_nodes.csv",
            "output_segments_csv": "output_segments.csv",
            "trajectory_csv": "trajectory_common_xz.csv",
            "window_models_json": "window_models.json",
            "skipped_items_csv": "skipped_items.csv",
        },
    }
    write_json(args.output_dir / "RESULT.json", result_document)
    write_json(
        args.output_dir / "STATUS.json",
        {
            "status": result_document["status"],
            "output_dir": str(args.output_dir.resolve()),
            "previous_outputs_modified": False,
            "requested_frames": [args.start_frame, args.end_frame],
            "window_count": len(windows),
            "window_side_fits": len(window_results),
            "warnings": warnings,
        },
    )
    (args.output_dir / "README_RESULT.md").write_text(
        "\n".join(
            [
                "# Final polynomial-window X/Z result",
                "",
                f"- Dataset: KITTI Odometry Sequence {args.sequence_id}",
                f"- Frames: {args.start_frame}-{args.end_frame}",
                *(
                    []
                    if args.core_end_frame is None
                    else (
                        [f"- Core frames: {args.start_frame}-{args.core_end_frame}"]
                        + (
                            [f"- Exit extension: {args.core_end_frame + 1}-{args.end_frame}"]
                            if args.core_end_frame < args.end_frame
                            else []
                        )
                    )
                ),
                f"- Common reference frame: {common_reference}",
                f"- Windows: {len(windows)} ({args.window_length} frames, stride {args.window_stride})",
                "- Straight windows: robust parametric polynomial in X/Z.",
                "- Every window uses the same robust parametric polynomial in X/Z.",
                "- Straight/transition/curve labels are diagnostics, not model switches.",
                "- Left and right observations are processed independently.",
                "- Failed overlap-continuity gates create separate output segments; gaps are not connected for display.",
                "- Model-comparison metrics are held-out detection consistency, not official accuracy.",
                "",
                "Read `RESULT.json` before using the figures or CSV files.",
            ]
        ),
        encoding="utf-8",
    )
    return result_document


def main() -> None:
    args = parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_dir": str(args.output_dir.resolve()),
                "requested_frames": result["requested_frames"],
                "window_count": result["windowing"]["window_count"],
                "window_side_fits": result["counts"]["window_side_fits"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
