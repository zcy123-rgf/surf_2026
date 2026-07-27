"""Evaluate denoising alternatives on the saved KITTI 00 first-five CLRNet points.

This is an experiment runner, not a replacement for the production pipeline.
It reconstructs metric lane points from the audited CLRNet output, applies the
same point IPM and KITTI pose alignment as ``run_full_point_pipeline.py``, and
writes every result to a new directory.

The manual annotations bundled with the result package are pseudo-labels.  Their
agreement scores are therefore reported as method agreement, never as accuracy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from scipy.optimize import least_squares
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


FRAME_COLORS_BGR = [
    (180, 119, 31),
    (14, 127, 255),
    (44, 160, 44),
    (40, 39, 214),
    (189, 103, 148),
]
FRAME_COLORS_RGB = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
FUSION_RANGE = ((-15.0, 15.0), (-10.0, 50.0))
EVALUATION_RANGE = ((-10.0, 10.0), (3.0, 50.0))
BEV_SIZE = 800
WEIGHTS = np.asarray([0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float64)
SEED = 20260717


def parse_args() -> argparse.Namespace:
    package = ROOT / "input_data" / "kitti00_first5"
    parser = argparse.ArgumentParser(
        description="Compare legacy and corrected point-denoising methods."
    )
    parser.add_argument(
        "--clrnet-json",
        type=Path,
        default=package / "clrnet_lanes.json",
    )
    parser.add_argument(
        "--manual-json",
        type=Path,
        default=package / "manual_annotations.json",
    )
    parser.add_argument(
        "--calib", type=Path, default=package / "calib.txt"
    )
    parser.add_argument(
        "--poses",
        type=Path,
        default=package / "poses_00_first5.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "workstation_outputs"
        / "kitti00_first5_denoise_comparison",
    )
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
            f"Output is not empty: {path}. Choose a new output directory."
        )
    path.mkdir(parents=True, exist_ok=True)


def imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"Cannot encode {path}")
    encoded.tofile(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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
    translation = np.linalg.solve(intrinsic, projection[:, 3])
    image_from_cam0 = np.eye(4, dtype=np.float64)
    image_from_cam0[:3, 3] = translation
    return {
        "projection": projection,
        "K": intrinsic,
        "cam0_from_image": np.linalg.inv(image_from_cam0),
    }


def bottom_x(lane: np.ndarray) -> float:
    threshold = np.quantile(lane[:, 1], 0.85)
    return float(np.median(lane[lane[:, 1] >= threshold, 0]))


def select_outer_two(lanes: list[np.ndarray]) -> list[np.ndarray]:
    ordered = sorted(lanes, key=bottom_x)
    if len(ordered) < 2:
        raise ValueError("Each frame must contain at least two CLRNet lanes.")
    return [ordered[0], ordered[-1]]


def filter_points(
    points: np.ndarray,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    valid = (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= x_range[0])
        & (points[:, 0] <= x_range[1])
        & (points[:, 1] >= z_range[0])
        & (points[:, 1] <= z_range[1])
    )
    return points[valid]


def resample_polyline(points: np.ndarray, count: int = 64) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(distances)])
    samples = np.linspace(0.0, cumulative[-1], count)
    return np.column_stack(
        [np.interp(samples, cumulative, points[:, axis]) for axis in range(2)]
    )


def load_image_lanes(path: Path) -> list[list[np.ndarray]]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if "items" in record:
        items = record["items"]
        lane_key = "lanes_image_xy_px"
    elif (
        isinstance(record.get("detector"), dict)
        and "frames" in record["detector"]
    ):
        items = record["detector"]["frames"]
        lane_key = "candidate_lanes_image_xy_px"
    else:
        raise ValueError(
            "Unsupported CLRNet JSON schema. Expected either an 'items' list "
            "or run_full_point_pipeline.py output with detector.frames."
        )

    output = []
    for item in items:
        selected = item.get("selected")
        if selected:
            lanes = [
                np.asarray(lane["image_xy_px"], dtype=np.float64).reshape(-1, 2)
                for lane in selected
            ]
        else:
            lanes = [
                np.asarray(lane, dtype=np.float64).reshape(-1, 2)
                for lane in item[lane_key]
            ]
            lanes = select_outer_two(lanes)
        if len(lanes) != 2:
            raise ValueError(
                f"Frame {item.get('frame_id', len(output))} did not provide "
                "exactly two selected lane polylines."
            )
        output.append(lanes)
    return output


def load_manual_lanes(path: Path) -> list[list[np.ndarray]]:
    record = json.loads(path.read_text(encoding="utf-8"))
    output = []
    for item in record["frames_xy"]:
        output.append(
            [
                resample_polyline(np.asarray(item["left"], dtype=np.float64)),
                resample_polyline(np.asarray(item["right"], dtype=np.float64)),
            ]
        )
    return output


def align_image_lanes(
    image_lanes: list[list[np.ndarray]],
    intrinsic: np.ndarray,
    poses: list[np.ndarray],
) -> tuple[list[list[np.ndarray]], np.ndarray, list[dict[str, object]]]:
    reference_pose = poses[4]
    frames: list[list[np.ndarray]] = []
    points: list[list[float]] = []
    ranges: list[dict[str, object]] = []
    for frame_id, lanes in enumerate(image_lanes):
        frame_lanes = []
        for side, lane in zip(("left", "right"), lanes):
            ground = image_to_ground_ipm(
                lane,
                intrinsic,
                camera_height=1.65,
                pitch_deg=0.0,
                z_range=(3.0, 50.0),
            )
            aligned = transform_lane_points_by_pose(
                ground,
                poses[frame_id],
                reference_pose,
                camera_height=1.65,
                pitch_deg=0.0,
            )
            aligned = filter_points(aligned, *FUSION_RANGE)
            start = len(points)
            points.extend(aligned.tolist())
            ranges.append(
                {
                    "frame_id": frame_id,
                    "side": side,
                    "start": start,
                    "end": len(points),
                }
            )
            frame_lanes.append(aligned)
        frames.append(frame_lanes)
    return frames, np.asarray(points, dtype=np.float64), ranges


def simple_cluster_by_x(points: np.ndarray) -> dict[str, np.ndarray]:
    x = points[:, 0]
    left_center = (
        float(np.percentile(x[x < 0], 75)) if np.any(x < 0) else -1.5
    )
    right_center = (
        float(np.percentile(x[x > 0], 25)) if np.any(x > 0) else 1.5
    )
    left_window = 0.8
    right_window = 0.8
    left = np.abs(x - left_center) < left_window
    right = np.abs(x - right_center) < right_window
    if np.count_nonzero(left) < 20:
        left = np.abs(x - left_center) < 1.2
    if np.count_nonzero(right) < 20:
        right = np.abs(x - right_center) < 1.2
    return {"left": points[left], "right": points[right]}


def legacy_ransac_line(
    points: np.ndarray,
    threshold: float,
    rng: np.random.Generator,
    iterations: int = 100,
) -> np.ndarray:
    if len(points) < 5:
        return points
    best = np.empty((0, 2), dtype=np.float64)
    for _ in range(iterations):
        first, second = points[rng.choice(len(points), 2, replace=False)]
        if abs(second[0] - first[0]) < 1e-6:
            continue
        slope = (second[1] - first[1]) / (second[0] - first[0])
        intercept = first[1] - slope * first[0]
        residual = np.abs(points[:, 1] - (slope * points[:, 0] + intercept))
        residual /= np.sqrt(slope**2 + 1.0)
        candidate = points[residual < threshold]
        if len(candidate) > len(best):
            best = candidate
    return best if len(best) >= 5 else points


def membership(points: np.ndarray, selected: np.ndarray) -> np.ndarray:
    if not len(selected):
        return np.zeros(len(points), dtype=bool)
    keys = {tuple(row) for row in np.round(selected, 9)}
    return np.asarray(
        [tuple(row) in keys for row in np.round(points, 9)], dtype=bool
    )


def legacy_mask(points: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    clustered = simple_cluster_by_x(points)
    rng = np.random.default_rng(SEED)
    arrays = [
        legacy_ransac_line(clustered["left"], 0.3, rng),
        legacy_ransac_line(clustered["right"], 0.3, rng),
    ]
    kept = np.unique(np.round(np.vstack(arrays), 9), axis=0)
    mask = membership(points, kept)
    return mask, {
        "description": "fixed X windows, then legacy Z=aX+b line RANSAC",
        "threshold_m": 0.3,
        "iterations": 100,
    }


def polynomial_design(z: np.ndarray, degree: int) -> np.ndarray:
    return np.vander(z, N=degree + 1, increasing=True)


def polynomial_ransac_mask(
    points: np.ndarray,
    ranges: list[dict[str, object]],
    degree: int,
    threshold: float,
    iterations: int = 500,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit X=f(Z) separately to the already identified left/right lanes."""

    final = np.zeros(len(points), dtype=bool)
    details: dict[str, object] = {}
    rng = np.random.default_rng(SEED + degree * 1000 + int(threshold * 1000))
    for side in ("left", "right"):
        indexes = np.concatenate(
            [
                np.arange(int(item["start"]), int(item["end"]))
                for item in ranges
                if item["side"] == side
            ]
        )
        subset = points[indexes]
        sample_size = degree + 1
        best_mask = np.zeros(len(subset), dtype=bool)
        best_error = float("inf")
        for _ in range(iterations):
            sample = subset[rng.choice(len(subset), sample_size, replace=False)]
            if np.ptp(sample[:, 1]) < 1.0:
                continue
            coefficients = np.linalg.lstsq(
                polynomial_design(sample[:, 1], degree),
                sample[:, 0],
                rcond=None,
            )[0]
            residual = np.abs(
                subset[:, 0]
                - polynomial_design(subset[:, 1], degree) @ coefficients
            )
            candidate = residual <= threshold
            error = float(np.median(residual[candidate])) if np.any(candidate) else float("inf")
            if np.count_nonzero(candidate) > np.count_nonzero(best_mask) or (
                np.count_nonzero(candidate) == np.count_nonzero(best_mask)
                and error < best_error
            ):
                best_mask = candidate
                best_error = error
        if np.count_nonzero(best_mask) >= sample_size + 2:
            coefficients = np.linalg.lstsq(
                polynomial_design(subset[best_mask, 1], degree),
                subset[best_mask, 0],
                rcond=None,
            )[0]
            residual = np.abs(
                subset[:, 0]
                - polynomial_design(subset[:, 1], degree) @ coefficients
            )
            best_mask = residual <= threshold
        final[indexes] = best_mask
        details[side] = {
            "kept": int(np.count_nonzero(best_mask)),
            "total": int(len(subset)),
            "coefficients_low_to_high": coefficients.tolist(),
        }
    details.update(
        {
            "description": f"side-aware X=f(Z), degree {degree}, no fixed-X window",
            "threshold_m": threshold,
            "iterations": iterations,
        }
    )
    return final, details


def robust_huber_mask(
    points: np.ndarray,
    ranges: list[dict[str, object]],
    threshold_floor: float = 0.3,
) -> tuple[np.ndarray, dict[str, object]]:
    final = np.zeros(len(points), dtype=bool)
    details: dict[str, object] = {
        "description": "side-aware quadratic X=f(Z), Huber robust least squares"
    }
    for side in ("left", "right"):
        indexes = np.concatenate(
            [
                np.arange(int(item["start"]), int(item["end"]))
                for item in ranges
                if item["side"] == side
            ]
        )
        subset = points[indexes]
        initial = np.linalg.lstsq(
            polynomial_design(subset[:, 1], 2), subset[:, 0], rcond=None
        )[0]
        result = least_squares(
            lambda coefficients: (
                polynomial_design(subset[:, 1], 2) @ coefficients - subset[:, 0]
            ),
            initial,
            loss="huber",
            f_scale=threshold_floor,
        )
        residual = np.abs(
            subset[:, 0] - polynomial_design(subset[:, 1], 2) @ result.x
        )
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        robust_sigma = 1.4826 * mad
        threshold = min(0.8, max(threshold_floor, median + 3.0 * robust_sigma))
        keep = residual <= threshold
        final[indexes] = keep
        details[side] = {
            "kept": int(np.count_nonzero(keep)),
            "total": int(len(subset)),
            "threshold_m": threshold,
            "median_abs_residual_m": median,
            "robust_sigma_m": robust_sigma,
            "coefficients_low_to_high": result.x.tolist(),
        }
    return final, details


def interpolate_lane(lane: np.ndarray, z: float) -> float | None:
    order = np.argsort(lane[:, 1])
    sorted_lane = lane[order]
    unique_z, unique_index = np.unique(sorted_lane[:, 1], return_index=True)
    unique_x = sorted_lane[unique_index, 0]
    if len(unique_z) < 2 or z < unique_z[0] or z > unique_z[-1]:
        return None
    return float(np.interp(z, unique_z, unique_x))


def temporal_consensus_mask(
    points: np.ndarray,
    ranges: list[dict[str, object]],
    base_threshold: float = 0.3,
) -> tuple[np.ndarray, dict[str, object]]:
    """Reject only points contradicted by at least three pose-aligned frames."""

    final = np.ones(len(points), dtype=bool)
    residuals: list[float] = []
    thresholds: list[float] = []
    support_counts: list[int] = []
    by_side = {
        side: [
            points[int(item["start"]) : int(item["end"])]
            for item in ranges
            if item["side"] == side
        ]
        for side in ("left", "right")
    }
    for item in ranges:
        start, end = int(item["start"]), int(item["end"])
        for offset, point in enumerate(points[start:end]):
            predictions = [
                prediction
                for lane in by_side[str(item["side"])]
                if (prediction := interpolate_lane(lane, float(point[1]))) is not None
            ]
            support_counts.append(len(predictions))
            if len(predictions) < 3:
                continue
            predictions_array = np.asarray(predictions)
            center = float(np.median(predictions_array))
            mad = float(np.median(np.abs(predictions_array - center)))
            threshold = min(0.8, max(base_threshold, 3.0 * 1.4826 * mad))
            residual = abs(float(point[0]) - center)
            residuals.append(residual)
            thresholds.append(threshold)
            final[start + offset] = residual <= threshold
    return final, {
        "description": (
            "pose-aligned cross-frame median X(Z); unsupported boundary points are kept"
        ),
        "base_threshold_m": base_threshold,
        "minimum_supporting_frames": 3,
        "adaptive_threshold_cap_m": 0.8,
        "evaluated_points": len(residuals),
        "median_residual_m": float(np.median(residuals)) if residuals else None,
        "median_threshold_m": float(np.median(thresholds)) if thresholds else None,
        "minimum_support_seen": min(support_counts) if support_counts else 0,
    }


def split_runs(points: np.ndarray, keep: np.ndarray) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    start: int | None = None
    for index, selected in enumerate(keep):
        if selected and start is None:
            start = index
        if start is not None and (not selected or index == len(keep) - 1):
            end = index if not selected else index + 1
            if end - start >= 2:
                output.append(points[start:end])
            start = None
    return output


def frames_from_mask(
    points: np.ndarray,
    ranges: list[dict[str, object]],
    mask: np.ndarray,
) -> list[list[np.ndarray]]:
    frames: list[list[np.ndarray]] = [[] for _ in range(5)]
    for item in ranges:
        start, end = int(item["start"]), int(item["end"])
        lane = points[start:end]
        frames[int(item["frame_id"])].extend(split_runs(lane, mask[start:end]))
    return frames


def metric_to_pixels(
    points: np.ndarray,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
) -> np.ndarray:
    points = filter_points(points, x_range, z_range)
    if not len(points):
        return np.empty((0, 2), dtype=np.int32)
    u = (points[:, 0] - x_range[0]) / (x_range[1] - x_range[0]) * (BEV_SIZE - 1)
    v = (z_range[1] - points[:, 1]) / (z_range[1] - z_range[0]) * (BEV_SIZE - 1)
    return np.rint(np.column_stack([u, v])).astype(np.int32)


def lane_mask(
    frames: list[list[np.ndarray]],
    x_range: tuple[float, float],
    z_range: tuple[float, float],
) -> tuple[list[np.ndarray], dict[str, object]]:
    masks = []
    for lanes in frames:
        mask = np.zeros((BEV_SIZE, BEV_SIZE), dtype=np.uint8)
        for lane in lanes:
            pixels = metric_to_pixels(lane, x_range, z_range)
            if len(pixels) >= 2:
                cv2.polylines(mask, [pixels], False, 255, 3, cv2.LINE_AA)
        masks.append(mask)
    accumulation = np.zeros((BEV_SIZE, BEV_SIZE), dtype=np.float32)
    for mask, weight in zip(masks, WEIGHTS):
        accumulation += mask.astype(np.float32) / 255.0 * float(weight)
    score = accumulation / float(np.sum(WEIGHTS))
    intensity = np.rint(np.clip(score, 0.0, 1.0) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(intensity, cv2.COLORMAP_TURBO)
    heatmap[intensity == 0] = 255
    binary = np.where(score >= 0.08, 255, 0).astype(np.uint8)
    hits = np.stack([mask > 0 for mask in masks]).sum(axis=0)
    union = int(np.count_nonzero(hits))
    overlap = int(np.count_nonzero(hits >= 2))
    return masks, {
        "heatmap": heatmap,
        "binary": binary,
        "union_pixels": union,
        "overlap_pixels": overlap,
        "overlap_fraction": float(overlap / union) if union else 0.0,
        "thresholded_pixels": int(np.count_nonzero(binary)),
    }


def draw_points(frames: list[list[np.ndarray]]) -> np.ndarray:
    output = np.full((BEV_SIZE, BEV_SIZE, 3), 255, dtype=np.uint8)
    for frame_id, lanes in enumerate(frames):
        for lane in lanes:
            for point in metric_to_pixels(lane, *FUSION_RANGE):
                cv2.circle(
                    output,
                    tuple(point),
                    3,
                    FRAME_COLORS_BGR[frame_id],
                    -1,
                    cv2.LINE_AA,
                )
    return output


def plot_points(path: Path, frames: list[list[np.ndarray]], title: str) -> None:
    figure, axis = plt.subplots(figsize=(7, 11))
    for frame_id, lanes in enumerate(frames):
        labelled = False
        for lane in lanes:
            if len(lane) < 2:
                continue
            axis.plot(
                lane[:, 0],
                lane[:, 1],
                "o-",
                color=FRAME_COLORS_RGB[frame_id],
                linewidth=1.2,
                markersize=2.2,
                label=f"frame {frame_id}" if not labelled else None,
            )
            labelled = True
    axis.set(
        xlim=FUSION_RANGE[0],
        ylim=FUSION_RANGE[1],
        xlabel="X right in reference frame [m]",
        ylabel="Z forward in reference frame [m]",
        title=title,
    )
    axis.set_aspect("equal")
    axis.grid(alpha=0.3)
    axis.legend(loc="upper right")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def cross_frame_consistency(
    points: np.ndarray,
    ranges: list[dict[str, object]],
    mask: np.ndarray,
) -> dict[str, float | int]:
    deviations: list[float] = []
    supports: list[int] = []
    cells = 0
    for side in ("left", "right"):
        lanes = []
        for item in ranges:
            if item["side"] != side:
                continue
            start, end = int(item["start"]), int(item["end"])
            lane = points[start:end][mask[start:end]]
            if len(lane) >= 2:
                lanes.append(lane)
        for z in np.arange(3.0, 50.0001, 0.5):
            predictions = [
                value
                for lane in lanes
                if (value := interpolate_lane(lane, float(z))) is not None
            ]
            if len(predictions) < 2:
                continue
            values = np.asarray(predictions)
            center = float(np.median(values))
            deviations.extend(np.abs(values - center).tolist())
            supports.append(len(values))
            cells += 1
    return {
        "common_side_z_cells": cells,
        "mean_supporting_frames": float(np.mean(supports)) if supports else 0.0,
        "median_cross_frame_abs_deviation_m": (
            float(np.median(deviations)) if deviations else float("nan")
        ),
        "mean_cross_frame_abs_deviation_m": (
            float(np.mean(deviations)) if deviations else float("nan")
        ),
    }


def pseudo_label_agreement(
    selected_points: np.ndarray,
    manual_points: np.ndarray,
    tolerance_m: float = 0.5,
) -> dict[str, float]:
    selected = filter_points(selected_points, *EVALUATION_RANGE)
    manual = filter_points(manual_points, *EVALUATION_RANGE)
    selected_distance = cKDTree(manual).query(selected, k=1)[0]
    manual_distance = cKDTree(selected).query(manual, k=1)[0]
    selected_near = float(np.mean(selected_distance <= tolerance_m))
    manual_near = float(np.mean(manual_distance <= tolerance_m))
    return {
        "pseudo_precision_within_0p5m": selected_near,
        "pseudo_recall_within_0p5m": manual_near,
        "pseudo_symmetric_agreement_0p5m": (selected_near + manual_near) / 2.0,
        "pseudo_selected_to_manual_median_m": float(np.median(selected_distance)),
        "pseudo_manual_to_selected_median_m": float(np.median(manual_distance)),
    }


def method_metrics(
    name: str,
    points: np.ndarray,
    ranges: list[dict[str, object]],
    mask: np.ndarray,
    manual_points: np.ndarray,
    fusion: dict[str, object],
) -> dict[str, object]:
    selected = points[mask]
    raw_far = np.count_nonzero(points[:, 1] >= 30.0)
    selected_far = np.count_nonzero(selected[:, 1] >= 30.0)
    evaluation_raw = filter_points(points, *EVALUATION_RANGE)
    evaluation_selected = filter_points(selected, *EVALUATION_RANGE)
    metrics: dict[str, object] = {
        "method": name,
        "points": int(len(selected)),
        "point_retention": float(len(selected) / len(points)),
        "evaluation_region_points": int(len(evaluation_selected)),
        "evaluation_region_retention": float(
            len(evaluation_selected) / len(evaluation_raw)
        ),
        "far_points_z_ge_30m": int(selected_far),
        "far_point_retention_z_ge_30m": float(
            selected_far / raw_far if raw_far else 0.0
        ),
        "maximum_z_m": float(np.max(selected[:, 1])),
        "z_95_percentile_m": float(np.percentile(selected[:, 1], 95)),
        "fusion_union_pixels": int(fusion["union_pixels"]),
        "fusion_overlap_pixels": int(fusion["overlap_pixels"]),
        "fusion_overlap_fraction": float(fusion["overlap_fraction"]),
        "fusion_thresholded_pixels": int(fusion["thresholded_pixels"]),
    }
    metrics.update(cross_frame_consistency(points, ranges, mask))
    metrics.update(pseudo_label_agreement(selected, manual_points))
    return metrics


def longitudinal_rows(
    method: str, points: np.ndarray, mask: np.ndarray
) -> list[dict[str, object]]:
    rows = []
    for lower, upper in [
        (-10, 0),
        (0, 3),
        (3, 10),
        (10, 20),
        (20, 30),
        (30, 40),
        (40, 50.0001),
    ]:
        raw = (points[:, 1] >= lower) & (points[:, 1] < upper)
        kept = raw & mask
        rows.append(
            {
                "method": method,
                "z_min_m": lower,
                "z_max_m": 50 if upper > 50 else upper,
                "raw_points": int(np.count_nonzero(raw)),
                "kept_points": int(np.count_nonzero(kept)),
                "retention": float(
                    np.count_nonzero(kept) / np.count_nonzero(raw)
                    if np.count_nonzero(raw)
                    else 0.0
                ),
            }
        )
    return rows


def make_summary_figure(
    path: Path,
    results: dict[str, dict[str, object]],
    selected_names: list[str],
) -> None:
    figure, axes = plt.subplots(
        len(selected_names), 2, figsize=(10, 4.7 * len(selected_names))
    )
    for row, name in enumerate(selected_names):
        result = results[name]
        point_image = cv2.cvtColor(result["point_image"], cv2.COLOR_BGR2RGB)
        heatmap = cv2.cvtColor(result["fusion"]["heatmap"], cv2.COLOR_BGR2RGB)
        axes[row, 0].imshow(point_image)
        axes[row, 1].imshow(heatmap)
        axes[row, 0].set_title(
            f"{name}: {result['metrics']['points']} points, "
            f"far retention {result['metrics']['far_point_retention_z_ge_30m']:.1%}"
        )
        axes[row, 1].set_title(
            f"overlap {result['metrics']['fusion_overlap_fraction']:.1%}, "
            f"coverage {result['metrics']['fusion_thresholded_pixels']} px"
        )
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    required = [args.clrnet_json, args.manual_json, args.calib, args.poses]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))
    require_empty_output(args.output_dir)

    calibration = parse_projection(args.calib)
    poses_cam0 = load_kitti_poses(args.poses)
    poses = [pose @ calibration["cam0_from_image"] for pose in poses_cam0]
    clrnet_image_lanes = load_image_lanes(args.clrnet_json)
    manual_image_lanes = load_manual_lanes(args.manual_json)
    raw_frames, points, ranges = align_image_lanes(
        clrnet_image_lanes, calibration["K"], poses
    )
    _, manual_points, _ = align_image_lanes(
        manual_image_lanes, calibration["K"], poses
    )

    methods: dict[str, tuple[np.ndarray, dict[str, object]]] = {
        "raw_no_denoise": (
            np.ones(len(points), dtype=bool),
            {"description": "no point rejection"},
        ),
        "legacy_fixed_x_ransac": legacy_mask(points),
    }
    for degree, label in ((1, "linear"), (2, "quadratic")):
        for threshold in (0.15, 0.30, 0.50):
            suffix = str(threshold).replace(".", "p")
            methods[f"x_of_z_{label}_t{suffix}"] = polynomial_ransac_mask(
                points, ranges, degree, threshold
            )
    methods["huber_quadratic_adaptive"] = robust_huber_mask(points, ranges)
    methods["temporal_median_consensus"] = temporal_consensus_mask(points, ranges)

    results: dict[str, dict[str, object]] = {}
    metric_rows = []
    distance_rows: list[dict[str, object]] = []
    per_lane_rows: list[dict[str, object]] = []
    for order, (name, (mask, details)) in enumerate(methods.items()):
        frames = frames_from_mask(points, ranges, mask)
        _, fusion = lane_mask(frames, *FUSION_RANGE)
        point_image = draw_points(frames)
        method_dir = args.output_dir / f"{order:02d}_{name}"
        imwrite(method_dir / "points_by_frame.png", point_image)
        imwrite(method_dir / "weighted_heatmap.png", fusion["heatmap"])
        imwrite(method_dir / "weighted_binary.png", fusion["binary"])
        plot_points(method_dir / "metric_plot.png", frames, name)
        metrics = method_metrics(
            name, points, ranges, mask, manual_points, fusion
        )
        metric_rows.append(metrics)
        distance_rows.extend(longitudinal_rows(name, points, mask))
        for item in ranges:
            start, end = int(item["start"]), int(item["end"])
            lane_mask_value = mask[start:end]
            per_lane_rows.append(
                {
                    "method": name,
                    "frame_id": item["frame_id"],
                    "side": item["side"],
                    "raw_points": end - start,
                    "kept_points": int(np.count_nonzero(lane_mask_value)),
                    "retention": float(np.mean(lane_mask_value)),
                    "raw_max_z_m": float(np.max(points[start:end, 1])),
                    "kept_max_z_m": (
                        float(np.max(points[start:end][lane_mask_value, 1]))
                        if np.any(lane_mask_value)
                        else None
                    ),
                }
            )
        results[name] = {
            "mask": mask,
            "frames": frames,
            "fusion": fusion,
            "point_image": point_image,
            "details": details,
            "metrics": metrics,
        }

    eligible = [
        row
        for row in metric_rows
        if row["method"] not in ("raw_no_denoise", "legacy_fixed_x_ransac")
        and row["far_point_retention_z_ge_30m"] >= 0.9
        and row["point_retention"] >= 0.8
    ]
    recommended = min(
        eligible,
        key=lambda row: (
            row["median_cross_frame_abs_deviation_m"],
            -row["point_retention"],
        ),
    )
    conservative_gate_name = str(recommended["method"])
    raw_row = next(row for row in metric_rows if row["method"] == "raw_no_denoise")
    gate_changed_points = recommended["points"] != raw_row["points"]
    recommended_name = conservative_gate_name if gate_changed_points else "raw_no_denoise"

    hard_filter_candidates = [
        row
        for row in metric_rows
        if row["method"]
        not in (
            "raw_no_denoise",
            "legacy_fixed_x_ransac",
            "temporal_median_consensus",
        )
        and row["point_retention"] >= 0.95
        and row["points"] < raw_row["points"]
    ]
    hard_filter = min(
        hard_filter_candidates,
        key=lambda row: (
            -row["common_side_z_cells"],
            row["mean_cross_frame_abs_deviation_m"],
            -row["far_point_retention_z_ge_30m"],
        ),
    )
    hard_filter_name = str(hard_filter["method"])

    audit_dir = args.output_dir / "00_audit"
    write_csv(audit_dir / "method_metrics.csv", metric_rows)
    write_csv(audit_dir / "longitudinal_retention.csv", distance_rows)
    write_csv(audit_dir / "per_frame_lane_retention.csv", per_lane_rows)
    make_summary_figure(
        args.output_dir / "comparison" / "main_comparison.png",
        results,
        ["raw_no_denoise", "legacy_fixed_x_ransac", hard_filter_name],
    )

    serializable_methods = {
        name: {
            "details": result["details"],
            "metrics": result["metrics"],
        }
        for name, result in results.items()
    }
    audit = {
        "status": "complete",
        "scope": (
            "Denoising comparison on saved CLRNet point output; original result "
            "directories were not modified."
        ),
        "inputs": {
            "clrnet_json": str(args.clrnet_json.resolve()),
            "clrnet_json_sha256": sha256(args.clrnet_json),
            "manual_json": str(args.manual_json.resolve()),
            "manual_json_sha256": sha256(args.manual_json),
            "manual_warning": (
                "Manual annotations are pseudo-labels with interpolated occlusions, "
                "not KITTI lane ground truth."
            ),
            "calib": str(args.calib.resolve()),
            "calib_sha256": sha256(args.calib),
            "poses": str(args.poses.resolve()),
            "poses_sha256": sha256(args.poses),
        },
        "geometry": {
            "projection": "ordered image points -> ray/flat-plane metric XZ",
            "camera_height_m": 1.65,
            "pitch_deg": 0.0,
            "pose_formula": "p_ref = inv(T_w_ref) @ T_w_src @ p_src",
            "reference_frame": 4,
            "fusion_range": FUSION_RANGE,
            "evaluation_range": EVALUATION_RANGE,
        },
        "evaluation_notes": [
            "Overlap is multi-frame consistency, not detection accuracy.",
            "Pseudo-label agreement is method agreement, not ground-truth accuracy.",
            "The recommendation requires >=90% far-point retention and >=80% total retention.",
        ],
        "raw_points": int(len(points)),
        "manual_pseudo_label_points": int(len(manual_points)),
        "recommended_method": recommended_name,
        "conservative_gate_method": conservative_gate_name,
        "conservative_gate_changed_points": gate_changed_points,
        "best_hard_filter_candidate": hard_filter_name,
        "recommendation_rule": (
            "First require >=90% Z>=30 m retention and >=80% total retention. "
            "If the conservative temporal gate finds no rejectable point, recommend "
            "no hard rejection because this sample has no lane ground truth. The "
            "hard-filter candidate is reported separately and is not approved as "
            "the production method."
        ),
        "methods": serializable_methods,
    }
    (audit_dir / "evaluation.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    legacy_row = next(
        row for row in metric_rows if row["method"] == "legacy_fixed_x_ransac"
    )
    hard_row = next(row for row in metric_rows if row["method"] == hard_filter_name)
    gate_row = next(
        row for row in metric_rows if row["method"] == conservative_gate_name
    )
    report_lines = [
        "# 去噪实验核验结论",
        "",
        "## 结论",
        "",
        "1. 旧方法不能继续作为正式去噪方法。它提高重叠率的主要代价是删除远处连续道路点。",
        "2. 当前五帧没有车道线真值；人工数据是含遮挡插值的伪标注，因此不能据此报告准确率。",
        f"3. 保守门控 `{conservative_gate_name}` 保留了 {gate_row['points']}/{raw_row['points']} 点，"
        "没有发现证据充分的离群点。当前最稳妥结果是保留未硬去噪点。",
        f"4. 如果必须继续验证硬去噪，优先候选为 `{hard_filter_name}`，但在更多帧或真值验证前不能替换主流程。",
        "",
        "## 核心数值",
        "",
        "| 方法 | 点数 | 30 m后保留 | 跨帧平均偏差 | 重叠率 | 阈值成图覆盖 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 不去噪 | {raw_row['points']} | {raw_row['far_point_retention_z_ge_30m']:.1%} | "
        f"{raw_row['mean_cross_frame_abs_deviation_m']:.3f} m | "
        f"{raw_row['fusion_overlap_fraction']:.1%} | {raw_row['fusion_thresholded_pixels']} px |",
        f"| 旧固定X窗口+RANSAC | {legacy_row['points']} | {legacy_row['far_point_retention_z_ge_30m']:.1%} | "
        f"{legacy_row['mean_cross_frame_abs_deviation_m']:.3f} m | "
        f"{legacy_row['fusion_overlap_fraction']:.1%} | {legacy_row['fusion_thresholded_pixels']} px |",
        f"| 候选 `{hard_filter_name}` | {hard_row['points']} | {hard_row['far_point_retention_z_ge_30m']:.1%} | "
        f"{hard_row['mean_cross_frame_abs_deviation_m']:.3f} m | "
        f"{hard_row['fusion_overlap_fraction']:.1%} | {hard_row['fusion_thresholded_pixels']} px |",
        f"| 保守门控 | {gate_row['points']} | {gate_row['far_point_retention_z_ge_30m']:.1%} | "
        f"{gate_row['mean_cross_frame_abs_deviation_m']:.3f} m | "
        f"{gate_row['fusion_overlap_fraction']:.1%} | {gate_row['fusion_thresholded_pixels']} px |",
        "",
        "## 为什么不能只看重叠率",
        "",
        f"旧方法把重叠率从 {raw_row['fusion_overlap_fraction']:.1%} 提高到 "
        f"{legacy_row['fusion_overlap_fraction']:.1%}，但阈值成图覆盖从 "
        f"{raw_row['fusion_thresholded_pixels']} px 降到 {legacy_row['fusion_thresholded_pixels']} px，"
        f"30 m 后只保留 {legacy_row['far_point_retention_z_ge_30m']:.1%}。"
        "删除不一致区域自然会提高剩余点的重叠比例，因此该指标必须与覆盖和分距离保留率一起看。",
        "",
        "## 核验入口",
        "",
        "- `comparison/main_comparison.png`：不去噪、旧方法、硬去噪候选的并排结果。",
        "- `00_audit/method_metrics.csv`：全部方法的原始指标。",
        "- `00_audit/longitudinal_retention.csv`：每个距离段的保留率。",
        "- `00_audit/per_frame_lane_retention.csv`：逐帧、左右线的保留数。",
        "- `00_audit/evaluation.json`：输入哈希、参数、方法细节和完整审计。",
    ]
    (args.output_dir / "EVALUATION_REPORT_ZH.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    (args.output_dir / "README.md").write_text(
        "\n".join(
            [
                "# KITTI 00 前五帧点式去噪实验",
                "",
                "本目录由 `scripts/evaluate_denoise_methods.py` 从已保存的 CLRNet",
                "有序二维点重新投影、位姿对齐并生成；没有覆盖历史结果。",
                "",
                f"- 原始点数：{len(points)}",
                f"- 自动推荐方法：`{recommended_name}`",
                f"- 保守门控：`{conservative_gate_name}`",
                f"- 硬去噪待验证候选：`{hard_filter_name}`",
                "- 推荐约束：总点保留率不低于 80%，30 m 以后点保留率不低于 90%。",
                "- 人工标注只是伪标注；相关数值只能称为一致性，不能称为准确率。",
                "- `EVALUATION_REPORT_ZH.md`：中文核验结论。",
                "- `00_audit/method_metrics.csv`：所有方法的核心指标。",
                "- `00_audit/longitudinal_retention.csv`：分距离段点保留情况。",
                "- `00_audit/per_frame_lane_retention.csv`：逐帧、逐侧保留情况。",
                "- `comparison/main_comparison.png`：未去噪、旧方法、推荐方法。",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output_dir),
                "recommended": recommended_name,
                "conservative_gate": conservative_gate_name,
                "hard_filter_candidate": hard_filter_name,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
