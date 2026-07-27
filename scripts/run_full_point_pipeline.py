"""Generate every core lane/BEV/fusion stage from raw inputs.

This runner is intentionally independent of pre-generated ``results/`` files.
It performs live CLRNet inference, keeps lane geometry as ordered points through
IPM, pose alignment, and legacy denoising, and only rasterizes when writing an
image.  Stage images contain no titles, labels, legends, or comparison panels.

The legacy denoiser is preserved only as a diagnostic baseline.  Its fixed-X
pre-clustering is known to remove long-range points on the current five-frame
sample and must not be presented as the final denoising solution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from surf_bev.detectors import CLRNetLaneDetector  # noqa: E402
from surf_bev.geometry import (  # noqa: E402
    image_to_ground_ipm,
    load_kitti_poses,
    transform_lane_points_by_pose,
)
from surf_bev.temporal_denoise import (  # noqa: E402
    leave_one_out_consensus_mask,
)


FRAME_COLORS_BGR = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
]
FRAME_COLORS_RGB = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
LANE_COLORS_BGR = [(255, 210, 0), (0, 255, 255)]
ALL_CANDIDATE_COLORS_BGR = [
    (255, 80, 80),
    (80, 220, 80),
    (80, 80, 255),
    (255, 180, 40),
    (220, 80, 220),
    (220, 220, 40),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="From-scratch CLRNet point BEV and fusion reproduction."
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--image-pattern", default="frame_{frame_id:06d}.png")
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-ids", default="0,1,2,3,4")
    parser.add_argument("--reference-id", type=int, default=4)
    parser.add_argument("--camera-height", type=float, default=1.65)
    parser.add_argument("--pitch-deg", type=float, default=0.0)
    parser.add_argument("--local-x-range", default="-10,10")
    parser.add_argument("--local-z-range", default="3,50")
    parser.add_argument("--fusion-x-range", default="-15,15")
    parser.add_argument("--fusion-z-range", default="-10,50")
    parser.add_argument("--bev-size", type=int, default=800)
    parser.add_argument("--weights", default="0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--legacy-ransac-iterations", type=int, default=100)
    parser.add_argument("--legacy-ransac-threshold", type=float, default=0.3)
    parser.add_argument("--temporal-base-threshold", type=float, default=0.30)
    parser.add_argument("--temporal-mad-multiplier", type=float, default=3.0)
    parser.add_argument("--temporal-threshold-cap", type=float, default=1.00)
    parser.add_argument("--temporal-min-other-frames", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--clrnet-root", type=Path, default=ROOT / "CLRNet")
    parser.add_argument(
        "--clrnet-config", default="configs/clrnet/clr_resnet18_culane.py"
    )
    parser.add_argument(
        "--clrnet-checkpoint", default="weights/culane_r18.pth"
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--provenance",
        type=Path,
        default=None,
        help="Optional JSON with expected SHA-256 values for the official frames.",
    )
    return parser.parse_args()


def comma_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def comma_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def imread(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(path)


def require_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {path}. Use a new directory so the run "
            "cannot reuse old results."
        )
    path.mkdir(parents=True, exist_ok=True)


def parse_range(value: str, option: str) -> tuple[float, float]:
    values = comma_floats(value)
    if len(values) != 2 or values[0] >= values[1]:
        raise ValueError(f"{option} requires two increasing comma-separated values.")
    return values[0], values[1]


def parse_kitti_projection(calib_path: Path) -> dict[str, object]:
    values: dict[str, np.ndarray] = {}
    for line in calib_path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        values[key.strip()] = np.asarray(
            [float(item) for item in raw.split()], dtype=np.float64
        )
    if "P2" in values:
        key = "P2"
    elif "P0" in values:
        key = "P0"
    else:
        raise KeyError("Calibration must contain P2 or P0.")
    projection = values[key].reshape(3, 4)
    intrinsic_k = projection[:, :3]

    # KITTI odometry poses are poses of rectified camera 0. P2 projects camera-0
    # coordinates into the left colour image, so its fourth column must not be
    # silently discarded when image_2 is paired with poses/00.txt.
    translation_cam_image_from_cam0 = np.linalg.solve(
        intrinsic_k, projection[:, 3]
    )
    T_cam_image_cam0 = np.eye(4, dtype=np.float64)
    T_cam_image_cam0[:3, 3] = translation_cam_image_from_cam0
    T_cam0_cam_image = np.linalg.inv(T_cam_image_cam0)
    return {
        "key": key,
        "projection": projection,
        "K": intrinsic_k,
        "T_cam_image_cam0": T_cam_image_cam0,
        "T_cam0_cam_image": T_cam0_cam_image,
    }


def poses_for_image_camera(
    poses_cam0: list[np.ndarray], T_cam0_cam_image: np.ndarray
) -> list[np.ndarray]:
    """Convert KITTI camera-0 world poses to the camera used by P0/P2."""

    return [pose @ T_cam0_cam_image for pose in poses_cam0]


def filter_metric_points(
    points: np.ndarray,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return points
    valid = (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= x_range[0])
        & (points[:, 0] <= x_range[1])
        & (points[:, 1] >= z_range[0])
        & (points[:, 1] <= z_range[1])
    )
    return points[valid]


def verify_provenance(
    provenance_path: Path | None,
    image_paths: list[Path],
    frame_ids: list[int],
) -> dict[str, object] | None:
    if provenance_path is None:
        return None
    if not provenance_path.is_file():
        raise FileNotFoundError(f"Missing provenance JSON: {provenance_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected = {
        int(item["verified_kitti_frame"]): item["sha256"]
        for item in provenance.get("mapping", [])
    }
    mismatches = []
    for frame_id, image_path in zip(frame_ids, image_paths):
        actual = sha256(image_path)
        if expected.get(frame_id) != actual:
            mismatches.append(
                {
                    "frame_id": frame_id,
                    "expected_sha256": expected.get(frame_id),
                    "actual_sha256": actual,
                }
            )
    if mismatches:
        raise ValueError(
            "Input images do not match the audited KITTI Sequence 00 frames: "
            + json.dumps(mismatches, ensure_ascii=False)
        )
    return {
        "path": str(provenance_path.resolve()),
        "sha256": sha256(provenance_path),
        "verified_frames": frame_ids,
        "conclusion": provenance.get("conclusion"),
    }


def checkpoint_path(clrnet_root: Path, checkpoint: str) -> Path:
    candidate = Path(checkpoint)
    return candidate if candidate.is_absolute() else clrnet_root / candidate


def bottom_x(lane: np.ndarray) -> float:
    lane = np.asarray(lane, dtype=np.float64).reshape(-1, 2)
    threshold = np.quantile(lane[:, 1], 0.85)
    return float(np.median(lane[lane[:, 1] >= threshold, 0]))


def select_outer_two(
    lanes: list[np.ndarray],
) -> tuple[list[tuple[int, np.ndarray]], list[int]]:
    if len(lanes) < 2:
        raise ValueError(f"Expected at least two CLRNet lanes, received {len(lanes)}")
    ordered = sorted(enumerate(lanes), key=lambda pair: bottom_x(pair[1]))
    selected = [ordered[0], ordered[-1]]
    selected_indices = {selected[0][0], selected[1][0]}
    rejected = [index for index in range(len(lanes)) if index not in selected_indices]
    return selected, rejected


def draw_image_points(
    image: np.ndarray,
    lanes: Iterable[np.ndarray],
    colors: list[tuple[int, int, int]],
    radius: int = 3,
) -> np.ndarray:
    output = image.copy()
    for lane_index, lane in enumerate(lanes):
        color = colors[lane_index % len(colors)]
        for point in np.asarray(lane, dtype=np.float64).reshape(-1, 2):
            if not np.isfinite(point).all():
                continue
            cv2.circle(
                output,
                tuple(np.rint(point).astype(np.int32)),
                radius,
                color,
                -1,
                cv2.LINE_AA,
            )
    return output


def metric_to_pixels(
    points: np.ndarray,
    bev_size: int,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
) -> np.ndarray:
    points = filter_metric_points(points, x_range, z_range)
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.int32)
    u = (
        (points[:, 0] - x_range[0])
        / (x_range[1] - x_range[0])
        * (bev_size - 1)
    )
    v = (
        (z_range[1] - points[:, 1])
        / (z_range[1] - z_range[0])
        * (bev_size - 1)
    )
    return np.rint(np.column_stack([u, v])).astype(np.int32)


def draw_metric_points(
    lanes: Iterable[np.ndarray],
    colors: list[tuple[int, int, int]],
    bev_size: int,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
    radius: int = 3,
    background: int = 255,
) -> np.ndarray:
    output = np.full((bev_size, bev_size, 3), background, dtype=np.uint8)
    for lane_index, lane in enumerate(lanes):
        color = colors[lane_index % len(colors)]
        for point in metric_to_pixels(lane, bev_size, x_range, z_range):
            cv2.circle(output, tuple(point), radius, color, -1, cv2.LINE_AA)
    return output


def draw_frame_accumulation(
    frames: list[list[np.ndarray]],
    bev_size: int,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
) -> np.ndarray:
    output = np.full((bev_size, bev_size, 3), 255, dtype=np.uint8)
    for frame_index, lanes in enumerate(frames):
        color = FRAME_COLORS_BGR[frame_index % len(FRAME_COLORS_BGR)]
        for lane in lanes:
            for point in metric_to_pixels(lane, bev_size, x_range, z_range):
                cv2.circle(output, tuple(point), 3, color, -1, cv2.LINE_AA)
    return output


def save_pose_metric_plot(
    path: Path,
    frames: list[list[np.ndarray]],
    frame_ids: list[int],
    camera_origins_xz: list[np.ndarray],
    reference_id: int,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
) -> None:
    """Save a frame-coloured metric plot that proves pose alignment was applied."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 12))
    for frame_index, (frame_id, lanes, origin) in enumerate(
        zip(frame_ids, frames, camera_origins_xz)
    ):
        color = FRAME_COLORS_RGB[frame_index % len(FRAME_COLORS_RGB)]
        labelled = False
        for lane in lanes:
            points = np.asarray(lane, dtype=np.float64).reshape(-1, 2)
            if not len(points):
                continue
            axis.plot(
                points[:, 0],
                points[:, 1],
                "o-",
                color=color,
                linewidth=1.5,
                markersize=3,
                label=f"frame_{frame_id:06d}" if not labelled else None,
            )
            labelled = True
        axis.scatter(origin[0], origin[1], marker="x", s=90, color=color)
    axis.set(
        xlim=x_range,
        ylim=z_range,
        xlabel="X right in reference frame [m]",
        ylabel="Z forward in reference frame [m]",
        title=(
            "KITTI Odometry Sequence 00: five-frame pose alignment "
            f"-> frame_{reference_id:06d}"
        ),
    )
    axis.set_aspect("equal")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def lane_mask(
    lanes: list[np.ndarray],
    bev_size: int,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
    thickness: int = 3,
) -> np.ndarray:
    mask = np.zeros((bev_size, bev_size), dtype=np.uint8)
    for lane in lanes:
        pixels = metric_to_pixels(lane, bev_size, x_range, z_range)
        if len(pixels) == 1:
            cv2.circle(mask, tuple(pixels[0]), max(1, thickness // 2), 255, -1)
        elif len(pixels) >= 2:
            cv2.polylines(mask, [pixels], False, 255, thickness, cv2.LINE_AA)
    return mask


def weighted_raster_fusion(
    frames: list[list[np.ndarray]],
    weights: np.ndarray,
    bev_size: int,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
) -> dict[str, np.ndarray | int | float]:
    masks = [
        lane_mask(lanes, bev_size, x_range, z_range, thickness=3)
        for lanes in frames
    ]
    accumulation = np.zeros((bev_size, bev_size), dtype=np.float32)
    for mask, weight in zip(masks, weights):
        accumulation += mask.astype(np.float32) / 255.0 * float(weight)
    score = np.clip(accumulation / float(np.sum(weights)), 0.0, 1.0)
    intensity = np.rint(score * 255.0).astype(np.uint8)
    heatmap = cv2.applyColorMap(intensity, cv2.COLORMAP_TURBO)
    heatmap[intensity == 0] = 255
    binary = np.where(score >= 0.08, 255, 0).astype(np.uint8)
    hits = np.stack([mask > 0 for mask in masks]).sum(axis=0)
    union = int(np.count_nonzero(hits))
    overlap = int(np.count_nonzero(hits >= 2))
    return {
        "masks": masks,
        "score": score,
        "intensity": intensity,
        "heatmap": heatmap,
        "binary": binary,
        "union_pixels": union,
        "overlap_pixels": overlap,
        "overlap_fraction": float(overlap / union) if union else 0.0,
        "thresholded_pixels": int(np.count_nonzero(binary)),
    }


def simple_cluster_by_x(points: np.ndarray) -> dict[str, np.ndarray | float]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 10:
        return {
            "left": points,
            "right": points,
            "left_center": float("nan"),
            "right_center": float("nan"),
            "left_window_m": float("nan"),
            "right_window_m": float("nan"),
        }
    x_values = points[:, 0]
    left_center = (
        float(np.percentile(x_values[x_values < 0], 75))
        if np.any(x_values < 0)
        else -1.5
    )
    right_center = (
        float(np.percentile(x_values[x_values > 0], 25))
        if np.any(x_values > 0)
        else 1.5
    )
    left_window = 0.8
    right_window = 0.8
    left_mask = np.abs(points[:, 0] - left_center) < left_window
    right_mask = np.abs(points[:, 0] - right_center) < right_window
    if np.count_nonzero(left_mask) < 20:
        left_window = 1.2
        left_mask = np.abs(points[:, 0] - left_center) < left_window
    if np.count_nonzero(right_mask) < 20:
        right_window = 1.2
        right_mask = np.abs(points[:, 0] - right_center) < right_window
    return {
        "left": points[left_mask],
        "right": points[right_mask],
        "left_center": left_center,
        "right_center": right_center,
        "left_window_m": left_window,
        "right_window_m": right_window,
    }


def ransac_line(
    points: np.ndarray,
    max_iter: int,
    threshold: float,
    rng: np.random.Generator,
) -> tuple[tuple[float, float] | None, np.ndarray]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 5:
        return None, points
    best_inliers = np.empty((0, 2), dtype=np.float64)
    for _ in range(max_iter):
        indexes = rng.choice(len(points), 2, replace=False)
        p1, p2 = points[indexes]
        if abs(p2[0] - p1[0]) < 1e-6:
            continue
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        intercept = p1[1] - slope * p1[0]
        distances = np.abs(points[:, 1] - (slope * points[:, 0] + intercept))
        distances /= np.sqrt(slope**2 + 1.0)
        inliers = points[distances < threshold]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
    if len(best_inliers) < 5:
        return None, points
    design = np.vstack([best_inliers[:, 0], np.ones(len(best_inliers))]).T
    slope, intercept = np.linalg.lstsq(
        design, best_inliers[:, 1], rcond=None
    )[0]
    return (float(slope), float(intercept)), best_inliers


def legacy_denoise(
    points: np.ndarray,
    max_iter: int,
    threshold: float,
    seed: int,
) -> dict[str, object]:
    clustered = simple_cluster_by_x(points)
    rng = np.random.default_rng(seed)
    left_model, left_inliers = ransac_line(
        clustered["left"], max_iter, threshold, rng
    )
    right_model, right_inliers = ransac_line(
        clustered["right"], max_iter, threshold, rng
    )
    return {
        **clustered,
        "left_model": left_model,
        "right_model": right_model,
        "left_inliers": left_inliers,
        "right_inliers": right_inliers,
    }


def point_membership(points: np.ndarray, selected: np.ndarray) -> np.ndarray:
    if len(points) == 0 or len(selected) == 0:
        return np.zeros(len(points), dtype=bool)
    keys = {tuple(row) for row in np.round(selected, 9)}
    return np.asarray(
        [tuple(row) in keys for row in np.round(points, 9)], dtype=bool
    )


def split_runs(points: np.ndarray, keep: np.ndarray) -> list[np.ndarray]:
    runs: list[np.ndarray] = []
    start: int | None = None
    for index, is_kept in enumerate(keep):
        if is_kept and start is None:
            start = index
        if start is not None and (not is_kept or index == len(keep) - 1):
            end = index if not is_kept else index + 1
            if end - start >= 2:
                runs.append(points[start:end])
            start = None
    return runs


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scalar_metrics(result: dict[str, object]) -> dict[str, int | float]:
    return {
        key: result[key]
        for key in (
            "union_pixels",
            "overlap_pixels",
            "overlap_fraction",
            "thresholded_pixels",
        )
    }


def save_fusion_stage(
    output_dir: Path,
    frames: list[list[np.ndarray]],
    fusion: dict[str, object],
    frame_ids: list[int],
    bev_size: int,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
) -> None:
    for frame_index, frame_id in enumerate(frame_ids):
        imwrite(
            output_dir / "per_frame_points" / f"frame_{frame_id:06d}.png",
            draw_metric_points(
                frames[frame_index],
                LANE_COLORS_BGR,
                bev_size,
                x_range,
                z_range,
            ),
        )
        imwrite(
            output_dir / "per_frame_masks" / f"frame_{frame_id:06d}.png",
            fusion["masks"][frame_index],
        )
    imwrite(
        output_dir / "accumulated_points_by_frame.png",
        draw_frame_accumulation(frames, bev_size, x_range, z_range),
    )
    imwrite(output_dir / "weighted_score_grayscale.png", fusion["intensity"])
    imwrite(output_dir / "weighted_score_heatmap.png", fusion["heatmap"])
    imwrite(output_dir / "thresholded_binary_mask.png", fusion["binary"])


def build_manifest(output_dir: Path) -> dict[str, object]:
    files = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": path.relative_to(output_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    return {"files": files, "file_count": len(files)}


def main() -> None:
    args = parse_args()
    frame_ids = comma_ints(args.frame_ids)
    weights = np.asarray(comma_floats(args.weights), dtype=np.float64)
    local_x_range = parse_range(args.local_x_range, "--local-x-range")
    local_z_range = parse_range(args.local_z_range, "--local-z-range")
    fusion_x_range = parse_range(args.fusion_x_range, "--fusion-x-range")
    fusion_z_range = parse_range(args.fusion_z_range, "--fusion-z-range")
    if len(frame_ids) != len(weights):
        raise ValueError("The number of frame IDs must equal the number of weights.")
    if not frame_ids or len(set(frame_ids)) != len(frame_ids):
        raise ValueError("Frame IDs must be a non-empty list without duplicates.")
    if not np.isfinite(weights).all() or np.any(weights < 0) or np.sum(weights) <= 0:
        raise ValueError("Weights must be finite, non-negative, and have a positive sum.")
    if (
        args.temporal_base_threshold <= 0
        or args.temporal_mad_multiplier <= 0
        or args.temporal_threshold_cap < args.temporal_base_threshold
        or args.temporal_min_other_frames < 1
    ):
        raise ValueError(
            "Temporal denoising requires positive thresholds/multiplier/support, "
            "and the threshold cap must be >= the base threshold."
        )
    if args.reference_id not in frame_ids:
        raise ValueError("Reference ID must be one of the requested frame IDs.")

    image_paths = [
        args.image_dir / args.image_pattern.format(frame_id=frame_id)
        for frame_id in frame_ids
    ]
    required = [*image_paths, args.calib, args.poses]
    if args.provenance is not None:
        required.append(args.provenance)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))
    resolved_checkpoint = checkpoint_path(args.clrnet_root, args.clrnet_checkpoint)
    if not resolved_checkpoint.is_file():
        raise FileNotFoundError(f"Missing CLRNet checkpoint: {resolved_checkpoint}")

    output_dir = args.output_dir.resolve()
    require_empty_output(output_dir)
    status_path = output_dir / "00_metadata" / "run_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps({"status": "running"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    try:
        import torch

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        provenance_audit = verify_provenance(
            args.provenance, image_paths, frame_ids
        )
        calibration = parse_kitti_projection(args.calib)
        intrinsic_k = calibration["K"]
        poses_cam0 = load_kitti_poses(args.poses)
        if max(frame_ids) >= len(poses_cam0):
            raise IndexError(
                f"Pose file contains {len(poses_cam0)} rows but frame {max(frame_ids)} was requested."
            )
        poses = poses_for_image_camera(
            poses_cam0, calibration["T_cam0_cam_image"]
        )
        reference_pose = poses[args.reference_id]
        relative_poses = [np.linalg.inv(reference_pose) @ poses[item] for item in frame_ids]
        camera_origins_xz = [pose[[0, 2], 3] for pose in relative_poses]
        detector = CLRNetLaneDetector(
            clrnet_root=str(args.clrnet_root.resolve()),
            config=args.clrnet_config,
            checkpoint=args.clrnet_checkpoint,
            device=args.device,
        )

        detection_records: list[dict[str, object]] = []
        frame_metrics: list[dict[str, object]] = []
        projected_frames: list[list[np.ndarray]] = []
        aligned_frames: list[list[np.ndarray]] = []
        fusion_frames: list[list[np.ndarray]] = []
        flat_points: list[list[float]] = []
        point_ranges: list[dict[str, object]] = []

        for frame_index, (frame_id, image_path) in enumerate(
            zip(frame_ids, image_paths)
        ):
            image = imread(image_path)
            original_out = (
                output_dir / "01_original_frames" / f"frame_{frame_id:06d}.png"
            )
            original_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, original_out)

            detection = detector.detect(image)
            lanes = [
                np.asarray(lane, dtype=np.float64).reshape(-1, 2)
                for lane in detection["lanes"]
            ]
            selected, rejected = select_outer_two(lanes)
            selected_lanes = [lane for _, lane in selected]

            imwrite(
                output_dir
                / "02_clrnet_points"
                / "all_candidates"
                / f"frame_{frame_id:06d}.png",
                draw_image_points(image, lanes, ALL_CANDIDATE_COLORS_BGR),
            )
            imwrite(
                output_dir
                / "02_clrnet_points"
                / "selected_two"
                / f"frame_{frame_id:06d}.png",
                draw_image_points(image, selected_lanes, LANE_COLORS_BGR),
            )

            projected_lanes: list[np.ndarray] = []
            aligned_lanes: list[np.ndarray] = []
            fusion_lanes: list[np.ndarray] = []
            selected_records = []
            for side, (lane_index, lane) in zip(("left", "right"), selected):
                ground = image_to_ground_ipm(
                    lane,
                    intrinsic_k,
                    camera_height=args.camera_height,
                    pitch_deg=args.pitch_deg,
                    z_range=local_z_range,
                )
                aligned = transform_lane_points_by_pose(
                    ground,
                    poses[frame_id],
                    reference_pose,
                    camera_height=args.camera_height,
                    pitch_deg=args.pitch_deg,
                )
                aligned = aligned[np.isfinite(aligned).all(axis=1)]
                aligned_for_fusion = filter_metric_points(
                    aligned, fusion_x_range, fusion_z_range
                )
                start = len(flat_points)
                flat_points.extend(aligned_for_fusion.tolist())
                point_ranges.append(
                    {
                        "frame_index": frame_index,
                        "frame_id": frame_id,
                        "side": side,
                        "lane_index": lane_index,
                        "start": start,
                        "end": len(flat_points),
                    }
                )
                projected_lanes.append(ground)
                aligned_lanes.append(aligned)
                fusion_lanes.append(aligned_for_fusion)
                selected_records.append(
                    {
                        "side": side,
                        "lane_index": lane_index,
                        "bottom_x_px": bottom_x(lane),
                        "image_points": int(len(lane)),
                        "projected_points": int(len(ground)),
                        "aligned_points": int(len(aligned)),
                        "aligned_in_fusion_range_points": int(
                            len(aligned_for_fusion)
                        ),
                        "image_xy_px": lane.tolist(),
                        "local_ground_xz_m": ground.tolist(),
                        "reference_ground_xz_m": aligned.tolist(),
                    }
                )

            projected_frames.append(projected_lanes)
            aligned_frames.append(aligned_lanes)
            fusion_frames.append(fusion_lanes)
            imwrite(
                output_dir / "03_bev_points" / f"frame_{frame_id:06d}.png",
                draw_metric_points(
                    projected_lanes,
                    LANE_COLORS_BGR,
                    args.bev_size,
                    local_x_range,
                    local_z_range,
                ),
            )
            imwrite(
                output_dir
                / "04_pose_aligned_points"
                / f"frame_{frame_id:06d}.png",
                draw_metric_points(
                    aligned_lanes,
                    LANE_COLORS_BGR,
                    args.bev_size,
                    fusion_x_range,
                    fusion_z_range,
                ),
            )

            detection_records.append(
                {
                    "frame_id": frame_id,
                    "image": str(image_path.resolve()),
                    "image_sha256": sha256(image_path),
                    "candidate_count": len(lanes),
                    "candidate_lanes_image_xy_px": [lane.tolist() for lane in lanes],
                    "selected": selected_records,
                    "rejected_candidate_indices": rejected,
                }
            )
            frame_metrics.append(
                {
                    "frame_id": frame_id,
                    "candidate_lanes": len(lanes),
                    "selected_image_points": sum(len(lane) for lane in selected_lanes),
                    "projected_points": sum(len(lane) for lane in projected_lanes),
                    "aligned_points": sum(len(lane) for lane in aligned_lanes),
                    "aligned_in_fusion_range_points": sum(
                        len(lane) for lane in fusion_lanes
                    ),
                    "camera_origin_x_ref_m": float(camera_origins_xz[frame_index][0]),
                    "camera_origin_z_ref_m": float(camera_origins_xz[frame_index][1]),
                }
            )

        imwrite(
            output_dir / "04_pose_aligned_points" / "five_frame_points_by_frame.png",
            draw_frame_accumulation(
                aligned_frames,
                args.bev_size,
                fusion_x_range,
                fusion_z_range,
            ),
        )
        save_pose_metric_plot(
            output_dir
            / "04_pose_aligned_points"
            / "five_frame_metric_pose_fusion.png",
            aligned_frames,
            frame_ids,
            camera_origins_xz,
            args.reference_id,
            fusion_x_range,
            fusion_z_range,
        )

        all_points = np.asarray(flat_points, dtype=np.float64).reshape(-1, 2)
        raw_fusion = weighted_raster_fusion(
            fusion_frames,
            weights,
            args.bev_size,
            fusion_x_range,
            fusion_z_range,
        )
        save_fusion_stage(
            output_dir / "05_fusion_without_denoise",
            fusion_frames,
            raw_fusion,
            frame_ids,
            args.bev_size,
            fusion_x_range,
            fusion_z_range,
        )

        legacy = legacy_denoise(
            all_points,
            args.legacy_ransac_iterations,
            args.legacy_ransac_threshold,
            args.seed,
        )
        clustered = np.vstack([legacy["left"], legacy["right"]])
        inlier_arrays = [legacy["left_inliers"], legacy["right_inliers"]]
        inlier_arrays = [array for array in inlier_arrays if len(array)]
        kept = (
            np.unique(np.round(np.vstack(inlier_arrays), 9), axis=0)
            if inlier_arrays
            else np.empty((0, 2), dtype=np.float64)
        )
        cluster_membership = point_membership(all_points, clustered)
        final_membership = point_membership(all_points, kept)
        clustered_frames: list[list[np.ndarray]] = [[] for _ in aligned_frames]
        denoised_frames: list[list[np.ndarray]] = [[] for _ in aligned_frames]
        retention_rows: list[dict[str, object]] = []
        for record in point_ranges:
            start = int(record["start"])
            end = int(record["end"])
            lane = all_points[start:end]
            cluster_keep = cluster_membership[start:end]
            final_keep = final_membership[start:end]
            frame_index = int(record["frame_index"])
            clustered_frames[frame_index].extend(split_runs(lane, cluster_keep))
            denoised_frames[frame_index].extend(split_runs(lane, final_keep))
            retention_rows.append(
                {
                    "frame_id": record["frame_id"],
                    "side": record["side"],
                    "raw_points": len(lane),
                    "after_x_cluster": int(np.count_nonzero(cluster_keep)),
                    "after_line_ransac": int(np.count_nonzero(final_keep)),
                }
            )

        cluster_fusion = weighted_raster_fusion(
            clustered_frames,
            weights,
            args.bev_size,
            fusion_x_range,
            fusion_z_range,
        )
        denoised_fusion = weighted_raster_fusion(
            denoised_frames,
            weights,
            args.bev_size,
            fusion_x_range,
            fusion_z_range,
        )
        save_fusion_stage(
            output_dir / "06_legacy_ransac_diagnostic" / "01_after_x_cluster",
            clustered_frames,
            cluster_fusion,
            frame_ids,
            args.bev_size,
            fusion_x_range,
            fusion_z_range,
        )
        save_fusion_stage(
            output_dir / "06_legacy_ransac_diagnostic" / "02_after_line_ransac",
            denoised_frames,
            denoised_fusion,
            frame_ids,
            args.bev_size,
            fusion_x_range,
            fusion_z_range,
        )

        z_bins = [(-10, 0), (0, 3), (3, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
        longitudinal_rows = []
        for z_min, z_max in z_bins:
            raw_bin = (all_points[:, 1] >= z_min) & (all_points[:, 1] < z_max)
            kept_count = int(np.count_nonzero(raw_bin & final_membership))
            raw_count = int(np.count_nonzero(raw_bin))
            longitudinal_rows.append(
                {
                    "z_min_m": z_min,
                    "z_max_m": z_max,
                    "raw_points": raw_count,
                    "kept_points": kept_count,
                    "retention_fraction": kept_count / raw_count if raw_count else 0.0,
                }
            )

        temporal_membership, temporal_details = leave_one_out_consensus_mask(
            all_points,
            point_ranges,
            base_threshold=args.temporal_base_threshold,
            mad_multiplier=args.temporal_mad_multiplier,
            threshold_cap=args.temporal_threshold_cap,
            minimum_other_frames=args.temporal_min_other_frames,
        )
        temporal_frames: list[list[np.ndarray]] = [[] for _ in aligned_frames]
        temporal_retention_rows: list[dict[str, object]] = []
        for record in point_ranges:
            start = int(record["start"])
            end = int(record["end"])
            lane = all_points[start:end]
            keep = temporal_membership[start:end]
            frame_index = int(record["frame_index"])
            temporal_frames[frame_index].extend(split_runs(lane, keep))
            temporal_retention_rows.append(
                {
                    "frame_id": record["frame_id"],
                    "side": record["side"],
                    "raw_points": len(lane),
                    "kept_points": int(np.count_nonzero(keep)),
                    "retention_fraction": float(np.mean(keep)) if len(keep) else 0.0,
                }
            )
        temporal_fusion = weighted_raster_fusion(
            temporal_frames,
            weights,
            args.bev_size,
            fusion_x_range,
            fusion_z_range,
        )
        save_fusion_stage(
            output_dir / "07_temporal_consensus_candidate",
            temporal_frames,
            temporal_fusion,
            frame_ids,
            args.bev_size,
            fusion_x_range,
            fusion_z_range,
        )
        temporal_longitudinal_rows = []
        for z_min, z_max in z_bins:
            raw_bin = (all_points[:, 1] >= z_min) & (all_points[:, 1] < z_max)
            raw_count = int(np.count_nonzero(raw_bin))
            kept_count = int(np.count_nonzero(raw_bin & temporal_membership))
            temporal_longitudinal_rows.append(
                {
                    "z_min_m": z_min,
                    "z_max_m": z_max,
                    "raw_points": raw_count,
                    "kept_points": kept_count,
                    "retention_fraction": kept_count / raw_count if raw_count else 0.0,
                }
            )

        audit = {
            "status": "complete",
            "method_scope": {
                "geometry": "ordered CLRNet points -> metric IPM -> pose alignment",
                "visualization": "point images without titles, labels, legends, or panels",
                "raster_fusion": "legacy per-frame polyline masks and temporal weights",
                "legacy_denoise_warning": (
                    "Diagnostic baseline only. Fixed-X clustering is known to remove "
                    "long-range points and is not the final denoising method."
                ),
                "temporal_candidate": (
                    "Leave-one-frame-out same-side X(Z) consensus. Unsupported "
                    "points are kept; parameters are experiment-tuned candidates."
                ),
            },
            "environment": {
                "platform": platform.platform(),
                "python": sys.version.split()[0],
                "opencv": cv2.__version__,
                "numpy": np.__version__,
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None,
            },
            "inputs": {
                "frames": frame_ids,
                "images": [
                    {"path": str(path.resolve()), "sha256": sha256(path)}
                    for path in image_paths
                ],
                "calibration": {
                    "path": str(args.calib.resolve()),
                    "sha256": sha256(args.calib),
                    "projection_key": calibration["key"],
                    "projection_matrix": calibration["projection"].tolist(),
                    "K": intrinsic_k.tolist(),
                    "T_cam_image_cam0": calibration[
                        "T_cam_image_cam0"
                    ].tolist(),
                    "T_cam0_cam_image": calibration[
                        "T_cam0_cam_image"
                    ].tolist(),
                },
                "poses": {
                    "path": str(args.poses.resolve()),
                    "sha256": sha256(args.poses),
                    "definition": (
                        "KITTI odometry T_world_camera0 converted to the P2 image "
                        "camera before relative alignment"
                    ),
                },
                "provenance": provenance_audit,
            },
            "detector": {
                "name": "CLRNet",
                "config": args.clrnet_config,
                "checkpoint": str(resolved_checkpoint.resolve()),
                "checkpoint_sha256": sha256(resolved_checkpoint),
                "device": args.device,
                "frames": detection_records,
            },
            "projection": {
                "type": "sparse point ray-plane IPM",
                "camera_height_m": args.camera_height,
                "pitch_deg": args.pitch_deg,
                "assumption_warning": (
                    "Camera height, pitch, and flat road are experiment assumptions, "
                    "not per-frame road-plane ground truth."
                ),
                "local_x_range_m": list(local_x_range),
                "local_z_range_m": list(local_z_range),
                "fusion_x_range_m": list(fusion_x_range),
                "fusion_z_range_m": list(fusion_z_range),
                "bev_size_px": [args.bev_size, args.bev_size],
            },
            "pose_alignment": {
                "formula": "p_ref = inv(T_w_ref) @ T_w_src @ p_src",
                "reference_frame": args.reference_id,
                "frames": [
                    {
                        "frame_id": frame_id,
                        "T_world_camera0": poses_cam0[frame_id].tolist(),
                        "T_world_image_camera": poses[frame_id].tolist(),
                        "T_reference_from_source": relative_pose.tolist(),
                        "camera_origin_reference_xz_m": origin.tolist(),
                        "translation_norm_m": float(
                            np.linalg.norm(relative_pose[:3, 3])
                        ),
                    }
                    for frame_id, relative_pose, origin in zip(
                        frame_ids, relative_poses, camera_origins_xz
                    )
                ],
            },
            "fusion_without_denoise": {
                "point_count": int(len(all_points)),
                "weights": weights.tolist(),
                "weights_note": (
                    "User-configured temporal raster weights. The default 0.2,0.4,"
                    "0.6,0.8,1.0 is a project heuristic, not a value published by "
                    "KITTI or CLRNet; the metric pose plot itself is unweighted."
                ),
                **scalar_metrics(raw_fusion),
            },
            "legacy_ransac_diagnostic": {
                "iterations": args.legacy_ransac_iterations,
                "threshold_m": args.legacy_ransac_threshold,
                "seed": args.seed,
                "left_center_m": legacy["left_center"],
                "right_center_m": legacy["right_center"],
                "left_window_m": legacy["left_window_m"],
                "right_window_m": legacy["right_window_m"],
                "raw_points": int(len(all_points)),
                "after_x_cluster_points": int(len(clustered)),
                "after_line_ransac_points": int(len(kept)),
                "after_x_cluster_fusion": scalar_metrics(cluster_fusion),
                "after_line_ransac_fusion": scalar_metrics(denoised_fusion),
                "retention_by_lane": retention_rows,
                "retention_by_longitudinal_bin": longitudinal_rows,
            },
            "temporal_consensus_candidate": {
                "method": (
                    "leave-one-frame-out median X at the same Z, with MAD-adaptive "
                    "threshold; the current frame never supports its own point"
                ),
                "parameters": temporal_details,
                "parameter_provenance": (
                    "Selected by controlled outlier-injection experiments with "
                    "real/far/manual-pseudo-label retention constraints; not a "
                    "published KITTI or CLRNet parameter."
                ),
                "raw_points": int(len(all_points)),
                "kept_points": int(np.count_nonzero(temporal_membership)),
                "point_retention_fraction": float(np.mean(temporal_membership)),
                "fusion": scalar_metrics(temporal_fusion),
                "retention_by_lane": temporal_retention_rows,
                "retention_by_longitudinal_bin": temporal_longitudinal_rows,
                "warning": (
                    "Candidate method only. Injection results do not replace lane "
                    "ground truth, and longer sequences still require validation."
                ),
            },
            "image_color_conventions_bgr": {
                "left_right": [list(color) for color in LANE_COLORS_BGR],
                "frames": [list(color) for color in FRAME_COLORS_BGR],
            },
        }

        metadata_dir = output_dir / "00_metadata"
        (metadata_dir / "detected_lane_points.json").write_text(
            json.dumps(
                {
                    "detector": audit["detector"],
                    "coordinate_system": "original image pixel coordinates (u,v)",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (metadata_dir / "audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (metadata_dir / "aligned_lane_points.json").write_text(
            json.dumps(
                {
                    "coordinate_system": (
                        f"metric X/Z in reference image camera frame {args.reference_id}"
                    ),
                    "frames": [
                        {
                            "frame_id": frame_id,
                            "camera_origin_reference_xz_m": origin.tolist(),
                            "lanes_reference_xz_m": [lane.tolist() for lane in lanes],
                        }
                        for frame_id, origin, lanes in zip(
                            frame_ids, camera_origins_xz, aligned_frames
                        )
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        write_csv(metadata_dir / "frame_metrics.csv", frame_metrics)
        write_csv(
            metadata_dir / "pose_alignment.csv",
            [
                {
                    "frame_id": frame_id,
                    "reference_frame_id": args.reference_id,
                    "camera_origin_x_ref_m": float(origin[0]),
                    "camera_origin_z_ref_m": float(origin[1]),
                    "translation_norm_m": float(
                        np.linalg.norm(relative_pose[:3, 3])
                    ),
                }
                for frame_id, relative_pose, origin in zip(
                    frame_ids, relative_poses, camera_origins_xz
                )
            ],
        )
        write_csv(metadata_dir / "legacy_retention_by_lane.csv", retention_rows)
        write_csv(
            metadata_dir / "legacy_retention_by_distance.csv", longitudinal_rows
        )
        write_csv(
            metadata_dir / "temporal_retention_by_lane.csv",
            temporal_retention_rows,
        )
        write_csv(
            metadata_dir / "temporal_retention_by_distance.csv",
            temporal_longitudinal_rows,
        )
        status_path.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "output_dir": str(output_dir),
                    "generated_file_count_before_manifest": sum(
                        1 for path in output_dir.rglob("*") if path.is_file()
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        manifest = build_manifest(output_dir)
        (metadata_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps({"status": "complete", "output_dir": str(output_dir)}, ensure_ascii=False))
    except Exception as exc:
        status_path.write_text(
            json.dumps(
                {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
