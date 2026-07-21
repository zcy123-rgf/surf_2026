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
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from surf_bev.detectors import CLRNetLaneDetector  # noqa: E402
from surf_bev.geometry import (  # noqa: E402
    image_to_ground_ipm,
    load_kitti_poses,
    transform_lane_points_by_pose,
)


FRAME_COLORS_BGR = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
]
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
    parser.add_argument("--x-range", default="-10,10")
    parser.add_argument("--z-range", default="3,50")
    parser.add_argument("--bev-size", type=int, default=800)
    parser.add_argument("--weights", default="0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--legacy-ransac-iterations", type=int, default=100)
    parser.add_argument("--legacy-ransac-threshold", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--clrnet-root", type=Path, default=ROOT / "CLRNet")
    parser.add_argument(
        "--clrnet-config", default="configs/clrnet/clr_resnet18_culane.py"
    )
    parser.add_argument(
        "--clrnet-checkpoint", default="weights/culane_r18.pth"
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
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


def parse_kitti_projection(calib_path: Path) -> tuple[np.ndarray, str]:
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
    return projection[:, :3], key


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
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.int32)
    valid = (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= x_range[0])
        & (points[:, 0] <= x_range[1])
        & (points[:, 1] >= z_range[0])
        & (points[:, 1] <= z_range[1])
    )
    points = points[valid]
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
            "window_m": float("nan"),
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
    x_values = comma_floats(args.x_range)
    z_values = comma_floats(args.z_range)
    if len(x_values) != 2 or len(z_values) != 2:
        raise ValueError("--x-range and --z-range require two comma-separated values.")
    x_range = (x_values[0], x_values[1])
    z_range = (z_values[0], z_values[1])
    if len(frame_ids) != len(weights):
        raise ValueError("The number of frame IDs must equal the number of weights.")
    if args.reference_id not in frame_ids:
        raise ValueError("Reference ID must be one of the requested frame IDs.")

    image_paths = [
        args.image_dir / args.image_pattern.format(frame_id=frame_id)
        for frame_id in frame_ids
    ]
    required = [*image_paths, args.calib, args.poses]
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

        intrinsic_k, projection_key = parse_kitti_projection(args.calib)
        poses = load_kitti_poses(args.poses)
        if max(frame_ids) >= len(poses):
            raise IndexError(
                f"Pose file contains {len(poses)} rows but frame {max(frame_ids)} was requested."
            )
        reference_pose = poses[args.reference_id]
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
            selected_records = []
            for side, (lane_index, lane) in zip(("left", "right"), selected):
                ground = image_to_ground_ipm(
                    lane,
                    intrinsic_k,
                    camera_height=args.camera_height,
                    pitch_deg=args.pitch_deg,
                    z_range=z_range,
                )
                aligned = transform_lane_points_by_pose(
                    ground,
                    poses[frame_id],
                    reference_pose,
                    camera_height=args.camera_height,
                    pitch_deg=args.pitch_deg,
                )
                valid = (
                    np.isfinite(aligned).all(axis=1)
                    & (aligned[:, 0] >= x_range[0])
                    & (aligned[:, 0] <= x_range[1])
                    & (aligned[:, 1] >= z_range[0])
                    & (aligned[:, 1] <= z_range[1])
                )
                aligned = aligned[valid]
                start = len(flat_points)
                flat_points.extend(aligned.tolist())
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
                selected_records.append(
                    {
                        "side": side,
                        "lane_index": lane_index,
                        "bottom_x_px": bottom_x(lane),
                        "image_points": int(len(lane)),
                        "projected_points": int(len(ground)),
                        "aligned_in_range_points": int(len(aligned)),
                    }
                )

            projected_frames.append(projected_lanes)
            aligned_frames.append(aligned_lanes)
            imwrite(
                output_dir / "03_bev_points" / f"frame_{frame_id:06d}.png",
                draw_metric_points(
                    projected_lanes,
                    LANE_COLORS_BGR,
                    args.bev_size,
                    x_range,
                    z_range,
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
                    x_range,
                    z_range,
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
                    "aligned_in_range_points": sum(len(lane) for lane in aligned_lanes),
                }
            )

        all_points = np.asarray(flat_points, dtype=np.float64).reshape(-1, 2)
        raw_fusion = weighted_raster_fusion(
            aligned_frames, weights, args.bev_size, x_range, z_range
        )
        save_fusion_stage(
            output_dir / "05_fusion_without_denoise",
            aligned_frames,
            raw_fusion,
            frame_ids,
            args.bev_size,
            x_range,
            z_range,
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
            clustered_frames, weights, args.bev_size, x_range, z_range
        )
        denoised_fusion = weighted_raster_fusion(
            denoised_frames, weights, args.bev_size, x_range, z_range
        )
        save_fusion_stage(
            output_dir / "06_legacy_ransac_diagnostic" / "01_after_x_cluster",
            clustered_frames,
            cluster_fusion,
            frame_ids,
            args.bev_size,
            x_range,
            z_range,
        )
        save_fusion_stage(
            output_dir / "06_legacy_ransac_diagnostic" / "02_after_line_ransac",
            denoised_frames,
            denoised_fusion,
            frame_ids,
            args.bev_size,
            x_range,
            z_range,
        )

        z_bins = [(3, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
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
                    "projection_key": projection_key,
                    "K": intrinsic_k.tolist(),
                },
                "poses": {
                    "path": str(args.poses.resolve()),
                    "sha256": sha256(args.poses),
                },
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
                "x_range_m": list(x_range),
                "z_range_m": list(z_range),
                "bev_size_px": [args.bev_size, args.bev_size],
            },
            "pose_alignment": {
                "formula": "p_ref = inv(T_w_ref) @ T_w_src @ p_src",
                "reference_frame": args.reference_id,
            },
            "fusion_without_denoise": {
                "point_count": int(len(all_points)),
                "weights": weights.tolist(),
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
        write_csv(metadata_dir / "frame_metrics.csv", frame_metrics)
        write_csv(metadata_dir / "legacy_retention_by_lane.csv", retention_rows)
        write_csv(
            metadata_dir / "legacy_retention_by_distance.csv", longitudinal_rows
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
