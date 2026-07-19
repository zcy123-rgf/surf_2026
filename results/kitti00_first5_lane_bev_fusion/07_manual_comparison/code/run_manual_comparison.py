"""Fuse independent manual driving-boundary pseudo-labels and compare with CLRNet.

No lane detector is used to produce the manual annotations. The saved CLRNet
polylines are loaded only for the final method-agreement comparison.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
MANUAL_ROOT = HERE.parent
PACKAGE_ROOT = MANUAL_ROOT.parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
DATA_DIR = PACKAGE_ROOT / "01_original_frames"
CALIB_PATH = PACKAGE_ROOT / "metadata" / "calib.txt"
POSE_PATH = PACKAGE_ROOT / "metadata" / "poses_00_first5.txt"
CLRNET_JSON = MANUAL_ROOT / "data" / "clrnet_lanes_kitti_ordered.json"
OUTPUT_DIR = MANUAL_ROOT / "results"
AUDIT_DIR = MANUAL_ROOT / "audit"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from surf_bev.geometry import (  # noqa: E402
    compute_image_to_bev_homography,
    image_to_ground_ipm,
    load_kitti_calib,
    load_kitti_poses,
    transform_lane_points_by_pose,
)
from ransac_denoise import denoise_left_right  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def imread(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return image


def imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix or ".png"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise ValueError(f"Cannot encode {path}")
    encoded.tofile(path)


def resample_polyline(points: np.ndarray, count: int = 64) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(distances)])
    samples = np.linspace(0.0, cumulative[-1], count)
    return np.column_stack(
        [np.interp(samples, cumulative, points[:, axis]) for axis in range(2)]
    )


def bottom_x(lane: np.ndarray) -> float:
    lane = np.asarray(lane, dtype=np.float64).reshape(-1, 2)
    threshold = np.quantile(lane[:, 1], 0.85)
    return float(np.median(lane[lane[:, 1] >= threshold, 0]))


def select_outer_two(lanes: list[np.ndarray]) -> list[np.ndarray]:
    ordered = sorted(lanes, key=bottom_x)
    if len(ordered) < 2:
        raise ValueError("Expected at least two CLRNet lanes")
    return [ordered[0], ordered[-1]]


def metric_pixels(points: np.ndarray, shape=(800, 800)) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    h, w = shape
    valid = (
        (points[:, 0] >= -10.0)
        & (points[:, 0] <= 10.0)
        & (points[:, 1] >= 3.0)
        & (points[:, 1] <= 50.0)
    )
    points = points[valid]
    u = (points[:, 0] + 10.0) / 20.0 * (w - 1)
    v = (50.0 - points[:, 1]) / 47.0 * (h - 1)
    return np.rint(np.column_stack([u, v])).astype(np.int32)


def add_bev_axes(image: np.ndarray) -> np.ndarray:
    output = image.copy()
    h, w = output.shape[:2]
    cv2.line(output, (w // 2, 0), (w // 2, h - 1), (150, 150, 150), 1)
    cv2.putText(output, "X: -10 m", (8, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(output, "X: +10 m", (w - 105, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(output, "Z: 50 m", (w // 2 + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(output, "Z: 3 m", (w // 2 + 8, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return output


def lane_mask(lanes: list[np.ndarray], thickness: int = 3) -> np.ndarray:
    mask = np.zeros((800, 800), dtype=np.uint8)
    for lane in lanes:
        pixels = metric_pixels(lane)
        if len(pixels) >= 2:
            cv2.polylines(mask, [pixels], False, 255, thickness, cv2.LINE_AA)
    return mask


def weighted_fusion(frames: list[list[np.ndarray]], weights: np.ndarray) -> dict:
    masks = [lane_mask(lanes) for lanes in frames]
    accumulation = np.zeros((800, 800), dtype=np.float32)
    for mask, weight in zip(masks, weights):
        accumulation += mask.astype(np.float32) / 255.0 * float(weight)
    score = np.clip(accumulation / float(np.sum(weights)), 0.0, 1.0)
    intensity = np.rint(score * 255.0).astype(np.uint8)
    heatmap = cv2.applyColorMap(intensity, cv2.COLORMAP_TURBO)
    heatmap[intensity == 0] = 0
    hits = np.stack([mask > 0 for mask in masks]).sum(axis=0)
    union = int(np.count_nonzero(hits))
    overlap = int(np.count_nonzero(hits >= 2))
    binary = np.where(score >= 0.08, 255, 0).astype(np.uint8)
    return {
        "heatmap": heatmap,
        "binary": binary,
        "union_pixels": union,
        "overlap_pixels": overlap,
        "overlap_fraction": float(overlap / union) if union else 0.0,
        "thresholded_pixels": int(np.count_nonzero(binary)),
    }


def point_membership(points: np.ndarray, selected: np.ndarray) -> np.ndarray:
    keys = {tuple(row) for row in np.round(selected, 9)}
    return np.asarray([tuple(row) in keys for row in np.round(points, 9)], dtype=bool)


def split_runs(points: np.ndarray, keep: np.ndarray) -> list[np.ndarray]:
    runs: list[np.ndarray] = []
    start = None
    for index, value in enumerate(keep):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(keep) - 1):
            end = index + 1 if value and index == len(keep) - 1 else index
            if end - start >= 2:
                runs.append(points[start:end])
            start = None
    return runs


def draw_annotation(image: np.ndarray, anchors: list[np.ndarray], dense: list[np.ndarray]) -> np.ndarray:
    output = image.copy()
    colors = [(255, 210, 0), (0, 255, 255)]
    labels = ["manual left", "manual right"]
    for color, label, anchor, lane in zip(colors, labels, anchors, dense):
        pixels = np.rint(lane).astype(np.int32)
        cv2.polylines(output, [pixels], False, color, 5, cv2.LINE_AA)
        for point in np.rint(anchor).astype(np.int32):
            cv2.circle(output, tuple(point), 5, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(output, tuple(point), 6, (255, 255, 255), 1, cv2.LINE_AA)
        text_point = tuple(pixels[min(12, len(pixels) - 1)])
        cv2.putText(output, label, text_point, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(output, label, text_point, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return output


def render_frame_bev(lanes: list[np.ndarray]) -> np.ndarray:
    image = np.zeros((800, 800, 3), dtype=np.uint8)
    for lane, color in zip(lanes, [(255, 210, 0), (0, 255, 255)]):
        pixels = metric_pixels(lane)
        if len(pixels) >= 2:
            cv2.polylines(image, [pixels], False, color, 4, cv2.LINE_AA)
    return add_bev_axes(image)


def plot_metric_fusion(
    axis: plt.Axes,
    frames: list[list[np.ndarray]],
    title: str,
    point_count: int,
    overlap_fraction: float,
) -> None:
    """Render aligned lane points in the white metric-axis style used previously."""
    frame_colors = ["#1f77ff", "#16ad73", "#ffb000", "#a43df0", "#ff4057"]
    for frame_id, (lanes, color) in enumerate(zip(frames, frame_colors)):
        first_segment = True
        for lane in lanes:
            lane = np.asarray(lane, dtype=np.float64).reshape(-1, 2)
            if len(lane) < 2:
                continue
            axis.plot(
                lane[:, 0],
                lane[:, 1],
                color=color,
                linewidth=1.8,
                marker="o",
                markersize=1.7,
                label=f"frame {frame_id}" if first_segment else None,
            )
            first_segment = False

    axis.scatter([0.0], [0.0], marker="x", s=70, linewidths=2.2, color="#e53935", zorder=5)
    axis.annotate("参考车", (0.0, 0.0), xytext=(6, 5), textcoords="offset points", fontsize=8, color="#555555")
    axis.set_xlim(-10.0, 10.0)
    axis.set_ylim(-2.0, 50.0)
    axis.set_xticks([-10.0, -5.0, 0.0, 5.0, 10.0])
    axis.set_yticks([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    axis.set_xlabel("X：参考帧右向 [m]")
    axis.set_ylabel("Z：参考帧前向 [m]")
    axis.grid(True, color="#d8d8d8", linewidth=0.7, alpha=0.8)
    axis.set_facecolor("white")
    axis.set_box_aspect(2.15)
    axis.set_title(
        f"{title}\n{point_count} 点｜多帧重叠 {overlap_fraction * 100:.1f}%",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    axis.legend(loc="upper right", fontsize=7, frameon=True, ncol=1)


def process_method(image_lanes: list[list[np.ndarray]], calib: dict, poses: list[np.ndarray]) -> dict:
    reference_pose = poses[4]
    projected_frames = []
    aligned_frames = []
    flat_points: list[list[float]] = []
    ranges = []
    for frame_id, lanes in enumerate(image_lanes):
        projected_lanes = []
        aligned_lanes = []
        for side, lane in zip(("left", "right"), lanes):
            ground = image_to_ground_ipm(lane, calib["K"], camera_height=1.65, pitch_deg=0.0, z_range=(3.0, 50.0))
            projected_lanes.append(ground)
            aligned = transform_lane_points_by_pose(
                ground,
                poses[frame_id],
                reference_pose,
                camera_height=1.65,
                pitch_deg=0.0,
            )
            valid = (
                (aligned[:, 0] >= -10.0)
                & (aligned[:, 0] <= 10.0)
                & (aligned[:, 1] >= 3.0)
                & (aligned[:, 1] <= 50.0)
            )
            aligned = aligned[valid]
            start = len(flat_points)
            flat_points.extend(aligned.tolist())
            ranges.append((frame_id, side, start, len(flat_points)))
            aligned_lanes.append(aligned)
        projected_frames.append(projected_lanes)
        aligned_frames.append(aligned_lanes)

    all_points = np.asarray(flat_points, dtype=np.float64).reshape(-1, 2)
    denoise = denoise_left_right(all_points, max_iter=100, threshold=0.3, seed=20260717)
    arrays = [array for array in (denoise["left_inliers"], denoise["right_inliers"]) if len(array)]
    kept = np.unique(np.round(np.vstack(arrays), 9), axis=0) if arrays else np.empty((0, 2))
    keep_all = point_membership(all_points, kept)
    denoised_frames: list[list[np.ndarray]] = [[] for _ in aligned_frames]
    per_lane = []
    for frame_id, side, start, end in ranges:
        lane = all_points[start:end]
        mask = keep_all[start:end]
        denoised_frames[frame_id].extend(split_runs(lane, mask))
        per_lane.append({"frame_id": frame_id, "side": side, "raw": len(lane), "kept": int(np.count_nonzero(mask))})

    weights = np.linspace(0.2, 1.0, 5)
    raw_fusion = weighted_fusion(aligned_frames, weights)
    den_fusion = weighted_fusion(denoised_frames, weights)
    return {
        "projected_frames": projected_frames,
        "aligned_frames": aligned_frames,
        "denoised_frames": denoised_frames,
        "raw_points": len(all_points),
        "kept_points": len(kept),
        "per_lane": per_lane,
        "raw_fusion": raw_fusion,
        "den_fusion": den_fusion,
    }


def x_at_y(lane: np.ndarray, levels: np.ndarray) -> np.ndarray:
    order = np.argsort(lane[:, 1])
    return np.interp(levels, lane[order, 1], lane[order, 0])


def image_agreement(manual: list[list[np.ndarray]], clrnet: list[list[np.ndarray]]) -> dict:
    records = []
    all_differences = []
    for frame_id, (manual_frame, clrnet_frame) in enumerate(zip(manual, clrnet)):
        for side, manual_lane, clrnet_lane in zip(("left", "right"), manual_frame, clrnet_frame):
            low = max(float(np.min(manual_lane[:, 1])), float(np.min(clrnet_lane[:, 1])), 200.0)
            high = min(float(np.max(manual_lane[:, 1])), float(np.max(clrnet_lane[:, 1])), 370.0)
            levels = np.linspace(low, high, 18)
            differences = np.abs(x_at_y(manual_lane, levels) - x_at_y(clrnet_lane, levels))
            all_differences.extend(differences.tolist())
            records.append(
                {
                    "frame_id": frame_id,
                    "side": side,
                    "mean_abs_x_difference_px": float(np.mean(differences)),
                    "max_abs_x_difference_px": float(np.max(differences)),
                }
            )
    return {
        "meaning": "Image-space method agreement only; not accuracy against ground truth.",
        "overall_mean_abs_x_difference_px": float(np.mean(all_differences)),
        "overall_max_abs_x_difference_px": float(np.max(all_differences)),
        "per_frame_side": records,
    }


def binary_iou(first: np.ndarray, second: np.ndarray) -> float:
    first = first > 0
    second = second > 0
    union = np.count_nonzero(first | second)
    return float(np.count_nonzero(first & second) / union) if union else 0.0


def symmetric_tolerance_overlap(first: np.ndarray, second: np.ndarray, radius_px: int) -> float:
    first = (first > 0).astype(np.uint8)
    second = (second > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1))
    first_near_second = np.count_nonzero(first & cv2.dilate(second, kernel)) / max(np.count_nonzero(first), 1)
    second_near_first = np.count_nonzero(second & cv2.dilate(first, kernel)) / max(np.count_nonzero(second), 1)
    return float((first_near_second + second_near_first) / 2.0)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    manual_record = json.loads((MANUAL_ROOT / "data" / "manual_annotations.json").read_text(encoding="utf-8"))
    clrnet_record = json.loads(CLRNET_JSON.read_text(encoding="utf-8"))
    calib = load_kitti_calib(CALIB_PATH)
    poses = load_kitti_poses(POSE_PATH)

    manual_anchors = []
    manual_lanes = []
    overlays = []
    clrnet_lanes = []
    for frame_id, frame in enumerate(manual_record["frames_xy"]):
        anchors = [np.asarray(frame["left"], dtype=np.float64), np.asarray(frame["right"], dtype=np.float64)]
        dense = [resample_polyline(lane, 64) for lane in anchors]
        manual_anchors.append(anchors)
        manual_lanes.append(dense)
        image = imread(DATA_DIR / f"frame_{frame_id:06d}.png")
        overlay = draw_annotation(image, anchors, dense)
        overlays.append(overlay)
        imwrite(OUTPUT_DIR / f"frame_{frame_id:06d}_manual_annotation.png", overlay)
        detected = [np.asarray(lane, dtype=np.float64) for lane in clrnet_record["items"][frame_id]["lanes_image_xy_px"]]
        clrnet_lanes.append(select_outer_two(detected))

    manual_result = process_method(manual_lanes, calib, poses)
    clrnet_result = process_method(clrnet_lanes, calib, poses)

    for frame_id in range(5):
        imwrite(OUTPUT_DIR / f"frame_{frame_id:06d}_manual_bev.png", render_frame_bev(manual_result["projected_frames"][frame_id]))
        imwrite(OUTPUT_DIR / f"frame_{frame_id:06d}_manual_aligned_to_000004.png", render_frame_bev(manual_result["aligned_frames"][frame_id]))

    manual_raw_image = add_bev_axes(manual_result["raw_fusion"]["heatmap"])
    manual_den_image = add_bev_axes(manual_result["den_fusion"]["heatmap"])
    clrnet_den_image = add_bev_axes(clrnet_result["den_fusion"]["heatmap"])
    imwrite(OUTPUT_DIR / "02_manual_fusion_without_denoise.png", manual_raw_image)
    imwrite(OUTPUT_DIR / "03_manual_fusion_with_ransac.png", manual_den_image)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    figure, axes = plt.subplots(1, 5, figsize=(18, 4.4), dpi=180)
    for frame_id, axis in enumerate(axes):
        axis.imshow(cv2.cvtColor(overlays[frame_id], cv2.COLOR_BGR2RGB))
        axis.set_title(f"KITTI 00 / {frame_id:06d}", fontsize=10)
        axis.axis("off")
    figure.suptitle("人工逐点标注：左右可见行驶边界（红点=人工锚点，青/黄=插值边界）", fontsize=17, fontweight="bold")
    figure.text(0.5, 0.015, "道路无连续喷涂车道线；右侧受停放车辆遮挡的区段为人工插值，整套标注不是车道线真值。", ha="center", fontsize=10)
    figure.tight_layout(rect=(0.01, 0.06, 0.99, 0.90))
    figure.savefig(OUTPUT_DIR / "01_manual_annotations_five_frames.png", bbox_inches="tight", facecolor="white")
    plt.close(figure)

    agreement = image_agreement(manual_lanes, clrnet_lanes)
    raw_iou = binary_iou(manual_result["raw_fusion"]["binary"], clrnet_result["raw_fusion"]["binary"])
    den_iou = binary_iou(manual_result["den_fusion"]["binary"], clrnet_result["den_fusion"]["binary"])
    raw_tolerance_overlap = symmetric_tolerance_overlap(
        manual_result["raw_fusion"]["binary"], clrnet_result["raw_fusion"]["binary"], radius_px=12
    )
    den_tolerance_overlap = symmetric_tolerance_overlap(
        manual_result["den_fusion"]["binary"], clrnet_result["den_fusion"]["binary"], radius_px=12
    )

    figure, axes = plt.subplots(1, 3, figsize=(14.8, 8.3), dpi=180)
    plot_metric_fusion(
        axes[0],
        manual_result["aligned_frames"],
        "人工标注融合：未去噪",
        manual_result["raw_points"],
        manual_result["raw_fusion"]["overlap_fraction"],
    )
    plot_metric_fusion(
        axes[1],
        manual_result["denoised_frames"],
        "人工标注融合：RANSAC 后",
        manual_result["kept_points"],
        manual_result["den_fusion"]["overlap_fraction"],
    )
    plot_metric_fusion(
        axes[2],
        clrnet_result["denoised_frames"],
        "CLRNet 融合：RANSAC 后",
        clrnet_result["kept_points"],
        clrnet_result["den_fusion"]["overlap_fraction"],
    )
    figure.suptitle("五帧位姿融合：人工边界标注与 CLRNet 对比", fontsize=17, fontweight="bold")
    figure.text(
        0.5,
        0.025,
        f"人工–CLRNet图像横向平均差={agreement['overall_mean_abs_x_difference_px']:.1f}px；"
        f"BEV±12px容差一致率：去噪前{raw_tolerance_overlap*100:.1f}%、去噪后{den_tolerance_overlap*100:.1f}%。"
        "这些是方法一致性，不是准确率。",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0.01, 0.075, 0.99, 0.92))
    comparison_path = OUTPUT_DIR / "04_manual_vs_clrnet_fusion_comparison.png"
    figure.savefig(comparison_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    audit = {
        "scope": "Manual-vs-CLRNet comparison appended to the result branch; not merged to main.",
        "manual_annotation": manual_record,
        "parameters": {
            "camera_height_m": 1.65,
            "pitch_deg": 0.0,
            "bev_size_hw_px": [800, 800],
            "x_range_m": [-10.0, 10.0],
            "z_range_m": [3.0, 50.0],
            "reference_frame": 4,
            "weights": [0.2, 0.4, 0.6, 0.8, 1.0],
            "ransac_iterations": 100,
            "ransac_threshold_m": 0.3,
            "ransac_seed": 20260717,
            "fixed_H_image_to_bev_pixels": compute_image_to_bev_homography(
                calib["K"], camera_height=1.65, pitch_deg=0.0, bev_size=(800, 800), x_range=(-10.0, 10.0), z_range=(3.0, 50.0)
            ).tolist(),
        },
        "manual_fusion": {
            "raw_points": manual_result["raw_points"],
            "kept_points": manual_result["kept_points"],
            "without_denoise": {key: manual_result["raw_fusion"][key] for key in ("union_pixels", "overlap_pixels", "overlap_fraction", "thresholded_pixels")},
            "after_ransac": {key: manual_result["den_fusion"][key] for key in ("union_pixels", "overlap_pixels", "overlap_fraction", "thresholded_pixels")},
            "per_lane_retention": manual_result["per_lane"],
        },
        "clrnet_recomputed_same_pipeline": {
            "raw_points": clrnet_result["raw_points"],
            "kept_points": clrnet_result["kept_points"],
            "without_denoise": {key: clrnet_result["raw_fusion"][key] for key in ("union_pixels", "overlap_pixels", "overlap_fraction", "thresholded_pixels")},
            "after_ransac": {key: clrnet_result["den_fusion"][key] for key in ("union_pixels", "overlap_pixels", "overlap_fraction", "thresholded_pixels")},
        },
        "method_agreement_not_accuracy": {
            "image_space": agreement,
            "fusion_binary_iou_without_denoise": raw_iou,
            "fusion_binary_iou_after_ransac": den_iou,
            "fusion_symmetric_tolerance_overlap_radius_px": 12,
            "fusion_symmetric_tolerance_overlap_without_denoise": raw_tolerance_overlap,
            "fusion_symmetric_tolerance_overlap_after_ransac": den_tolerance_overlap,
        },
        "input_sha256": {
            f"frame_{frame_id:06d}.png": sha256(DATA_DIR / f"frame_{frame_id:06d}.png")
            for frame_id in range(5)
        },
        "manual_annotations_sha256": sha256(MANUAL_ROOT / "data" / "manual_annotations.json"),
        "output_sha256": {path.name: sha256(path) for path in sorted(OUTPUT_DIR.glob("*.png"))},
    }
    (AUDIT_DIR / "manual_vs_clrnet_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit["manual_fusion"], ensure_ascii=False, indent=2))
    print(comparison_path)


if __name__ == "__main__":
    main()
