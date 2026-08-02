"""Audit CLRNet candidate counts before fitting two lane curves.

This scanner performs live inference but does not invent a missing lane.  It
selects only a contiguous subrange in which every frame has at least two CLRNet
candidates.  The selected length is a multiple of the requested segment size.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from surf_bev.detectors import CLRNetLaneDetector  # noqa: E402
from surf_bev.geometry import (  # noqa: E402
    image_to_ground_ipm,
    transform_lane_points_by_pose,
)


COLORS_BGR = [
    (255, 80, 80),
    (80, 220, 80),
    (80, 80, 255),
    (255, 180, 40),
    (220, 80, 220),
    (220, 220, 40),
]


def comma_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_range(value: str, option: str) -> tuple[float, float]:
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 2 or not parts[0] < parts[1]:
        raise ValueError(f"{option} requires two increasing comma-separated values.")
    return parts[0], parts[1]


def load_calibration(calib_path: Path) -> dict[str, np.ndarray]:
    entries: dict[str, np.ndarray] = {}
    with calib_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if ":" not in line:
                continue
            key, values = line.split(":", 1)
            entries[key.strip()] = np.fromstring(values, sep=" ", dtype=np.float64)
    if "P2" not in entries or entries["P2"].size != 12:
        raise ValueError(f"Calibration file has no valid P2 projection: {calib_path}")
    projection = entries["P2"].reshape(3, 4)
    intrinsic = projection[:, :3]
    image_from_cam0 = np.eye(4, dtype=np.float64)
    image_from_cam0[:3, 3] = np.linalg.solve(intrinsic, projection[:, 3])
    return {
        "K": intrinsic,
        "cam0_from_image": np.linalg.inv(image_from_cam0),
    }


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


def bottom_x(lane: np.ndarray) -> float:
    lane = np.asarray(lane, dtype=np.float64).reshape(-1, 2)
    threshold = np.quantile(lane[:, 1], 0.85)
    return float(np.median(lane[lane[:, 1] >= threshold, 0]))


def draw_candidates(image: np.ndarray, lanes: list[np.ndarray]) -> np.ndarray:
    output = image.copy()
    for lane_index, lane in enumerate(lanes):
        color = COLORS_BGR[lane_index % len(COLORS_BGR)]
        for point in np.asarray(lane, dtype=np.float64).reshape(-1, 2):
            if np.isfinite(point).all():
                cv2.circle(
                    output,
                    tuple(np.rint(point).astype(np.int32)),
                    3,
                    color,
                    -1,
                    cv2.LINE_AA,
                )
    return output


def eligible_runs(
    rows: list[dict[str, object]], minimum_candidates: int = 2
) -> list[list[int]]:
    """Return consecutive runs satisfying both candidate and metric gates."""

    runs: list[list[int]] = []
    current: list[int] = []
    for row in rows:
        frame_id = int(row["frame_id"])
        eligible = bool(
            row.get(
                "eligible_for_two_curve_fit",
                int(row["candidate_count"]) >= minimum_candidates,
            )
        )
        if eligible and (not current or frame_id == current[-1] + 1):
            current.append(frame_id)
        elif eligible:
            if current:
                runs.append(current)
            current = [frame_id]
        else:
            if current:
                runs.append(current)
                current = []
    if current:
        runs.append(current)
    return runs


def pose_window_stats(poses: np.ndarray, frame_ids: list[int]) -> dict[str, float]:
    selected = poses[np.asarray(frame_ids, dtype=np.int64)]
    positions = selected[:, :, 3][:, [0, 2]]
    rotations = selected[:, :, :3]
    forward = rotations[:, :, 2]
    yaw = np.unwrap(np.arctan2(forward[:, 0], forward[:, 2]))
    steps = np.diff(positions, axis=0)
    return {
        "path_length_m": float(np.linalg.norm(steps, axis=1).sum()),
        "displacement_m": float(np.linalg.norm(positions[-1] - positions[0])),
        "net_heading_change_deg": float(np.degrees(yaw[-1] - yaw[0])),
        "total_absolute_heading_change_deg": float(
            np.degrees(np.abs(np.diff(yaw)).sum())
        ),
    }


def select_window(
    rows: list[dict[str, object]],
    poses: np.ndarray,
    segment_size: int = 5,
    minimum_candidates: int = 2,
) -> dict[str, object]:
    """Choose the longest valid window, then the one with most heading change."""

    runs = eligible_runs(rows, minimum_candidates)
    maximum_length = max(
        ((len(run) // segment_size) * segment_size for run in runs), default=0
    )
    if maximum_length < segment_size:
        return {
            "status": "no_valid_run",
            "minimum_candidates_per_frame": minimum_candidates,
            "segment_size": segment_size,
            "eligible_runs": runs,
            "reason": (
                "No consecutive run has enough frames satisfying both gates: "
                "at least two CLRNet candidates and at least the required BEV "
                "points on each selected side. No missing lane was synthesized."
            ),
        }

    choices: list[dict[str, object]] = []
    for run in runs:
        if len(run) < maximum_length:
            continue
        for offset in range(len(run) - maximum_length + 1):
            frame_ids = run[offset : offset + maximum_length]
            stats = pose_window_stats(poses, frame_ids)
            choices.append({"frame_ids": frame_ids, **stats})

    selected = max(
        choices,
        key=lambda item: (
            float(item["total_absolute_heading_change_deg"]),
            abs(float(item["net_heading_change_deg"])),
            float(item["path_length_m"]),
        ),
    )
    frame_ids = list(selected["frame_ids"])
    return {
        "status": "selected",
        "minimum_candidates_per_frame": minimum_candidates,
        "segment_size": segment_size,
        "selection_rule": (
            "longest consecutive run whose every frame has >=2 candidates "
            "and enough valid metric BEV points on both selected sides; "
            "ties use total absolute pose heading change, then net heading "
            "change magnitude, then path length"
        ),
        "eligible_runs": runs,
        "selected_start": frame_ids[0],
        "selected_end": frame_ids[-1],
        "selected_length": len(frame_ids),
        **selected,
    }


def select_window_with_aligned_points(
    rows: list[dict[str, object]],
    poses_image: list[np.ndarray],
    ground_lanes: dict[int, list[np.ndarray]],
    segment_size: int,
    minimum_candidates: int,
    minimum_points_per_side: int,
    fusion_x_range: tuple[float, float],
    fusion_z_range: tuple[float, float],
    camera_height: float,
    pitch_deg: float,
) -> dict[str, object]:
    """Select a window using the exact post-pose metric point-count gate."""

    runs = eligible_runs(rows, minimum_candidates)
    max_possible = max(
        ((len(run) // segment_size) * segment_size for run in runs), default=0
    )
    rejected_windows = 0
    for window_length in range(max_possible, segment_size - 1, -segment_size):
        choices: list[dict[str, object]] = []
        for run in runs:
            if len(run) < window_length:
                continue
            for offset in range(len(run) - window_length + 1):
                frame_ids = run[offset : offset + window_length]
                reference_id = frame_ids[-1]
                reference_pose = poses_image[reference_id]
                counts: dict[str, list[int]] = {}
                valid = True
                for frame_id in frame_ids:
                    side_counts = []
                    for lane in ground_lanes[frame_id]:
                        aligned = transform_lane_points_by_pose(
                            lane,
                            poses_image[frame_id],
                            reference_pose,
                            camera_height=camera_height,
                            pitch_deg=pitch_deg,
                        )
                        keep = (
                            np.isfinite(aligned).all(axis=1)
                            & (aligned[:, 0] >= fusion_x_range[0])
                            & (aligned[:, 0] <= fusion_x_range[1])
                            & (aligned[:, 1] >= fusion_z_range[0])
                            & (aligned[:, 1] <= fusion_z_range[1])
                        )
                        side_counts.append(int(np.count_nonzero(keep)))
                    counts[str(frame_id)] = side_counts
                    if len(side_counts) != 2 or min(side_counts) < minimum_points_per_side:
                        valid = False
                if not valid:
                    rejected_windows += 1
                    continue
                stats = pose_window_stats(
                    np.asarray([pose[:3, :] for pose in poses_image]), frame_ids
                )
                choices.append(
                    {
                        "frame_ids": frame_ids,
                        "aligned_points_per_side_by_frame": counts,
                        **stats,
                    }
                )
        if choices:
            selected = max(
                choices,
                key=lambda item: (
                    float(item["total_absolute_heading_change_deg"]),
                    abs(float(item["net_heading_change_deg"])),
                    float(item["path_length_m"]),
                ),
            )
            frame_ids = list(selected["frame_ids"])
            return {
                "status": "selected",
                "minimum_candidates_per_frame": minimum_candidates,
                "minimum_aligned_points_per_side": minimum_points_per_side,
                "segment_size": segment_size,
                "selection_rule": (
                    "longest consecutive window passing live CLRNet count and "
                    "exact post-IPM/post-pose fusion-range point-count gates; "
                    "ties use pose heading change and path length"
                ),
                "eligible_runs_before_aligned_gate": runs,
                "rejected_windows_by_aligned_gate": rejected_windows,
                "selected_start": frame_ids[0],
                "selected_end": frame_ids[-1],
                "selected_length": len(frame_ids),
                **selected,
            }
    return {
        "status": "no_valid_run",
        "minimum_candidates_per_frame": minimum_candidates,
        "minimum_aligned_points_per_side": minimum_points_per_side,
        "segment_size": segment_size,
        "eligible_runs_before_aligned_gate": runs,
        "rejected_windows_by_aligned_gate": rejected_windows,
        "reason": (
            "No consecutive window passes both live CLRNet and exact post-pose "
            "metric point-count gates. No lane or point was synthesized."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--image-pattern", default="{frame_id:06d}.png")
    parser.add_argument("--frame-ids", required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--segment-size", type=int, default=5)
    parser.add_argument("--minimum-candidates", type=int, default=2)
    parser.add_argument("--minimum-bev-points-per-side", type=int, default=4)
    parser.add_argument("--camera-height", type=float, default=1.65)
    parser.add_argument("--pitch-deg", type=float, default=0.0)
    parser.add_argument("--local-z-range", default="3,50")
    parser.add_argument("--fusion-x-range", default="-20,20")
    parser.add_argument("--fusion-z-range", default="-20,50")
    parser.add_argument("--clrnet-root", type=Path, default=ROOT / "CLRNet")
    parser.add_argument(
        "--clrnet-config", default="configs/clrnet/clr_resnet18_culane.py"
    )
    parser.add_argument("--clrnet-checkpoint", default="weights/culane_r18.pth")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame_ids = comma_ints(args.frame_ids)
    local_z_range = parse_range(args.local_z_range, "--local-z-range")
    fusion_x_range = parse_range(args.fusion_x_range, "--fusion-x-range")
    fusion_z_range = parse_range(args.fusion_z_range, "--fusion-z-range")
    if not frame_ids or len(set(frame_ids)) != len(frame_ids):
        raise ValueError("Frame IDs must be non-empty and unique.")
    if (
        args.segment_size < 2
        or args.minimum_candidates < 2
        or args.minimum_bev_points_per_side < 2
    ):
        raise ValueError(
            "segment-size, minimum-candidates and minimum-bev-points-per-side "
            "must all be >= 2."
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory must be new or empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [
        args.image_dir / args.image_pattern.format(frame_id=frame_id)
        for frame_id in frame_ids
    ]
    missing = [
        str(path)
        for path in [*image_paths, args.calib, args.poses]
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing inputs: " + ", ".join(missing))

    pose_rows = np.loadtxt(args.poses, dtype=np.float64).reshape(-1, 3, 4)
    if max(frame_ids) >= len(pose_rows):
        raise ValueError("Pose file does not cover every requested frame ID.")

    calibration = load_calibration(args.calib)
    poses_image = []
    for row in pose_rows:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :] = row
        poses_image.append(pose @ calibration["cam0_from_image"])
    detector = CLRNetLaneDetector(
        clrnet_root=str(args.clrnet_root.resolve()),
        config=args.clrnet_config,
        checkpoint=args.clrnet_checkpoint,
        device=args.device,
    )
    (args.output_dir / "original_frames").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "all_candidates").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    ground_lanes: dict[int, list[np.ndarray]] = {}
    for frame_id, image_path in zip(frame_ids, image_paths):
        image = imread(image_path)
        lanes = detector.detect(image)["lanes"]
        selected_lanes: list[np.ndarray] = []
        if len(lanes) >= args.minimum_candidates:
            ordered = sorted(lanes, key=bottom_x)
            selected_lanes = [ordered[0], ordered[-1]]
        projected_counts = [
            len(
                image_to_ground_ipm(
                    lane,
                    calibration["K"],
                    camera_height=args.camera_height,
                    pitch_deg=args.pitch_deg,
                    z_range=local_z_range,
                )
            )
            for lane in selected_lanes
        ]
        metric_gate = (
            len(projected_counts) == 2
            and min(projected_counts) >= args.minimum_bev_points_per_side
        )
        ground_lanes[frame_id] = [
            image_to_ground_ipm(
                lane,
                calibration["K"],
                camera_height=args.camera_height,
                pitch_deg=args.pitch_deg,
                z_range=local_z_range,
            )
            for lane in selected_lanes
        ]
        shutil.copy2(
            image_path,
            args.output_dir / "original_frames" / f"frame_{frame_id:06d}.png",
        )
        imwrite(
            args.output_dir / "all_candidates" / f"frame_{frame_id:06d}.png",
            draw_candidates(image, lanes),
        )
        rows.append(
            {
                "frame_id": frame_id,
                "candidate_count": len(lanes),
                "candidate_point_counts": ";".join(str(len(lane)) for lane in lanes),
                "candidate_bottom_x_px": ";".join(
                    f"{bottom_x(lane):.6f}" for lane in lanes
                ),
                "selected_projected_point_counts": ";".join(
                    str(count) for count in projected_counts
                ),
                "candidate_count_gate": len(lanes) >= args.minimum_candidates,
                "metric_bev_point_gate": metric_gate,
                "eligible_for_two_curve_fit": metric_gate,
            }
        )
        print(
            f"frame {frame_id:06d}: {len(lanes)} candidate(s), "
            f"selected BEV counts={projected_counts}, eligible={metric_gate}",
            flush=True,
        )

    csv_path = args.output_dir / "lane_counts.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    recommendation = select_window_with_aligned_points(
        rows,
        poses_image,
        ground_lanes,
        segment_size=args.segment_size,
        minimum_candidates=args.minimum_candidates,
        minimum_points_per_side=args.minimum_bev_points_per_side,
        fusion_x_range=fusion_x_range,
        fusion_z_range=fusion_z_range,
        camera_height=args.camera_height,
        pitch_deg=args.pitch_deg,
    )
    report = {
        "status": "complete",
        "requested_frame_ids": frame_ids,
        "frame_count": len(frame_ids),
        "frames_with_at_least_two_candidates": sum(
            int(row["candidate_count"]) >= 2 for row in rows
        ),
        "frames_passing_metric_bev_gate": sum(
            bool(row["metric_bev_point_gate"]) for row in rows
        ),
        "counts": rows,
        "recommendation": recommendation,
        "warnings": [
            "Candidate count does not prove left/right lane-boundary identity.",
            "A selected image lane can still have too few valid points after metric IPM.",
            "Inspect original_frames and all_candidates before accepting a segment.",
            "No missing second lane is interpolated or synthesized by this scanner.",
        ],
    }
    (args.output_dir / "scan.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "recommendation.json").write_text(
        json.dumps(recommendation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(recommendation, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
