"""Export absolute pose curvature for a KITTI odometry frame interval.

This is a pose-only road-shape diagnostic. It does not measure lane accuracy
and it does not synthesize missing lane observations.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


def load_kitti_poses(path: Path) -> np.ndarray:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if len(values) != 12:
            raise ValueError(f"Expected 12 pose values at line {line_number}.")
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :] = np.asarray(values, dtype=np.float64).reshape(3, 4)
        rows.append(pose)
    if not rows:
        raise ValueError(f"No KITTI poses found in {path}.")
    return np.asarray(rows)


def choose_smoothing_window(frame_count: int, requested: int) -> int:
    if frame_count < 5:
        return 0
    window = min(requested, frame_count if frame_count % 2 else frame_count - 1)
    if window < 5:
        return 0
    return window


def compute_curvature(poses: np.ndarray, smoothing_window: int = 11) -> dict[str, np.ndarray]:
    positions = poses[:, :3, 3][:, [0, 2]]
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(steps)])
    if len(s) < 3 or np.any(np.diff(s) <= 1e-8):
        raise ValueError("Pose trajectory must contain at least three distinct positions.")

    window = choose_smoothing_window(len(s), smoothing_window)
    if window:
        smooth_x = savgol_filter(positions[:, 0], window, 2, mode="interp")
        smooth_z = savgol_filter(positions[:, 1], window, 2, mode="interp")
    else:
        smooth_x, smooth_z = positions[:, 0], positions[:, 1]

    dx = np.gradient(smooth_x, s)
    dz = np.gradient(smooth_z, s)
    ddx = np.gradient(dx, s)
    ddz = np.gradient(dz, s)
    speed_sq = np.maximum(dx * dx + dz * dz, 1e-12)
    signed = (dx * ddz - dz * ddx) / np.power(speed_sq, 1.5)
    absolute = np.abs(signed)
    heading = np.unwrap(np.arctan2(dx, dz))
    heading_change = np.diff(heading, prepend=heading[0])
    return {
        "frame_offset": np.arange(len(s), dtype=np.int64),
        "x_m": positions[:, 0],
        "z_m": positions[:, 1],
        "arc_length_m": s,
        "signed_curvature_1_per_m": signed,
        "absolute_curvature_1_per_m": absolute,
        "heading_rad": heading,
        "heading_change_rad": heading_change,
    }


def summarize(curvature: dict[str, np.ndarray], start_frame: int, end_frame: int) -> dict[str, float | int]:
    absolute = curvature["absolute_curvature_1_per_m"]
    ds = np.diff(curvature["arc_length_m"])
    return {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frame_count": int(len(absolute)),
        "path_length_m": float(curvature["arc_length_m"][-1]),
        "mean_absolute_curvature_1_per_m": float(np.mean(absolute)),
        "median_absolute_curvature_1_per_m": float(np.median(absolute)),
        "p90_absolute_curvature_1_per_m": float(np.percentile(absolute, 90)),
        "max_absolute_curvature_1_per_m": float(np.max(absolute)),
        "integrated_absolute_curvature_rad": float(np.sum(absolute[:-1] * ds)),
        "total_absolute_heading_change_rad": float(np.sum(np.abs(np.diff(curvature["heading_rad"])) )),
    }


def write_outputs(output_dir: Path, curvature: dict[str, np.ndarray], summary: dict[str, object], start_frame: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "pose_curvature.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_id", *curvature.keys()])
        for index in range(len(curvature["frame_offset"])):
            writer.writerow(
                [start_frame + index]
                + [curvature[key][index].item() for key in curvature]
            )
    (output_dir / "pose_curvature_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    plt.figure(figsize=(11, 5))
    frames = start_frame + curvature["frame_offset"]
    plt.plot(frames, curvature["absolute_curvature_1_per_m"], linewidth=2)
    plt.xlabel("Frame")
    plt.ylabel("Absolute curvature [1/m]")
    plt.title(f"Pose absolute curvature, frames {start_frame:06d}-{int(summary['end_frame']):06d}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "pose_absolute_curvature.png", dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoothing-window", type=int, default=11)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_frame < 0 or args.end_frame < args.start_frame:
        raise ValueError("Frame range is invalid.")
    all_poses = load_kitti_poses(args.poses)
    if args.end_frame >= len(all_poses):
        raise ValueError("Requested frame range exceeds the pose file.")
    curvature = compute_curvature(
        all_poses[args.start_frame : args.end_frame + 1], args.smoothing_window
    )
    summary = summarize(curvature, args.start_frame, args.end_frame)
    write_outputs(args.output_dir, curvature, summary, args.start_frame)
    print(json.dumps({"status": "complete", "output_dir": str(args.output_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()
