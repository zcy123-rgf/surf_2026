"""Run the pose-curvature diagnostic for every KITTI Odometry pose file.

This batch tool does not run CLRNet or read camera images. It only uses the
KITTI pose trajectories, so it is the appropriate first pass for all available
sequences before selecting lane-detection windows.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from analyze_pose_curvature import compute_curvature, load_kitti_poses, summarize, write_outputs


def discover_pose_files(poses_root: Path) -> list[tuple[str, Path]]:
    """Return numeric KITTI sequence files in deterministic sequence order."""
    files: list[tuple[str, Path]] = []
    for path in poses_root.glob("*.txt"):
        if path.stem.isdigit():
            files.append((f"{int(path.stem):02d}", path))
    if not files:
        raise FileNotFoundError(f"No numeric pose files (*.txt) found in {poses_root}")
    return sorted(files, key=lambda item: int(item[0]))


def write_summary(output_root: Path, rows: list[dict[str, object]]) -> None:
    """Write machine-readable summary files for comparison across sequences."""
    fields = [
        "sequence_id",
        "pose_file",
        "start_frame",
        "end_frame",
        "frame_count",
        "path_length_m",
        "mean_absolute_curvature_1_per_m",
        "median_absolute_curvature_1_per_m",
        "p90_absolute_curvature_1_per_m",
        "max_absolute_curvature_1_per_m",
        "integrated_absolute_curvature_rad",
        "total_absolute_heading_change_rad",
    ]
    with (output_root / "all_sequences_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "all_sequences_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )


def write_overview(output_root: Path, curves: list[tuple[str, dict]]) -> None:
    """Create one figure containing the absolute-curvature curve for each sequence."""
    columns = 2
    rows = (len(curves) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(14, max(5, rows * 3.0)), squeeze=False)
    axes_flat = axes.ravel()
    for axis, (sequence_id, curvature) in zip(axes_flat, curves):
        frames = curvature["frame_offset"]
        axis.plot(frames, curvature["absolute_curvature_1_per_m"], linewidth=1.0)
        axis.axhline(0.004, color="#2e7d32", linestyle="--", linewidth=0.8)
        axis.axhline(0.005, color="#c62828", linestyle="--", linewidth=0.8)
        axis.set_title(f"Sequence {sequence_id}")
        axis.set_xlabel("Frame")
        axis.set_ylabel("Absolute curvature [1/m]")
        axis.grid(alpha=0.25)
    for axis in axes_flat[len(curves) :]:
        axis.set_visible(False)
    figure.suptitle("KITTI Odometry: pose absolute curvature for all available sequences")
    figure.tight_layout()
    figure.savefig(output_root / "all_sequences_curvature_overview.png", dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoothing-window", type=int, default=11)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pose_files = discover_pose_files(args.poses_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    curves: list[tuple[str, dict]] = []

    for sequence_id, pose_path in pose_files:
        poses = load_kitti_poses(pose_path)
        end_frame = len(poses) - 1
        curvature = compute_curvature(poses, args.smoothing_window)
        summary = summarize(curvature, 0, end_frame)
        sequence_output = args.output_root / f"sequence_{sequence_id}"
        write_outputs(sequence_output, curvature, summary, 0)
        summary_rows.append({"sequence_id": sequence_id, "pose_file": str(pose_path), **summary})
        curves.append((sequence_id, curvature))
        print(
            f"Sequence {sequence_id}: {len(poses)} frames, "
            f"path {summary['path_length_m']:.2f} m, "
            f"median kappa {summary['median_absolute_curvature_1_per_m']:.6f} 1/m"
        )

    write_summary(args.output_root, summary_rows)
    write_overview(args.output_root, curves)
    print(json.dumps({"status": "complete", "sequence_count": len(pose_files), "output_root": str(args.output_root)}, indent=2))


if __name__ == "__main__":
    main()
