"""Audit SemanticKITTI lane-marking labels aligned with KITTI Odometry.

The default mode only reads label files and point-cloud file sizes.  Optional
point export loads Velodyne scans and transforms raw semantic id 60 points into
a selected KITTI camera-0 reference frame.  Existing result directories are
never reused.
"""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_THRESHOLDS = (1, 5, 10, 20, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit raw SemanticKITTI lane-marking labels against KITTI "
            "Odometry frame IDs without touching previous outputs."
        )
    )
    parser.add_argument("--velodyne-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--poses", type=Path)
    parser.add_argument("--calib", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lane-label", type=int, default=60)
    parser.add_argument("--expected-frames", type=int, default=4541)
    parser.add_argument("--reference-id", type=int, default=0)
    parser.add_argument(
        "--thresholds",
        default=",".join(str(item) for item in DEFAULT_THRESHOLDS),
        help="Diagnostic minimum lane-point counts used for contiguous runs.",
    )
    parser.add_argument(
        "--export-points",
        action="store_true",
        help=(
            "Load scans and export all raw semantic-id lane-marking points in "
            "the selected camera-0 reference frame."
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError("{} does not exist: {}".format(description, path))
    return path.resolve()


def require_directory(path: Path, description: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError("{} does not exist: {}".format(description, path))
    return path.resolve()


def require_empty_output(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(
            "Output directory already exists and is not empty: {}".format(resolved)
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def parse_thresholds(value: str) -> Tuple[int, ...]:
    values = tuple(sorted(set(int(item.strip()) for item in value.split(",") if item.strip())))
    if not values or values[0] < 1:
        raise ValueError("All diagnostic thresholds must be positive integers.")
    return values


def numeric_files(directory: Path, suffix: str) -> Dict[int, Path]:
    output = {}
    for path in directory.glob("*{}".format(suffix)):
        if path.stem.isdigit():
            frame_id = int(path.stem)
            if frame_id in output:
                raise ValueError("Duplicate frame ID {} in {}".format(frame_id, directory))
            output[frame_id] = path
    return output


def raw_semantic_ids(labels: np.ndarray) -> np.ndarray:
    """Return the lower 16-bit SemanticKITTI class before learning_map."""

    return np.asarray(labels, dtype=np.uint32) & np.uint32(0xFFFF)


def contiguous_runs(frame_ids: Iterable[int]) -> List[Tuple[int, int, int]]:
    values = sorted(set(int(item) for item in frame_ids))
    if not values:
        return []
    runs = []
    start = values[0]
    previous = values[0]
    for frame_id in values[1:]:
        if frame_id != previous + 1:
            runs.append((start, previous, previous - start + 1))
            start = frame_id
        previous = frame_id
    runs.append((start, previous, previous - start + 1))
    return runs


def parse_poses(path: Path) -> List[np.ndarray]:
    poses = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            values = np.fromstring(line, sep=" ", dtype=np.float64)
            if values.size != 12:
                raise ValueError(
                    "Pose row {} must contain 12 values: {}".format(line_number, path)
                )
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :4] = values.reshape(3, 4)
            poses.append(pose)
    return poses


def parse_velodyne_to_camera(path: Path) -> np.ndarray:
    values = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            values[key.strip()] = np.fromstring(raw, sep=" ", dtype=np.float64)
    key = next(
        (item for item in ("Tr", "Tr_velo_to_cam", "Tr_velo_cam") if item in values),
        None,
    )
    if key is None or values[key].size != 12:
        raise KeyError(
            "Calibration must contain a 3x4 Tr/Tr_velo_to_cam row: {}".format(path)
        )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :4] = values[key].reshape(3, 4)
    return transform


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def audit_frames(
    velodyne_files: Dict[int, Path],
    label_files: Dict[int, Path],
    lane_label: int,
    image_dir: Optional[Path],
) -> Tuple[List[Dict[str, object]], int, int]:
    velodyne_ids = set(velodyne_files)
    label_ids = set(label_files)
    if velodyne_ids != label_ids:
        missing_labels = sorted(velodyne_ids - label_ids)
        missing_scans = sorted(label_ids - velodyne_ids)
        raise ValueError(
            "Velodyne/label frame IDs differ. Missing labels: {}; missing scans: {}".format(
                missing_labels[:20], missing_scans[:20]
            )
        )

    rows = []
    total_points = 0
    total_lane_points = 0
    for frame_id in sorted(label_ids):
        label_path = label_files[frame_id]
        velodyne_path = velodyne_files[frame_id]
        if label_path.stat().st_size % 4 != 0:
            raise ValueError("Label file size is not divisible by 4: {}".format(label_path))
        if velodyne_path.stat().st_size % 16 != 0:
            raise ValueError("Velodyne file size is not divisible by 16: {}".format(velodyne_path))
        labels = np.fromfile(str(label_path), dtype=np.uint32)
        point_count = velodyne_path.stat().st_size // 16
        if len(labels) != point_count:
            raise ValueError(
                "Point/label count mismatch for frame {:06d}: {} versus {}".format(
                    frame_id, point_count, len(labels)
                )
            )
        semantic = raw_semantic_ids(labels)
        lane_points = int(np.count_nonzero(semantic == lane_label))
        image_exists = None
        if image_dir is not None:
            image_exists = (image_dir / "{:06d}.png".format(frame_id)).is_file()
        rows.append(
            {
                "frame_id": frame_id,
                "velodyne_points": int(point_count),
                "label_points": int(len(labels)),
                "lane_marking_points_raw_id_{}".format(lane_label): lane_points,
                "lane_marking_fraction": lane_points / float(point_count) if point_count else 0.0,
                "image_2_exists": image_exists,
            }
        )
        total_points += int(point_count)
        total_lane_points += lane_points
    return rows, total_points, total_lane_points


def export_reference_points(
    rows: Sequence[Dict[str, object]],
    velodyne_files: Dict[int, Path],
    label_files: Dict[int, Path],
    poses: Sequence[np.ndarray],
    transform_cam_velo: np.ndarray,
    reference_id: int,
    lane_label: int,
    output_path: Path,
) -> int:
    if reference_id < 0 or reference_id >= len(poses):
        raise IndexError("Reference pose {} is unavailable.".format(reference_id))
    inverse_reference = np.linalg.inv(poses[reference_id])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported = 0
    fieldnames = [
        "frame_id",
        "point_index",
        "velo_x_forward_m",
        "velo_y_left_m",
        "velo_z_up_m",
        "remission",
        "ref_x_right_m",
        "ref_y_down_m",
        "ref_z_forward_m",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            frame_id = int(row["frame_id"])
            if frame_id >= len(poses):
                raise IndexError("Pose is unavailable for frame {}.".format(frame_id))
            labels = np.fromfile(str(label_files[frame_id]), dtype=np.uint32)
            mask = raw_semantic_ids(labels) == lane_label
            if not np.any(mask):
                continue
            scan = np.fromfile(str(velodyne_files[frame_id]), dtype=np.float32).reshape(-1, 4)
            selected_indexes = np.flatnonzero(mask)
            selected = scan[mask]
            points_velo_h = np.column_stack(
                [selected[:, :3].astype(np.float64), np.ones(len(selected), dtype=np.float64)]
            )
            transform_ref_velo = inverse_reference @ poses[frame_id] @ transform_cam_velo
            points_ref = (transform_ref_velo @ points_velo_h.T).T[:, :3]
            for index, velo, ref in zip(selected_indexes, selected, points_ref):
                writer.writerow(
                    {
                        "frame_id": frame_id,
                        "point_index": int(index),
                        "velo_x_forward_m": float(velo[0]),
                        "velo_y_left_m": float(velo[1]),
                        "velo_z_up_m": float(velo[2]),
                        "remission": float(velo[3]),
                        "ref_x_right_m": float(ref[0]),
                        "ref_y_down_m": float(ref[1]),
                        "ref_z_forward_m": float(ref[2]),
                    }
                )
            exported += len(selected)
    return exported


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    velodyne_dir = require_directory(args.velodyne_dir, "Velodyne directory")
    label_dir = require_directory(args.label_dir, "SemanticKITTI label directory")
    image_dir = (
        require_directory(args.image_dir, "image_2 directory")
        if args.image_dir is not None
        else None
    )
    if not 0 <= args.lane_label <= 0xFFFF:
        raise ValueError("Raw semantic label must fit in the lower 16 bits.")

    velodyne_files = numeric_files(velodyne_dir, ".bin")
    label_files = numeric_files(label_dir, ".label")
    if not velodyne_files or not label_files:
        raise FileNotFoundError("No numeric .bin/.label files were found.")

    rows, total_points, total_lane_points = audit_frames(
        velodyne_files, label_files, args.lane_label, image_dir
    )
    if args.expected_frames > 0 and len(rows) != args.expected_frames:
        raise ValueError(
            "Expected {} aligned frames, found {}.".format(args.expected_frames, len(rows))
        )
    if image_dir is not None:
        missing_images = [row["frame_id"] for row in rows if not row["image_2_exists"]]
        if missing_images:
            raise ValueError("Missing image_2 frames: {}".format(missing_images[:20]))

    poses = None
    transform_cam_velo = None
    if args.export_points:
        if args.poses is None or args.calib is None:
            raise ValueError("--export-points requires --poses and --calib.")
        poses_path = require_file(args.poses, "Pose file")
        calib_path = require_file(args.calib, "Calibration file")
        poses = parse_poses(poses_path)
        transform_cam_velo = parse_velodyne_to_camera(calib_path)
        if len(poses) < len(rows):
            raise ValueError(
                "Pose file has {} rows for {} aligned frames.".format(len(poses), len(rows))
            )

    output_dir = require_empty_output(args.output_dir)
    frame_count_field = "lane_marking_points_raw_id_{}".format(args.lane_label)
    write_csv(
        output_dir / "frame_lane_marking_counts.csv",
        rows,
        [
            "frame_id",
            "velodyne_points",
            "label_points",
            frame_count_field,
            "lane_marking_fraction",
            "image_2_exists",
        ],
    )

    run_rows = []
    threshold_summary = {}
    for threshold in thresholds:
        qualifying = [
            int(row["frame_id"])
            for row in rows
            if int(row[frame_count_field]) >= threshold
        ]
        runs = contiguous_runs(qualifying)
        for start, end, length in runs:
            run_rows.append(
                {
                    "minimum_lane_points": threshold,
                    "start_frame": start,
                    "end_frame": end,
                    "frame_count": length,
                }
            )
        threshold_summary[str(threshold)] = {
            "qualifying_frames": len(qualifying),
            "qualifying_fraction": len(qualifying) / float(len(rows)),
            "contiguous_run_count": len(runs),
            "longest_contiguous_run_frames": max((item[2] for item in runs), default=0),
        }
    write_csv(
        output_dir / "contiguous_lane_marking_runs.csv",
        run_rows,
        ["minimum_lane_points", "start_frame", "end_frame", "frame_count"],
    )

    exported_points = 0
    if args.export_points:
        exported_points = export_reference_points(
            rows,
            velodyne_files,
            label_files,
            poses,
            transform_cam_velo,
            args.reference_id,
            args.lane_label,
            output_dir / "lane_marking_points_reference_camera.csv",
        )

    nonzero_frames = sum(int(row[frame_count_field]) > 0 for row in rows)
    summary = {
        "status": "complete",
        "claim_scope": (
            "raw SemanticKITTI semantic-id lane-marking points; not pre-associated "
            "left/right lane-boundary polylines"
        ),
        "inputs": {
            "velodyne_dir": str(velodyne_dir),
            "label_dir": str(label_dir),
            "image_dir": str(image_dir) if image_dir is not None else None,
            "poses": str(args.poses.resolve()) if args.poses is not None else None,
            "calib": str(args.calib.resolve()) if args.calib is not None else None,
        },
        "raw_lane_semantic_id": args.lane_label,
        "aligned_frames": len(rows),
        "first_frame": int(rows[0]["frame_id"]),
        "last_frame": int(rows[-1]["frame_id"]),
        "frames_with_at_least_one_lane_point": nonzero_frames,
        "fraction_with_at_least_one_lane_point": nonzero_frames / float(len(rows)),
        "total_velodyne_points": total_points,
        "total_lane_marking_points": total_lane_points,
        "lane_marking_point_fraction": total_lane_points / float(total_points),
        "diagnostic_thresholds": threshold_summary,
        "point_export": {
            "enabled": bool(args.export_points),
            "reference_frame": args.reference_id if args.export_points else None,
            "exported_points": exported_points,
        },
        "warnings": [
            "A label file per frame does not imply enough lane-marking points per frame.",
            "Raw semantic id 60 has no ego-left/ego-right instance identity.",
            "Do not call these sparse points two continuous ground-truth lane curves.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    readme_lines = [
        "# SemanticKITTI Sequence 00 lane-marking audit",
        "",
        "This directory was generated from raw semantic ID `{}`.".format(args.lane_label),
        "It does not contain pre-associated left/right ground-truth curves.",
        "",
        "- Aligned frames: {}".format(len(rows)),
        "- Frames with at least one lane-marking point: {} ({:.2%})".format(
            nonzero_frames, nonzero_frames / float(len(rows))
        ),
        "- Total lane-marking points: {}".format(total_lane_points),
        "- Point coordinates exported: {}".format(args.export_points),
        "",
        "See `summary.json`, `frame_lane_marking_counts.csv`, and "
        "`contiguous_lane_marking_runs.csv` for auditable values.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
