"""Shared input preparation for the separate extended-range curve models.

This module deliberately reuses the already reviewed pose/Frenet conversion
from ``run_weekly_lane_hierarchy.py``.  It does not define a curve model; the
polynomial and B-spline algorithms remain in separate executable files.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import fit_first5_two_curves as lane_io  # noqa: E402
from scripts import run_weekly_lane_hierarchy as weekly  # noqa: E402
from surf_bev.geometry import load_kitti_poses  # noqa: E402


@dataclass
class ExtendedCurveInput:
    selection: dict[str, object]
    start_frame: int
    end_frame: int
    reference_frame: int
    reference_path: weekly.FrenetPath
    frames: list[dict[str, object]]
    groups: list[dict[str, object]]
    frame_audit: dict[int, dict[str, object]]
    points_path: Path


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-height", type=float, default=1.65)
    parser.add_argument("--pitch-deg", type=float, default=0.0)
    parser.add_argument("--bin-size-m", type=float, default=0.50)
    parser.add_argument("--huber-delta-m", type=float, default=0.20)
    parser.add_argument("--irls-iterations", type=int, default=4)
    parser.add_argument("--curve-samples", type=int, default=600)
    parser.add_argument("--maximum-cv-folds", type=int, default=20)


def validate_common_arguments(args: argparse.Namespace) -> None:
    required = [args.selection_json, args.poses, args.calib]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs: " + ", ".join(missing))
    if args.bin_size_m <= 0 or args.huber_delta_m <= 0:
        raise ValueError("bin-size-m and huber-delta-m must be positive.")
    if args.irls_iterations < 1 or args.curve_samples < 20:
        raise ValueError("irls-iterations must be >=1 and curve-samples >=20.")
    if args.maximum_cv_folds < 2:
        raise ValueError("maximum-cv-folds must be at least two.")
    weekly.require_empty_output(args.output_dir)


def load_input(args: argparse.Namespace) -> ExtendedCurveInput:
    """Load one ranked option without changing the established coordinates."""

    validate_common_arguments(args)
    selection = weekly.load_json(args.selection_json)
    if selection.get("status") != "selected":
        raise ValueError(
            "Selected option did not meet the registered block-coverage gate: "
            f"{selection.get('status')}. Review candidate_options.csv or lower "
            "the gate explicitly in a new run."
        )

    selected = selection["selected"]
    start = int(selected["start_frame"])
    end = int(selected["end_frame"])
    reference_id = end

    calibration = lane_io.parse_projection(args.calib)
    poses_cam0 = load_kitti_poses(args.poses)
    if start < 0 or end >= len(poses_cam0) or end < start:
        raise ValueError(
            f"Selected interval {start}-{end} is outside the pose file."
        )
    poses_image = [pose @ calibration["cam0_from_image"] for pose in poses_cam0]
    frame_ids = list(range(start, end + 1))
    reference_path = weekly.build_reference_path(
        poses_image, frame_ids, reference_id
    )
    frames, frame_audit, points_path = weekly.load_aligned_frames(
        selection,
        poses_image,
        reference_id,
        reference_path,
        args.camera_height,
        args.pitch_deg,
    )
    groups = [
        {
            "group_id": int(frame["frame_id"]),
            "lanes_sd": frame["lanes_sd"],
        }
        for frame in frames
    ]
    if len(groups) < 10:
        raise ValueError(
            f"Only {len(groups)} valid frames remain; at least ten are required."
        )
    return ExtendedCurveInput(
        selection=selection,
        start_frame=start,
        end_frame=end,
        reference_frame=reference_id,
        reference_path=reference_path,
        frames=frames,
        groups=groups,
        frame_audit=frame_audit,
        points_path=points_path,
    )


def audit_document(
    data: ExtendedCurveInput, args: argparse.Namespace
) -> dict[str, object]:
    return {
        "dataset": data.selection.get(
            "dataset", "KITTI Odometry Sequence 00"
        ),
        "requested_search_is_not_one_curve": (
            "The configured sequence scan supplies candidate road sections. "
            "Only the selected "
            "ranked section is fitted in this output."
        ),
        "selected_option_rank": int(data.selection.get("selected_rank", 1)),
        "selected_frames": [data.start_frame, data.end_frame],
        "reference_frame": data.reference_frame,
        "valid_frame_count": len(data.frames),
        "selection_json": str(args.selection_json.resolve()),
        "selected_lane_points_json": str(data.points_path.resolve()),
        "poses": str(args.poses.resolve()),
        "calib": str(args.calib.resolve()),
        "coordinate_policy": (
            "unchanged reviewed pipeline: KITTI pose alignment followed by the "
            "existing trajectory Frenet s/d conversion"
        ),
        "accuracy_warning": (
            "Metrics measure held-out consistency with CLRNet-derived points, "
            "not official lane ground-truth accuracy."
        ),
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )

