"""Audit CLRNet candidate counts before fitting two lane curves.

This scanner performs live inference but does not invent a missing lane.  It
can preserve the legacy outermost-candidate policy, select the nearest
candidate on each side of the calibrated camera centre, or temporally associate
the ego-adjacent pair using KITTI poses.  ``temporal_joint`` keeps the initial
ego-adjacent identity but subsequently scores every ordered pair jointly, so a
sharp bend is not rejected merely because both visible boundaries fall on the
same side of the image principal point.  ``temporal_independent`` keeps the two
project sides independent after conservative pair initialization, allowing one
observed side to survive a short occlusion without inventing the missing lane.
Temporal track IDs are created by this project; CLRNet itself does not output a
persistent lane identity.
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


def resolve_frame_ids(args: argparse.Namespace) -> list[int]:
    """Resolve either an explicit list or one inclusive frame interval."""

    if args.frame_ids is not None:
        if args.frame_end is not None:
            raise ValueError("--frame-end cannot be combined with --frame-ids.")
        return comma_ints(args.frame_ids)
    if args.frame_start is None or args.frame_end is None:
        raise ValueError("Use --frame-ids or provide both --frame-start and --frame-end.")
    if args.frame_start < 0 or args.frame_end < args.frame_start:
        raise ValueError("Frame range must satisfy 0 <= frame-start <= frame-end.")
    return list(range(args.frame_start, args.frame_end + 1))


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


def select_candidate_pair(
    lanes: list[np.ndarray], mode: str, image_center_x: float
) -> tuple[list[int], list[np.ndarray], str]:
    """Select an ordered left/right pair without synthesizing a missing side."""

    indexed = [
        (index, lane, bottom_x(lane)) for index, lane in enumerate(lanes)
    ]
    if len(indexed) < 2:
        return [], [], "fewer_than_two_candidates"

    ordered = sorted(indexed, key=lambda item: item[2])
    if mode == "outermost":
        chosen = [ordered[0], ordered[-1]]
        return (
            [int(item[0]) for item in chosen],
            [item[1] for item in chosen],
            "selected_outermost",
        )
    if mode != "ego_adjacent":
        raise ValueError(f"Unsupported candidate-selection mode: {mode}")

    left = [item for item in indexed if item[2] < image_center_x]
    right = [item for item in indexed if item[2] > image_center_x]
    if not left or not right:
        return [], [], "missing_candidate_on_one_side_of_camera_center"
    chosen = [max(left, key=lambda item: item[2]), min(right, key=lambda item: item[2])]
    return (
        [int(item[0]) for item in chosen],
        [item[1] for item in chosen],
        "selected_nearest_on_each_side_of_camera_center",
    )


def symmetric_curve_distance_m(first: np.ndarray, second: np.ndarray) -> float:
    """Return a robust, order-independent distance between two metric curves."""

    first = np.asarray(first, dtype=np.float64).reshape(-1, 2)
    second = np.asarray(second, dtype=np.float64).reshape(-1, 2)
    if len(first) < 2 or len(second) < 2:
        return float("inf")
    pairwise = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
    return float(
        0.5
        * (
            np.median(np.min(pairwise, axis=1))
            + np.median(np.min(pairwise, axis=0))
        )
    )


def select_temporal_candidate_pair(
    lanes: list[np.ndarray],
    candidate_ground_lanes: list[np.ndarray],
    image_center_x: float,
    previous_ground_lanes: list[np.ndarray] | None,
    previous_pose: np.ndarray | None,
    current_pose: np.ndarray,
    camera_height: float,
    pitch_deg: float,
    maximum_match_cost_m: float,
) -> tuple[
    list[int],
    list[np.ndarray],
    list[np.ndarray],
    str,
    list[float | None],
]:
    """Associate ego-left/right candidates without treating CLRNet order as ID.

    Candidate indices are only frame-local.  When a previous pair is available,
    its metric points are transformed into the current camera frame with the
    KITTI poses and matched independently on the left and right of the calibrated
    image centre.  A failed distance gate invalidates the frame instead of
    silently switching to a sidewalk or another lane marking.
    """

    if len(lanes) != len(candidate_ground_lanes):
        raise ValueError("Image and metric candidate lists must have equal length.")
    indexed = [
        (index, lane, candidate_ground_lanes[index], bottom_x(lane))
        for index, lane in enumerate(lanes)
        if len(candidate_ground_lanes[index]) >= 2
    ]
    pools = [
        [item for item in indexed if item[3] < image_center_x],
        [item for item in indexed if item[3] > image_center_x],
    ]
    if not pools[0] or not pools[1]:
        return [], [], [], "missing_metric_candidate_on_one_side", [None, None]

    if previous_ground_lanes is None or previous_pose is None:
        chosen = [
            max(pools[0], key=lambda item: item[3]),
            min(pools[1], key=lambda item: item[3]),
        ]
        return (
            [int(item[0]) for item in chosen],
            [item[1] for item in chosen],
            [item[2] for item in chosen],
            "initialized_temporal_tracks_from_ego_adjacent_pair",
            [None, None],
        )

    chosen = []
    costs: list[float | None] = []
    for side_index, pool in enumerate(pools):
        predicted = transform_lane_points_by_pose(
            previous_ground_lanes[side_index],
            previous_pose,
            current_pose,
            camera_height=camera_height,
            pitch_deg=pitch_deg,
        )
        scored = [
            (symmetric_curve_distance_m(predicted, item[2]), item) for item in pool
        ]
        cost, item = min(scored, key=lambda pair: pair[0])
        if not np.isfinite(cost) or cost > maximum_match_cost_m:
            return (
                [],
                [],
                [],
                f"temporal_{('left', 'right')[side_index]}_distance_gate_failed",
                [*(costs), float(cost)],
            )
        chosen.append(item)
        costs.append(float(cost))
    return (
        [int(item[0]) for item in chosen],
        [item[1] for item in chosen],
        [item[2] for item in chosen],
        "matched_project_lane_tracks_with_pose",
        costs,
    )


def select_temporal_joint_candidate_pair(
    lanes: list[np.ndarray],
    candidate_ground_lanes: list[np.ndarray],
    image_center_x: float,
    previous_ground_lanes: list[np.ndarray] | None,
    previous_pose: np.ndarray | None,
    current_pose: np.ndarray,
    camera_height: float,
    pitch_deg: float,
    maximum_match_cost_m: float,
) -> tuple[
    list[int],
    list[np.ndarray],
    list[np.ndarray],
    str,
    list[float | None],
]:
    """Jointly associate two ordered lane tracks across a sharp image-space bend.

    The first valid frame is initialized by the conservative ego-adjacent rule.
    Afterwards, every distinct pair whose bottom-x order is left-to-right is
    tested against the two pose-predicted tracks.  Both individual distances
    must pass the registered metric gate.
    """

    if len(lanes) != len(candidate_ground_lanes):
        raise ValueError("Image and metric candidate lists must have equal length.")
    indexed = [
        (index, lane, candidate_ground_lanes[index], bottom_x(lane))
        for index, lane in enumerate(lanes)
        if len(candidate_ground_lanes[index]) >= 2
    ]
    if len(indexed) < 2:
        return [], [], [], "fewer_than_two_metric_candidates", [None, None]

    if previous_ground_lanes is None or previous_pose is None:
        selected_indices, selected_lanes, reason = select_candidate_pair(
            lanes, mode="ego_adjacent", image_center_x=image_center_x
        )
        if len(selected_indices) != 2:
            return [], [], [], f"temporal_joint_initialization_{reason}", [None, None]
        return (
            selected_indices,
            selected_lanes,
            [candidate_ground_lanes[index] for index in selected_indices],
            "initialized_temporal_joint_tracks_from_ego_adjacent_pair",
            [None, None],
        )

    predicted = [
        transform_lane_points_by_pose(
            previous_ground_lanes[side_index],
            previous_pose,
            current_pose,
            camera_height=camera_height,
            pitch_deg=pitch_deg,
        )
        for side_index in range(2)
    ]
    scored_pairs = []
    for left in indexed:
        for right in indexed:
            if left[0] == right[0] or left[3] >= right[3]:
                continue
            costs = [
                symmetric_curve_distance_m(predicted[0], left[2]),
                symmetric_curve_distance_m(predicted[1], right[2]),
            ]
            scored_pairs.append((max(costs), sum(costs), costs, left, right))
    if not scored_pairs:
        return [], [], [], "no_distinct_left_to_right_candidate_pair", [None, None]

    _, _, costs, left, right = min(scored_pairs, key=lambda item: (item[0], item[1]))
    if any(not np.isfinite(cost) for cost in costs):
        return [], [], [], "temporal_joint_non_finite_distance", [float(cost) for cost in costs]
    if max(costs) > maximum_match_cost_m:
        return (
            [],
            [],
            [],
            "temporal_joint_distance_gate_failed",
            [float(cost) for cost in costs],
        )
    chosen = [left, right]
    return (
        [int(item[0]) for item in chosen],
        [item[1] for item in chosen],
        [item[2] for item in chosen],
        "matched_joint_ordered_project_lane_tracks_with_pose",
        [float(cost) for cost in costs],
    )


def empty_lane_points() -> np.ndarray:
    """Return one shape-stable empty lane for fixed left/right output slots."""

    return np.empty((0, 2), dtype=np.float64)


def expire_independent_tracks(
    previous_ground_lanes: list[np.ndarray | None],
    previous_poses: list[np.ndarray | None],
    previous_frames: list[int | None],
    track_generations: list[int],
    current_frame: int,
    maximum_gap_frames: int,
) -> list[bool]:
    """Expire each project side separately after its registered missing gap."""

    if not all(
        len(values) == 2
        for values in (
            previous_ground_lanes,
            previous_poses,
            previous_frames,
            track_generations,
        )
    ):
        raise ValueError("Independent temporal state must contain left/right slots.")
    resets = [False, False]
    for side in range(2):
        last_frame = previous_frames[side]
        resets[side] = bool(
            last_frame is not None
            and current_frame - last_frame > maximum_gap_frames + 1
        )
        if resets[side]:
            previous_ground_lanes[side] = None
            previous_poses[side] = None
            previous_frames[side] = None
            track_generations[side] += 1
    return resets


def select_temporal_independent_candidates(
    lanes: list[np.ndarray],
    candidate_ground_lanes: list[np.ndarray],
    image_center_x: float,
    previous_ground_lanes: list[np.ndarray | None],
    previous_poses: list[np.ndarray | None],
    current_pose: np.ndarray,
    camera_height: float,
    pitch_deg: float,
    maximum_match_cost_m: float,
) -> tuple[
    list[int | None],
    list[np.ndarray],
    list[np.ndarray],
    list[str],
    list[float | None],
]:
    """Track ego-left and ego-right independently without inventing a lane.

    The first observation still needs a conservative ego-adjacent pair.  Once
    initialized, each active side is pose-predicted and matched separately.
    Candidate assignments remain one-to-one, and a missing side is returned as
    an empty fixed slot so the observed side can still be used downstream.
    """

    if len(lanes) != len(candidate_ground_lanes):
        raise ValueError("Image and metric candidate lists must have equal length.")
    if len(previous_ground_lanes) != 2 or len(previous_poses) != 2:
        raise ValueError("Independent temporal state must contain left/right slots.")

    selected_indices: list[int | None] = [None, None]
    selected_lanes = [empty_lane_points(), empty_lane_points()]
    selected_ground = [empty_lane_points(), empty_lane_points()]
    reasons = ["track_not_initialized", "track_not_initialized"]
    costs: list[float | None] = [None, None]
    indexed = [
        (index, lanes[index], candidate_ground_lanes[index], bottom_x(lanes[index]))
        for index in range(len(lanes))
        if len(candidate_ground_lanes[index]) >= 2
    ]
    active = [
        previous_ground_lanes[side] is not None
        and previous_poses[side] is not None
        for side in range(2)
    ]

    # A lone candidate has no safe left/right identity before either track has
    # been initialized.  Preserve it in the all-candidate fields, but do not
    # assign it to a project side.
    if not any(active):
        pair_indices, pair_lanes, pair_reason = select_candidate_pair(
            lanes, mode="ego_adjacent", image_center_x=image_center_x
        )
        if len(pair_indices) != 2:
            reasons = [
                f"independent_initialization_{pair_reason}",
                f"independent_initialization_{pair_reason}",
            ]
            return selected_indices, selected_lanes, selected_ground, reasons, costs
        for side, (index, lane) in enumerate(zip(pair_indices, pair_lanes)):
            ground = candidate_ground_lanes[index]
            if len(ground) < 2:
                reasons[side] = "independent_initialization_too_few_metric_points"
                continue
            selected_indices[side] = int(index)
            selected_lanes[side] = lane
            selected_ground[side] = ground
            reasons[side] = "initialized_independent_track_from_ego_adjacent_pair"
        return selected_indices, selected_lanes, selected_ground, reasons, costs

    # Build a complete cost table for active tracks.  Keeping the ungated
    # minimum makes a rejected match auditable in the JSON/CSV outputs.
    cost_table: list[dict[int, float]] = [{}, {}]
    for side in range(2):
        if not active[side]:
            continue
        predicted = transform_lane_points_by_pose(
            previous_ground_lanes[side],
            previous_poses[side],
            current_pose,
            camera_height=camera_height,
            pitch_deg=pitch_deg,
        )
        cost_table[side] = {
            int(item[0]): symmetric_curve_distance_m(predicted, item[2])
            for item in indexed
        }

    # Enumerate the two-track assignment explicitly.  This is smaller and more
    # transparent than adding a dependency for a 2xN assignment problem.
    choices: list[tuple[int, float, int | None, int | None]] = []
    left_options: list[int | None] = [None]
    right_options: list[int | None] = [None]
    if active[0]:
        left_options += [
            index
            for index, cost in cost_table[0].items()
            if np.isfinite(cost) and cost <= maximum_match_cost_m
        ]
    if active[1]:
        right_options += [
            index
            for index, cost in cost_table[1].items()
            if np.isfinite(cost) and cost <= maximum_match_cost_m
        ]
    bottom_by_index = {int(item[0]): float(item[3]) for item in indexed}
    for left_index in left_options:
        for right_index in right_options:
            if left_index is not None and left_index == right_index:
                continue
            if (
                left_index is not None
                and right_index is not None
                and bottom_by_index[left_index] >= bottom_by_index[right_index]
            ):
                continue
            matched = int(left_index is not None) + int(right_index is not None)
            total_cost = sum(
                cost_table[side][index]
                for side, index in enumerate((left_index, right_index))
                if index is not None
            )
            choices.append((matched, float(total_cost), left_index, right_index))
    _, _, chosen_left, chosen_right = min(
        choices,
        key=lambda item: (-item[0], item[1]),
    )

    indexed_by_id = {int(item[0]): item for item in indexed}
    for side, index in enumerate((chosen_left, chosen_right)):
        if not active[side]:
            continue
        finite_costs = [
            value for value in cost_table[side].values() if np.isfinite(value)
        ]
        if index is None:
            costs[side] = min(finite_costs) if finite_costs else None
            reasons[side] = "no_candidate_within_independent_temporal_gate"
            continue
        item = indexed_by_id[index]
        selected_indices[side] = int(index)
        selected_lanes[side] = item[1]
        selected_ground[side] = item[2]
        costs[side] = float(cost_table[side][index])
        reasons[side] = "matched_independent_project_lane_track_with_pose"

    # If exactly one side had expired, a complete ego-adjacent pair may safely
    # re-seed the missing side only when the still-active side selected the same
    # member of that conservative pair.  Otherwise keep the missing slot empty.
    if sum(active) == 1:
        active_side = 0 if active[0] else 1
        missing_side = 1 - active_side
        pair_indices, pair_lanes, _ = select_candidate_pair(
            lanes, mode="ego_adjacent", image_center_x=image_center_x
        )
        if (
            len(pair_indices) == 2
            and selected_indices[active_side] == pair_indices[active_side]
            and len(candidate_ground_lanes[pair_indices[missing_side]]) >= 2
        ):
            index = int(pair_indices[missing_side])
            selected_indices[missing_side] = index
            selected_lanes[missing_side] = pair_lanes[missing_side]
            selected_ground[missing_side] = candidate_ground_lanes[index]
            reasons[missing_side] = (
                "reinitialized_missing_track_from_validated_ego_adjacent_pair"
            )

    return selected_indices, selected_lanes, selected_ground, reasons, costs


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


def draw_selected_pair(
    image: np.ndarray,
    lanes: list[np.ndarray],
    image_center_x: float,
) -> np.ndarray:
    """Draw only the selected left/right pair plus the calibrated centre."""

    output = image.copy()
    center = int(round(image_center_x))
    cv2.line(output, (center, 0), (center, image.shape[0] - 1), (0, 255, 255), 1)
    for lane, color in zip(lanes, ((255, 80, 80), (80, 80, 255))):
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
    path_length = float(np.linalg.norm(steps, axis=1).sum())
    displacement = float(np.linalg.norm(positions[-1] - positions[0]))

    # Keep camera-yaw diagnostics, but do not silently equate a brief steering
    # correction with road curvature.  These additional metrics are based on
    # the actual X/Z position trajectory.
    baseline = max(1, min(10, (len(positions) - 1) // 3))
    start_vector = positions[baseline] - positions[0]
    end_vector = positions[-1] - positions[-1 - baseline]
    start_heading = float(np.arctan2(start_vector[0], start_vector[1]))
    end_heading = float(np.arctan2(end_vector[0], end_vector[1]))
    trajectory_turn = float(
        np.arctan2(
            np.sin(end_heading - start_heading),
            np.cos(end_heading - start_heading),
        )
    )

    chord = positions[-1] - positions[0]
    if displacement > 1e-9:
        relative = positions - positions[0]
        chord_deviation = np.abs(
            chord[1] * relative[:, 0] - chord[0] * relative[:, 1]
        ) / displacement
        maximum_chord_deviation = float(chord_deviation.max())
        path_displacement_ratio = path_length / displacement
    else:
        maximum_chord_deviation = 0.0
        path_displacement_ratio = float("inf")
    return {
        "path_length_m": path_length,
        "displacement_m": displacement,
        "net_heading_change_deg": float(np.degrees(yaw[-1] - yaw[0])),
        "total_absolute_heading_change_deg": float(
            np.degrees(np.abs(np.diff(yaw)).sum())
        ),
        "trajectory_net_heading_change_deg": float(np.degrees(trajectory_turn)),
        "maximum_deviation_from_chord_m": maximum_chord_deviation,
        "path_displacement_ratio": path_displacement_ratio,
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
    parser.add_argument(
        "--dataset-name", default="KITTI Odometry Sequence 00"
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--image-pattern", default="{frame_id:06d}.png")
    frame_group = parser.add_mutually_exclusive_group(required=True)
    frame_group.add_argument("--frame-ids")
    frame_group.add_argument("--frame-start", type=int)
    parser.add_argument(
        "--frame-end",
        type=int,
        help="Inclusive end frame; required when --frame-start is used.",
    )
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--segment-size", type=int, default=5)
    parser.add_argument("--minimum-candidates", type=int, default=2)
    parser.add_argument(
        "--candidate-selection-mode",
        choices=(
            "outermost",
            "ego_adjacent",
            "temporal_ego",
            "temporal_joint",
            "temporal_independent",
        ),
        default="outermost",
        help=(
            "outermost preserves the reviewed legacy result; ego_adjacent "
            "selects the bottom-x candidate nearest to each side of P2 cx; "
            "temporal_ego adds pose-based side-split association; temporal_joint "
            "jointly matches an ordered pair after pose prediction; "
            "temporal_independent keeps fixed left/right slots and permits one "
            "observed side without synthesizing the missing lane"
        ),
    )
    parser.add_argument("--temporal-maximum-match-cost-m", type=float, default=1.50)
    parser.add_argument("--temporal-maximum-gap-frames", type=int, default=3)
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
    parser.add_argument(
        "--skip-recommendation",
        action="store_true",
        help=(
            "Skip the scanner's single longest-window recommendation. Use this "
            "for large range scans followed by select_hierarchy_150_frames.py."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame_ids = resolve_frame_ids(args)
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
    if (
        args.temporal_maximum_match_cost_m <= 0
        or args.temporal_maximum_gap_frames < 0
    ):
        raise ValueError("Temporal match cost must be positive and gap non-negative.")
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
    image_center_x = float(calibration["K"][0, 2])
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
    (args.output_dir / "selected_pairs").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    ground_lanes: dict[int, list[np.ndarray]] = {}
    point_records: list[dict[str, object]] = []
    previous_temporal_ground: list[np.ndarray] | None = None
    previous_temporal_pose: np.ndarray | None = None
    previous_temporal_frame: int | None = None
    temporal_track_generation = 0
    previous_independent_ground: list[np.ndarray | None] = [None, None]
    previous_independent_pose: list[np.ndarray | None] = [None, None]
    previous_independent_frame: list[int | None] = [None, None]
    independent_track_generation = [0, 0]
    for frame_id, image_path in zip(frame_ids, image_paths):
        image = imread(image_path)
        lanes = detector.detect(image)["lanes"]
        candidate_ground_lanes = [
            image_to_ground_ipm(
                lane,
                calibration["K"],
                camera_height=args.camera_height,
                pitch_deg=args.pitch_deg,
                z_range=local_z_range,
            )
            for lane in lanes
        ]
        temporal_match_costs: list[float | None] = [None, None]
        side_selection_reasons = ["not_applicable", "not_applicable"]
        independent_gap_resets = [False, False]
        if args.candidate_selection_mode == "temporal_independent":
            independent_gap_resets = expire_independent_tracks(
                previous_independent_ground,
                previous_independent_pose,
                previous_independent_frame,
                independent_track_generation,
                current_frame=frame_id,
                maximum_gap_frames=args.temporal_maximum_gap_frames,
            )
        temporal_gap_reset = bool(
            previous_temporal_frame is not None
            and frame_id - previous_temporal_frame
            > args.temporal_maximum_gap_frames + 1
        )
        if temporal_gap_reset:
            previous_temporal_ground = None
            previous_temporal_pose = None
            previous_temporal_frame = None
            temporal_track_generation += 1
        if args.candidate_selection_mode == "temporal_independent":
            (
                selected_indices,
                selected_lanes,
                selected_ground_lanes,
                side_selection_reasons,
                temporal_match_costs,
            ) = select_temporal_independent_candidates(
                lanes,
                candidate_ground_lanes,
                image_center_x=image_center_x,
                previous_ground_lanes=previous_independent_ground,
                previous_poses=previous_independent_pose,
                current_pose=poses_image[frame_id],
                camera_height=args.camera_height,
                pitch_deg=args.pitch_deg,
                maximum_match_cost_m=args.temporal_maximum_match_cost_m,
            )
            pair_reason = ";".join(side_selection_reasons)
        elif len(lanes) >= args.minimum_candidates:
            if args.candidate_selection_mode in ("temporal_ego", "temporal_joint"):
                selector = (
                    select_temporal_candidate_pair
                    if args.candidate_selection_mode == "temporal_ego"
                    else select_temporal_joint_candidate_pair
                )
                (
                    selected_indices,
                    selected_lanes,
                    selected_ground_lanes,
                    pair_reason,
                    temporal_match_costs,
                ) = selector(
                    lanes,
                    candidate_ground_lanes,
                    image_center_x=image_center_x,
                    previous_ground_lanes=previous_temporal_ground,
                    previous_pose=previous_temporal_pose,
                    current_pose=poses_image[frame_id],
                    camera_height=args.camera_height,
                    pitch_deg=args.pitch_deg,
                    maximum_match_cost_m=args.temporal_maximum_match_cost_m,
                )
            else:
                selected_indices, selected_lanes, pair_reason = select_candidate_pair(
                    lanes,
                    mode=args.candidate_selection_mode,
                    image_center_x=image_center_x,
                )
                selected_ground_lanes = [
                    candidate_ground_lanes[index] for index in selected_indices
                ]
            side_selection_reasons = [pair_reason, pair_reason]
        else:
            selected_indices, selected_lanes, selected_ground_lanes, pair_reason = (
                [],
                [],
                [],
                "below_minimum_candidate_count",
            )
            side_selection_reasons = [pair_reason, pair_reason]
        selected_observed_mask = [len(lane) >= 2 for lane in selected_lanes]
        selected_bottom_x: list[float | None] = [
            bottom_x(lane) if observed else None
            for lane, observed in zip(selected_lanes, selected_observed_mask)
        ]
        projected_counts = [len(lane) for lane in selected_ground_lanes]
        side_metric_gates = [False, False]
        if len(projected_counts) == 2:
            side_metric_gates = [
                count >= args.minimum_bev_points_per_side
                for count in projected_counts
            ]
        metric_gate = len(projected_counts) == 2 and all(side_metric_gates)
        independent_metric_gate = any(side_metric_gates)
        ground_lanes[frame_id] = selected_ground_lanes
        if args.candidate_selection_mode in ("temporal_ego", "temporal_joint") and metric_gate:
            previous_temporal_ground = selected_ground_lanes
            previous_temporal_pose = poses_image[frame_id]
            previous_temporal_frame = frame_id
        if args.candidate_selection_mode == "temporal_independent":
            for side in range(2):
                if side_metric_gates[side]:
                    previous_independent_ground[side] = selected_ground_lanes[side]
                    previous_independent_pose[side] = poses_image[frame_id]
                    previous_independent_frame[side] = frame_id
            track_ids: list[str | None] = [
                (
                    f"ego_{('left', 'right')[side]}_"
                    f"{independent_track_generation[side]:03d}"
                )
                if side_metric_gates[side]
                else None
                for side in range(2)
            ]
            active_track_ids: list[str | None] = [
                (
                    f"ego_{('left', 'right')[side]}_"
                    f"{independent_track_generation[side]:03d}"
                )
                if previous_independent_ground[side] is not None
                else None
                for side in range(2)
            ]
        else:
            track_ids = (
                [
                    f"ego_left_{temporal_track_generation:03d}",
                    f"ego_right_{temporal_track_generation:03d}",
                ]
                if args.candidate_selection_mode
                in ("temporal_ego", "temporal_joint")
                and metric_gate
                else []
            )
            active_track_ids = list(track_ids)
        point_records.append(
            {
                "frame_id": frame_id,
                "candidate_count": len(lanes),
                "candidate_selection_mode": args.candidate_selection_mode,
                "image_center_x_px": image_center_x,
                "pair_selection_reason": pair_reason,
                "side_selection_reasons": side_selection_reasons,
                "selected_candidate_indices": selected_indices,
                "selected_candidate_bottom_x_px": selected_bottom_x,
                "project_track_ids": track_ids,
                "active_project_track_ids": active_track_ids,
                "temporal_match_cost_m": temporal_match_costs,
                "temporal_gap_reset": temporal_gap_reset,
                "temporal_side_gap_reset": independent_gap_resets,
                "all_candidate_lanes_image_xy_px": [
                    lane.tolist() for lane in lanes
                ],
                "all_candidate_lanes_local_ground_xz_m": [
                    lane.tolist() for lane in candidate_ground_lanes
                ],
                "selected_lanes_image_xy_px": [lane.tolist() for lane in selected_lanes],
                "selected_lanes_local_ground_xz_m": [
                    lane.tolist() for lane in ground_lanes[frame_id]
                ],
                "observed_side_mask": selected_observed_mask,
                "side_metric_gate": side_metric_gates,
                "eligible_for_independent_curve_fit": independent_metric_gate,
                "eligible_for_two_curve_fit": metric_gate,
            }
        )
        shutil.copy2(
            image_path,
            args.output_dir / "original_frames" / f"frame_{frame_id:06d}.png",
        )
        imwrite(
            args.output_dir / "all_candidates" / f"frame_{frame_id:06d}.png",
            draw_candidates(image, lanes),
        )
        imwrite(
            args.output_dir / "selected_pairs" / f"frame_{frame_id:06d}.png",
            draw_selected_pair(image, selected_lanes, image_center_x),
        )
        rows.append(
            {
                "frame_id": frame_id,
                "candidate_count": len(lanes),
                "candidate_point_counts": ";".join(str(len(lane)) for lane in lanes),
                "candidate_bottom_x_px": ";".join(
                    f"{bottom_x(lane):.6f}" for lane in lanes
                ),
                "candidate_selection_mode": args.candidate_selection_mode,
                "image_center_x_px": image_center_x,
                "pair_selection_reason": pair_reason,
                "side_selection_reasons": ";".join(side_selection_reasons),
                "selected_candidate_indices": ";".join(
                    "" if value is None else str(value)
                    for value in selected_indices
                ),
                "selected_candidate_bottom_x_px": ";".join(
                    "" if value is None else f"{value:.6f}"
                    for value in selected_bottom_x
                ),
                "project_track_ids": ";".join(
                    "" if value is None else value for value in track_ids
                ),
                "active_project_track_ids": ";".join(
                    "" if value is None else value for value in active_track_ids
                ),
                "temporal_match_cost_m": ";".join(
                    "" if value is None else f"{value:.6f}"
                    for value in temporal_match_costs
                ),
                "temporal_gap_reset": temporal_gap_reset,
                "temporal_side_gap_reset": ";".join(
                    str(value) for value in independent_gap_resets
                ),
                "left_observed": (
                    selected_observed_mask[0]
                    if len(selected_observed_mask) == 2
                    else False
                ),
                "right_observed": (
                    selected_observed_mask[1]
                    if len(selected_observed_mask) == 2
                    else False
                ),
                "pair_selection_gate": (
                    len(selected_observed_mask) == 2
                    and all(selected_observed_mask)
                ),
                "selected_projected_point_counts": ";".join(
                    str(count) for count in projected_counts
                ),
                "left_projected_point_count": (
                    projected_counts[0] if len(projected_counts) == 2 else 0
                ),
                "right_projected_point_count": (
                    projected_counts[1] if len(projected_counts) == 2 else 0
                ),
                "candidate_count_gate": len(lanes) >= args.minimum_candidates,
                "metric_bev_point_gate": metric_gate,
                "eligible_for_independent_curve_fit": independent_metric_gate,
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

    if args.skip_recommendation:
        recommendation = {
            "status": "not_requested",
            "reason": (
                "Large-range scan requested. Candidate windows are ranked by "
                "select_hierarchy_150_frames.py instead."
            ),
        }
    else:
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
        "dataset": args.dataset_name,
        "requested_frame_ids": frame_ids,
        "frame_count": len(frame_ids),
        "candidate_selection_mode": args.candidate_selection_mode,
        "calibrated_image_center_x_px": image_center_x,
        "frames_with_at_least_two_candidates": sum(
            int(row["candidate_count"]) >= 2 for row in rows
        ),
        "frames_with_selected_pair": sum(
            bool(row["pair_selection_gate"]) for row in rows
        ),
        "frames_passing_metric_bev_gate": sum(
            bool(row["metric_bev_point_gate"]) for row in rows
        ),
        "frames_with_at_least_one_independent_side": sum(
            bool(row["eligible_for_independent_curve_fit"]) for row in rows
        ),
        "frames_with_left_observation": sum(
            bool(row["left_observed"]) for row in rows
        ),
        "frames_with_right_observation": sum(
            bool(row["right_observed"]) for row in rows
        ),
        "counts": rows,
        "recommendation": recommendation,
        "warnings": [
            "Candidate count does not prove left/right lane-boundary identity.",
            "ego_adjacent is a geometric pairing rule, not a semantic proof that a candidate is painted lane marking.",
            "Temporal track IDs are created by this project and are not CLRNet or KITTI map lane IDs.",
            "temporal_joint relaxes only the per-frame principal-point split; it still requires two distinct ordered candidates and a metric pose gate.",
            "temporal_independent stores a missing side as an empty slot and never treats a track prediction as an observed lane.",
            "Pose continuity cannot by itself distinguish a persistent sidewalk edge from a persistent lane marking.",
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
    point_document = {
        "status": "complete",
        "dataset": args.dataset_name,
        "coordinate_system": (
            "per-frame image-camera ground X/Z in metres; X right, Z forward"
        ),
        "camera_height_m": args.camera_height,
        "pitch_deg": args.pitch_deg,
        "local_z_range_m": list(local_z_range),
        "candidate_selection_mode": args.candidate_selection_mode,
        "calibrated_image_center_x_px": image_center_x,
        "selection": (
            "outermost candidates ordered by bottom image x"
            if args.candidate_selection_mode == "outermost"
            else (
                "nearest bottom-x candidate on each side of calibrated P2 cx"
                if args.candidate_selection_mode == "ego_adjacent"
                else (
                    "pose-associated ego-left/ego-right project tracks with image-centre side pools"
                    if args.candidate_selection_mode == "temporal_ego"
                    else (
                        "pose-associated jointly matched ordered ego-left/ego-right project tracks"
                        if args.candidate_selection_mode == "temporal_joint"
                        else "independently pose-associated ego-left/ego-right project tracks with fixed empty slots for missing observations"
                    )
                )
            )
        ),
        "all_candidate_definition": (
            "all post-confidence-threshold and post-NMS CLRNet candidates returned by the detector adapter"
        ),
        "frames": point_records,
        "warning": (
            "Geometrically or temporally selected candidates are not guaranteed to be painted lane markings."
        ),
    }
    (args.output_dir / "selected_lane_points.json").write_text(
        json.dumps(point_document, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(recommendation, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
