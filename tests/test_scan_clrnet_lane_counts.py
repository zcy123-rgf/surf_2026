import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "scan_clrnet_lane_counts", ROOT / "scripts" / "scan_clrnet_lane_counts.py"
)
SCANNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SCANNER)


def make_poses(count: int, curved_from: int = 0) -> np.ndarray:
    poses = np.zeros((count, 3, 4), dtype=np.float64)
    for index in range(count):
        yaw = 0.0 if index < curved_from else 0.02 * (index - curved_from)
        rotation = np.asarray(
            [
                [np.cos(yaw), 0.0, np.sin(yaw)],
                [0.0, 1.0, 0.0],
                [-np.sin(yaw), 0.0, np.cos(yaw)],
            ]
        )
        poses[index, :, :3] = rotation
        poses[index, 0, 3] = index * 0.1
        poses[index, 2, 3] = index * 0.8
    return poses


def rows(counts: list[int]) -> list[dict[str, object]]:
    return [
        {"frame_id": frame_id, "candidate_count": count}
        for frame_id, count in enumerate(counts)
    ]


def lane_at_bottom_x(x: float) -> np.ndarray:
    return np.asarray([[x, 100.0], [x, 200.0], [x, 300.0]])


def test_ego_adjacent_selects_nearest_candidate_on_each_side_of_center():
    lanes = [lane_at_bottom_x(100.0), lane_at_bottom_x(500.0), lane_at_bottom_x(900.0)]
    indices, selected, reason = SCANNER.select_candidate_pair(
        lanes, mode="ego_adjacent", image_center_x=620.0
    )
    assert indices == [1, 2]
    assert [SCANNER.bottom_x(lane) for lane in selected] == [500.0, 900.0]
    assert reason == "selected_nearest_on_each_side_of_camera_center"


def test_outermost_mode_preserves_legacy_pairing():
    lanes = [lane_at_bottom_x(100.0), lane_at_bottom_x(500.0), lane_at_bottom_x(900.0)]
    indices, selected, reason = SCANNER.select_candidate_pair(
        lanes, mode="outermost", image_center_x=620.0
    )
    assert indices == [0, 2]
    assert [SCANNER.bottom_x(lane) for lane in selected] == [100.0, 900.0]
    assert reason == "selected_outermost"


def test_ego_adjacent_rejects_frame_without_both_sides():
    lanes = [lane_at_bottom_x(100.0), lane_at_bottom_x(500.0)]
    indices, selected, reason = SCANNER.select_candidate_pair(
        lanes, mode="ego_adjacent", image_center_x=620.0
    )
    assert indices == []
    assert selected == []
    assert reason == "missing_candidate_on_one_side_of_camera_center"


def test_temporal_pair_uses_pose_aligned_curve_distance_not_frame_local_index():
    lanes = [
        lane_at_bottom_x(100.0),
        lane_at_bottom_x(500.0),
        lane_at_bottom_x(740.0),
        lane_at_bottom_x(1100.0),
    ]
    z = np.linspace(3.0, 30.0, 20)
    candidates_ground = [
        np.column_stack([np.full_like(z, -6.0), z]),
        np.column_stack([np.full_like(z, -3.1), z]),
        np.column_stack([np.full_like(z, 3.1), z]),
        np.column_stack([np.full_like(z, 6.0), z]),
    ]
    previous = [
        np.column_stack([np.full_like(z, -3.0), z]),
        np.column_stack([np.full_like(z, 3.0), z]),
    ]
    pose = np.eye(4, dtype=np.float64)
    indices, _, selected_ground, reason, costs = (
        SCANNER.select_temporal_candidate_pair(
            lanes,
            candidates_ground,
            image_center_x=620.0,
            previous_ground_lanes=previous,
            previous_pose=pose,
            current_pose=pose,
            camera_height=1.65,
            pitch_deg=0.0,
            maximum_match_cost_m=1.0,
        )
    )
    assert indices == [1, 2]
    assert reason == "matched_project_lane_tracks_with_pose"
    assert len(selected_ground) == 2
    assert max(costs) < 0.11


def test_temporal_pair_invalidates_frame_when_track_gate_fails():
    lanes = [lane_at_bottom_x(100.0), lane_at_bottom_x(900.0)]
    z = np.linspace(3.0, 30.0, 20)
    candidates_ground = [
        np.column_stack([np.full_like(z, -8.0), z]),
        np.column_stack([np.full_like(z, 8.0), z]),
    ]
    previous = [
        np.column_stack([np.full_like(z, -3.0), z]),
        np.column_stack([np.full_like(z, 3.0), z]),
    ]
    pose = np.eye(4, dtype=np.float64)
    indices, selected, _, reason, _ = SCANNER.select_temporal_candidate_pair(
        lanes,
        candidates_ground,
        image_center_x=620.0,
        previous_ground_lanes=previous,
        previous_pose=pose,
        current_pose=pose,
        camera_height=1.65,
        pitch_deg=0.0,
        maximum_match_cost_m=1.0,
    )
    assert indices == []
    assert selected == []
    assert "distance_gate_failed" in reason


def test_temporal_joint_recovers_ordered_pair_when_both_candidates_are_one_side():
    lanes = [
        lane_at_bottom_x(700.0),
        lane_at_bottom_x(820.0),
        lane_at_bottom_x(1050.0),
    ]
    z = np.linspace(3.0, 30.0, 20)
    candidates_ground = [
        np.column_stack([np.full_like(z, -3.1), z]),
        np.column_stack([np.full_like(z, 3.1), z]),
        np.column_stack([np.full_like(z, 7.0), z]),
    ]
    previous = [
        np.column_stack([np.full_like(z, -3.0), z]),
        np.column_stack([np.full_like(z, 3.0), z]),
    ]
    pose = np.eye(4, dtype=np.float64)

    old_indices, *_ = SCANNER.select_temporal_candidate_pair(
        lanes,
        candidates_ground,
        image_center_x=620.0,
        previous_ground_lanes=previous,
        previous_pose=pose,
        current_pose=pose,
        camera_height=1.65,
        pitch_deg=0.0,
        maximum_match_cost_m=1.0,
    )
    indices, _, selected_ground, reason, costs = (
        SCANNER.select_temporal_joint_candidate_pair(
            lanes,
            candidates_ground,
            image_center_x=620.0,
            previous_ground_lanes=previous,
            previous_pose=pose,
            current_pose=pose,
            camera_height=1.65,
            pitch_deg=0.0,
            maximum_match_cost_m=1.0,
        )
    )

    assert old_indices == []
    assert indices == [0, 1]
    assert len(set(indices)) == 2
    assert len(selected_ground) == 2
    assert reason == "matched_joint_ordered_project_lane_tracks_with_pose"
    assert max(costs) < 0.11


def test_temporal_joint_keeps_distance_gate_instead_of_forcing_a_pair():
    lanes = [lane_at_bottom_x(700.0), lane_at_bottom_x(820.0)]
    z = np.linspace(3.0, 30.0, 20)
    candidates_ground = [
        np.column_stack([np.full_like(z, -8.0), z]),
        np.column_stack([np.full_like(z, 8.0), z]),
    ]
    previous = [
        np.column_stack([np.full_like(z, -3.0), z]),
        np.column_stack([np.full_like(z, 3.0), z]),
    ]
    pose = np.eye(4, dtype=np.float64)
    indices, selected, _, reason, costs = (
        SCANNER.select_temporal_joint_candidate_pair(
            lanes,
            candidates_ground,
            image_center_x=620.0,
            previous_ground_lanes=previous,
            previous_pose=pose,
            current_pose=pose,
            camera_height=1.65,
            pitch_deg=0.0,
            maximum_match_cost_m=1.0,
        )
    )
    assert indices == []
    assert selected == []
    assert reason == "temporal_joint_distance_gate_failed"
    assert max(costs) > 1.0


def test_eligible_runs_split_on_single_candidate_frame():
    result = SCANNER.eligible_runs(rows([2, 3, 1, 2, 2, 1, 4]))
    assert result == [[0, 1], [3, 4], [6]]


def test_no_five_frame_run_does_not_synthesize_missing_lane():
    result = SCANNER.select_window(
        rows([2, 2, 1, 2, 2, 2, 2]), make_poses(7), segment_size=5
    )
    assert result["status"] == "no_valid_run"
    assert "No missing lane was synthesized" in result["reason"]


def test_metric_bev_gate_splits_run_even_when_candidate_count_is_two():
    input_rows = rows([2] * 10)
    input_rows[4]["eligible_for_two_curve_fit"] = False
    result = SCANNER.select_window(input_rows, make_poses(10), segment_size=5)
    assert result["status"] == "selected"
    assert result["frame_ids"] == [5, 6, 7, 8, 9]


def test_exact_aligned_gate_rejects_points_outside_fusion_range():
    input_rows = rows([2] * 10)
    for row in input_rows:
        row["eligible_for_two_curve_fit"] = True
    poses = []
    ground_lanes = {}
    for frame_id in range(10):
        pose = np.eye(4, dtype=np.float64)
        pose[2, 3] = frame_id * 0.1
        poses.append(pose)
        x_left = 25.0 if frame_id == 4 else -3.0
        ground_lanes[frame_id] = [
            np.column_stack([np.full(4, x_left), np.arange(5.0, 9.0)]),
            np.column_stack([np.full(4, 3.0), np.arange(5.0, 9.0)]),
        ]
    result = SCANNER.select_window_with_aligned_points(
        input_rows,
        poses,
        ground_lanes,
        segment_size=5,
        minimum_candidates=2,
        minimum_points_per_side=4,
        fusion_x_range=(-20.0, 20.0),
        fusion_z_range=(-20.0, 50.0),
        camera_height=1.65,
        pitch_deg=0.0,
    )
    assert result["status"] == "selected"
    assert result["frame_ids"] == [5, 6, 7, 8, 9]


def test_selects_longest_run_and_trims_to_segment_multiple():
    result = SCANNER.select_window(
        rows([2] * 8 + [1] + [2] * 6), make_poses(15, curved_from=9), segment_size=5
    )
    assert result["status"] == "selected"
    assert result["selected_length"] == 5
    # Both eligible runs permit five frames; the later, curved run wins the
    # documented pose-heading-change tie break.
    assert result["frame_ids"] == [9, 10, 11, 12, 13]


def test_longer_valid_window_has_priority_over_curvature():
    result = SCANNER.select_window(
        rows([2] * 10 + [1] + [2] * 5), make_poses(16, curved_from=11), segment_size=5
    )
    assert result["status"] == "selected"
    assert result["frame_ids"] == list(range(10))


def test_position_trajectory_metric_rejects_yaw_only_false_curve():
    poses = make_poses(20)
    poses[:, 0, 3] = 0.0
    poses[:, 2, 3] = np.arange(20, dtype=np.float64)
    stats = SCANNER.pose_window_stats(poses, list(range(20)))
    assert stats["net_heading_change_deg"] > 20.0
    assert abs(stats["trajectory_net_heading_change_deg"]) < 1e-9
    assert stats["maximum_deviation_from_chord_m"] < 1e-9
    assert abs(stats["path_displacement_ratio"] - 1.0) < 1e-9
