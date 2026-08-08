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
