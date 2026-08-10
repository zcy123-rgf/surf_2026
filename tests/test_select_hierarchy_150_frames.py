import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "select_hierarchy_150_frames",
    ROOT / "scripts" / "select_hierarchy_150_frames.py",
)
SELECTOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SELECTOR)


def test_candidate_spans_score_ten_fixed_fifteen_frame_blocks(tmp_path):
    rows = [
        {
            "frame_id": frame_id,
            "candidate_count": 2,
            "eligible_for_two_curve_fit": frame_id % 15 < 10,
        }
        for frame_id in range(170)
    ]
    poses = np.zeros((170, 3, 4), dtype=np.float64)
    poses[:, 0, 0] = 1.0
    poses[:, 1, 1] = 1.0
    poses[:, 2, 2] = 1.0
    poses[:, 2, 3] = np.arange(170, dtype=np.float64)
    result = SELECTOR.candidate_spans(
        rows,
        poses,
        block_size=15,
        block_count=10,
        block_stride=15,
        minimum_valid=5,
        scan_json=tmp_path / "scan.json",
    )
    assert len(result) == 21
    first = result[0]
    assert first["start_frame"] == 0
    assert first["end_frame"] == 149
    assert first["block_valid_counts"] == [10] * 10
    assert first["minimum_valid_frames_in_a_block"] == 10
    assert first["all_blocks_meet_minimum"] is True


def test_overlapping_ten_by_fifteen_windows_cover_105_unique_frames(tmp_path):
    rows = [
        {
            "frame_id": frame_id,
            "candidate_count": 2,
            "eligible_for_two_curve_fit": True,
        }
        for frame_id in range(170)
    ]
    poses = np.zeros((170, 3, 4), dtype=np.float64)
    poses[:, 0, 0] = 1.0
    poses[:, 1, 1] = 1.0
    poses[:, 2, 2] = 1.0
    result = SELECTOR.candidate_spans(
        rows,
        poses,
        block_size=15,
        block_count=10,
        block_stride=10,
        minimum_valid=5,
        scan_json=tmp_path / "scan.json",
    )
    assert len(result) == 66
    assert result[0]["start_frame"] == 0
    assert result[0]["end_frame"] == 104
    assert result[0]["block_valid_counts"] == [15] * 10


def test_candidate_counts_sustained_position_trajectory_turns(tmp_path):
    count = 170
    rows = [
        {
            "frame_id": frame_id,
            "candidate_count": 2,
            "eligible_for_two_curve_fit": True,
        }
        for frame_id in range(count)
    ]
    poses = np.zeros((count, 3, 4), dtype=np.float64)
    poses[:, 0, 0] = 1.0
    poses[:, 1, 1] = 1.0
    poses[:, 2, 2] = 1.0
    angle = np.linspace(0.0, np.pi / 2.0, count)
    poses[:, 0, 3] = 30.0 * np.sin(angle)
    poses[:, 2, 3] = 30.0 * (1.0 - np.cos(angle))
    result = SELECTOR.candidate_spans(
        rows,
        poses,
        block_size=15,
        block_count=10,
        block_stride=10,
        minimum_valid=10,
        scan_json=tmp_path / "scan.json",
        minimum_trajectory_turn_deg_per_block=1.0,
    )
    assert result[0]["sustained_turn_block_count"] >= 8
    assert result[0]["maximum_deviation_from_chord_m"] > 1.0
    assert result[0]["path_displacement_ratio"] > 1.0


def test_straight_curve_straight_profile_is_detected(tmp_path):
    count = 105
    rows = [
        {
            "frame_id": frame_id,
            "candidate_count": 2,
            "eligible_for_two_curve_fit": True,
        }
        for frame_id in range(count)
    ]
    headings = np.zeros(count, dtype=np.float64)
    headings[25:76] = np.linspace(0.0, np.deg2rad(45.0), 51)
    headings[76:] = headings[75]
    positions = np.zeros((count, 2), dtype=np.float64)
    for index in range(1, count):
        positions[index] = positions[index - 1] + np.asarray(
            [np.sin(headings[index]), np.cos(headings[index])]
        )
    poses = np.zeros((count, 3, 4), dtype=np.float64)
    poses[:, 0, 0] = 1.0
    poses[:, 1, 1] = 1.0
    poses[:, 2, 2] = 1.0
    poses[:, 0, 3] = positions[:, 0]
    poses[:, 2, 3] = positions[:, 1]
    result = SELECTOR.candidate_spans(
        rows,
        poses,
        block_size=15,
        block_count=10,
        block_stride=10,
        minimum_valid=10,
        scan_json=tmp_path / "scan.json",
        maximum_straight_turn_deg_per_block=1.5,
        minimum_curve_turn_deg_per_block=3.0,
        minimum_middle_curve_blocks=2,
    )
    assert len(result) == 1
    profile = result[0]
    assert profile["straight_curve_straight_gate"] is True
    assert profile["entry_mean_trajectory_turn_deg"] <= 1.5
    assert profile["exit_mean_trajectory_turn_deg"] <= 1.5
    assert profile["middle_curve_block_count"] >= 2
