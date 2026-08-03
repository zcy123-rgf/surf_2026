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
