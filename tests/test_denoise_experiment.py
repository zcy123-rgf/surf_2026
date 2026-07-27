from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_denoise_methods", ROOT / "scripts" / "evaluate_denoise_methods.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def make_ranges(points_per_lane: int) -> list[dict[str, object]]:
    ranges = []
    cursor = 0
    for frame_id in range(5):
        for side in ("left", "right"):
            ranges.append(
                {
                    "frame_id": frame_id,
                    "side": side,
                    "start": cursor,
                    "end": cursor + points_per_lane,
                }
            )
            cursor += points_per_lane
    return ranges


def test_x_of_z_ransac_preserves_vertical_lane_extent() -> None:
    rng = np.random.default_rng(10)
    lanes = []
    for frame_id in range(5):
        z = np.linspace(3.0, 50.0, 50)
        left = np.column_stack([-2.0 + 0.01 * z + rng.normal(0, 0.02, len(z)), z])
        right = np.column_stack([2.0 + 0.01 * z + rng.normal(0, 0.02, len(z)), z])
        lanes.extend([left, right])
    points = np.vstack(lanes)
    mask, _ = MODULE.polynomial_ransac_mask(
        points, make_ranges(50), degree=1, threshold=0.15, iterations=100
    )
    assert np.mean(mask) > 0.98
    assert np.max(points[mask, 1]) == 50.0


def test_temporal_consensus_keeps_unsupported_far_boundary() -> None:
    lanes = []
    ranges = []
    cursor = 0
    for frame_id in range(5):
        for side, x in (("left", -2.0), ("right", 2.0)):
            z_max = 50.0 if frame_id == 0 else 30.0
            z = np.linspace(3.0, z_max, 30)
            lane = np.column_stack([np.full_like(z, x), z])
            lanes.append(lane)
            ranges.append(
                {
                    "frame_id": frame_id,
                    "side": side,
                    "start": cursor,
                    "end": cursor + len(lane),
                }
            )
            cursor += len(lane)
    points = np.vstack(lanes)
    mask, _ = MODULE.temporal_consensus_mask(points, ranges)
    first_lane_far = (points[:30, 1] > 30.0)
    assert np.all(mask[:30][first_lane_far])
