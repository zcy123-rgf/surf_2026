from __future__ import annotations

import numpy as np

from scripts.optimize_temporal_denoise import classifier_metrics
from surf_bev.temporal_denoise import leave_one_out_consensus_mask


def synthetic_five_frames() -> tuple[np.ndarray, list[dict[str, object]]]:
    lanes = []
    ranges = []
    cursor = 0
    for frame_id in range(5):
        for side, x in (("left", -2.0), ("right", 2.0)):
            z = np.linspace(3.0, 50.0, 48)
            lane = np.column_stack(
                [
                    x + 0.01 * z + 0.01 * frame_id,
                    z,
                ]
            )
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
    return np.vstack(lanes), ranges


def test_leave_one_out_rejects_single_frame_lateral_outlier() -> None:
    points, ranges = synthetic_five_frames()
    contaminated = points.copy()
    outlier_index = 20
    contaminated[outlier_index, 0] += 1.2
    keep, _ = leave_one_out_consensus_mask(
        contaminated,
        ranges,
        base_threshold=0.15,
        mad_multiplier=2.5,
        threshold_cap=0.6,
        minimum_other_frames=3,
    )
    assert not keep[outlier_index]
    assert np.mean(keep) > 0.99


def test_classifier_metrics_separates_false_and_true_rejection() -> None:
    points, _ = synthetic_five_frames()
    known = np.zeros(len(points), dtype=bool)
    known[[1, 2, 3]] = True
    keep = np.ones(len(points), dtype=bool)
    keep[[1, 2, 10]] = False
    metrics = classifier_metrics(keep, known, points)
    assert metrics["tp"] == 2
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert np.isclose(metrics["precision"], 2 / 3)
    assert np.isclose(metrics["recall"], 2 / 3)
