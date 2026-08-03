import json

import numpy as np

from scripts.optimize_ransac_parameters import (
    RansacEvaluator,
    config_id,
    phase1_grid,
)
from scripts.optimize_ransac_safety_expansion import prior_trial_count


def synthetic_two_lane_points():
    z = np.linspace(3.0, 45.0, 60)
    left = np.column_stack([-3.5 - 0.03 * z + 0.0002 * z**2, z])
    right = np.column_stack([3.5 - 0.02 * z + 0.0001 * z**2, z])
    points = np.vstack([left, right])
    ranges = [
        {"frame_id": 0, "side": "left", "start": 0, "end": len(left)},
        {
            "frame_id": 0,
            "side": "right",
            "start": len(left),
            "end": len(points),
        },
    ]
    return points, ranges


def test_threshold_profile_increases_with_distance_and_respects_cap():
    points, ranges = synthetic_two_lane_points()
    evaluator = RansacEvaluator(points, ranges, seed=123)
    config = {
        "base_threshold": 0.2,
        "distance_slope": 0.02,
        "threshold_cap": 0.6,
    }
    threshold = evaluator.threshold_vector(np.asarray([3.0, 13.0, 50.0]), config)
    np.testing.assert_allclose(threshold, np.asarray([0.2, 0.4, 0.6]))


def test_same_seed_and_config_are_reproducible():
    points, ranges = synthetic_two_lane_points()
    config = {
        "degree": 2,
        "base_threshold": 0.2,
        "distance_slope": 0.01,
        "threshold_cap": 0.6,
        "iterations": 200,
        "minimum_sample_z_span": 5.0,
        "score_mode": "balanced",
        "local_refinements": 2,
    }
    first, _ = RansacEvaluator(points, ranges, seed=456).apply(config)
    second, _ = RansacEvaluator(points, ranges, seed=456).apply(config)
    np.testing.assert_array_equal(first, second)
    assert np.all(first)


def test_phase1_grid_has_unique_auditable_identifiers():
    identifiers = [config_id(config) for config in phase1_grid()]
    assert len(identifiers) == 126
    assert len(set(identifiers)) == len(identifiers)


def test_safety_expansion_has_no_implicit_historical_result_dependency(tmp_path):
    assert prior_trial_count(None) == 0
    audit = tmp_path / "heldout_audit.json"
    audit.write_text(
        json.dumps({"cumulative_controlled_trials": 1234}),
        encoding="utf-8",
    )
    assert prior_trial_count(audit) == 1234
