from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_pose_curvature.py"
SPEC = importlib.util.spec_from_file_location("analyze_pose_curvature", SCRIPT)
assert SPEC and SPEC.loader
CURVATURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CURVATURE)


def synthetic_straight_arc_straight() -> np.ndarray:
    first = np.column_stack([np.zeros(35), np.linspace(0.0, 34.0, 35)])
    radius = 40.0
    angles = np.linspace(0.0, np.pi / 3.0, 50)
    arc = np.column_stack(
        [radius * (1.0 - np.cos(angles)), 34.0 + radius * np.sin(angles)]
    )
    tangent = np.array([np.sin(angles[-1]), np.cos(angles[-1])])
    distances = np.arange(1.0, 36.0)
    last = arc[-1] + distances[:, None] * tangent
    return np.vstack([first, arc[1:], last])


def test_curvature_separates_straight_and_arc() -> None:
    points = synthetic_straight_arc_straight()
    result = CURVATURE.estimate_curvature(points, 0.5, 9.0)
    kappa = np.abs(result["frame_signed_curvature_1pm"])
    assert np.median(kappa[:20]) < 0.003
    assert np.median(kappa[50:75]) > 0.015
    assert np.median(kappa[-20:]) < 0.003


def test_thresholds_are_explicit_and_noise_aware() -> None:
    values = np.array([0.0001, 0.0002, 0.0001, 0.0003, 0.010, 0.020])
    mask = np.array([True, True, True, True, False, False])
    result = CURVATURE.calibrate_thresholds(
        values, mask, 0.0015, 0.0030, 3.0, 6.0
    )
    assert result["baseline_source"] == "registered_straight_seed_ranges"
    assert result["straight_threshold_abs_curvature_1pm"] >= 0.0015
    assert result["curve_threshold_abs_curvature_1pm"] >= 0.0030
    assert (
        result["curve_threshold_abs_curvature_1pm"]
        > result["straight_threshold_abs_curvature_1pm"]
    )


def test_hysteresis_requires_persistent_curve_support() -> None:
    values = np.array(
        [0.0002] * 5 + [0.004] * 2 + [0.0002] * 3 + [0.005] * 5 + [0.0002] * 5
    )
    states = CURVATURE.hysteresis_states(values, 0.0015, 0.0030, 4)
    assert "curve" not in states[5:7]
    assert states[10:15].count("curve") == 5
    assert states[-4:] == ["straight"] * 4


def test_window_layout_covers_registered_range() -> None:
    result = CURVATURE.windows(851, 1005, 15, 10)
    assert result[0] == (851, 865)
    assert result[-1] == (991, 1005)
    assert len(result) == 15
