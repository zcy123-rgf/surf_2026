import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "fit_polynomial_windows.py"
SPEC = importlib.util.spec_from_file_location("fit_polynomial_windows", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

CURVATURE_PATH = Path(__file__).parents[1] / "scripts" / "analyze_pose_curvature.py"
CURVATURE_SPEC = importlib.util.spec_from_file_location("analyze_pose_curvature", CURVATURE_PATH)
CURVATURE = importlib.util.module_from_spec(CURVATURE_SPEC)
assert CURVATURE_SPEC.loader is not None
sys.modules[CURVATURE_SPEC.name] = CURVATURE
CURVATURE_SPEC.loader.exec_module(CURVATURE)


def test_short_internal_gaps_only_returns_bracketed_short_runs():
    assert MODULE.short_internal_gaps(0, 9, [0, 1, 3, 4, 7, 8, 9], 2) == [2, 5, 6]
    assert MODULE.short_internal_gaps(0, 9, [0, 1, 5, 9], 2) == []


def test_build_windows_adds_terminal_anchor():
    windows = MODULE.build_windows(0, 104, 15, 10)
    assert windows[0].start_frame == 0
    assert windows[0].end_frame == 14
    assert windows[-1].start_frame == 90
    assert windows[-1].end_frame == 104


def test_polynomial_fit_stays_in_metric_xz():
    q = np.linspace(0.0, 1.0, 12)
    aggregate = np.column_stack([2.0 * q + 0.1 * q**2, 5.0 * q + 0.2 * q**2])
    fit = MODULE.fit_parametric_polynomial(
        aggregate,
        np.full(len(aggregate), 3, dtype=np.int64),
        degree=2,
        huber_delta_m=0.2,
        irls_iterations=3,
        curve_samples=40,
    )
    assert fit.model == "parametric_polynomial"
    assert fit.curve_xz.shape == (40, 2)
    assert np.allclose(fit.curve_xz[0], aggregate[0], atol=0.05)
    assert np.allclose(fit.curve_xz[-1], aggregate[-1], atol=0.05)


def test_pose_curvature_is_near_zero_for_a_straight_trajectory():
    poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], 20, axis=0)
    poses[:, 2, 3] = np.arange(20, dtype=np.float64)
    result = CURVATURE.compute_curvature(poses, smoothing_window=5)
    assert float(np.max(result["absolute_curvature_1_per_m"])) < 1e-8


def test_curvature_window_uses_persistence_and_majority():
    values = np.array(
        [0.001, 0.001, 0.006, 0.006, 0.006, 0.0045, 0.006, 0.006, 0.001, 0.001],
        dtype=float,
    )
    classification, fractions = MODULE.classify_curvature_window(
        values,
        straight_max=0.004,
        curve_min=0.005,
        persistence_frames=3,
        majority_fraction=0.60,
    )
    assert classification == "transition"
    assert np.isclose(fractions["curve"], 0.3)
    assert np.isclose(fractions["straight"], 0.0)
