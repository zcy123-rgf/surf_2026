from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from scripts.optimize_ransac_parameters import RansacEvaluator
from surf_bev.optimized_ransac import SideAwarePolynomialRansac, config_id


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "selected_reference", ROOT / "scripts" / "run_selected_ransac_reference.py"
)
REFERENCE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(REFERENCE)


def synthetic_points() -> tuple[np.ndarray, list[dict[str, object]]]:
    z = np.linspace(3.0, 50.0, 80)
    left = np.column_stack([-3.2 + 0.02 * z + 0.0005 * z**2, z])
    right = np.column_stack([3.2 + 0.015 * z - 0.0004 * z**2, z])
    left[[9, 61], 0] += np.asarray([2.5, -3.0])
    right[[13, 70], 0] += np.asarray([-2.0, 3.2])
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


def test_fixed_configuration_is_the_passed_safety_expansion() -> None:
    assert config_id(REFERENCE.SELECTED_CONFIG) == (
        "d3_b0.350_s0.015_c0.80_i1000_z10.0_qbalanced_r1"
    )


def test_extracted_runtime_matches_audited_evaluator() -> None:
    points, ranges = synthetic_points()
    seed = 20261716
    expected_mask, expected_details = RansacEvaluator(
        points, ranges, seed
    ).apply(REFERENCE.SELECTED_CONFIG)
    actual_mask, actual_details = SideAwarePolynomialRansac(
        points, ranges, seed
    ).apply(REFERENCE.SELECTED_CONFIG)
    np.testing.assert_array_equal(actual_mask, expected_mask)
    assert actual_details["config_id"] == expected_details["config_id"]
    assert actual_details["left"]["kept"] == expected_details["left"]["kept"]
    assert actual_details["right"]["kept"] == expected_details["right"]["kept"]


def test_release_entrypoints_do_not_default_to_sparse_hierarchy() -> None:
    ransac = (
        ROOT / "workstation_release" / "01_ransac_reference" / "run_windows.ps1"
    ).read_text(encoding="utf-8-sig")
    curves = (
        ROOT / "workstation_release" / "02_curve_models" / "run_windows.ps1"
    ).read_text(encoding="utf-8-sig")
    assert "run_selected_ransac_reference.py" in ransac
    assert "compare_polynomial_bspline.py" in curves
    assert "run_weekly_lane_hierarchy.py" not in ransac + curves
    assert "run_meeting_four_questions_windows.ps1" not in ransac + curves
