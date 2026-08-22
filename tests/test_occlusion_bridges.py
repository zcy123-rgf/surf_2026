from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "bridge_occluded_lane_segments.py"
SPEC = importlib.util.spec_from_file_location("bridge_occluded_lane_segments", SCRIPT)
assert SPEC and SPEC.loader
BRIDGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BRIDGE
SPEC.loader.exec_module(BRIDGE)


def segment(segment_id: int, start: float, end: float, bend: float = 0.0):
    keys = np.linspace(start, end, 30)
    points = np.column_stack([0.02 * keys + bend * keys**2, keys])
    return BRIDGE.Segment(
        side="right",
        segment_id=segment_id,
        keys=keys,
        points=points,
        supports=np.ones(len(keys), dtype=int),
    )


def args() -> argparse.Namespace:
    return argparse.Namespace(
        maximum_order_gap=30.0,
        maximum_endpoint_gap_m=35.0,
        maximum_tangent_angle_deg=45.0,
        maximum_chord_tangent_angle_deg=60.0,
        maximum_bridge_curvature_1pm=0.20,
        minimum_segment_nodes=10,
        tangent_fit_nodes=12,
        bridge_samples=80,
        lane_width_min_m=1.5,
        lane_width_max_m=8.0,
        lane_width_p90_deviation_m=1.5,
    )


def test_supported_gap_creates_g1_hypothesis() -> None:
    first = segment(0, 0.0, 20.0)
    second = segment(1, 30.0, 50.0)
    diagnostic, keys, points = BRIDGE.evaluate_gap(first, second, [], args())
    assert diagnostic["accepted_as_low_confidence_hypothesis"] is True
    assert keys is not None and points is not None
    assert np.allclose(points[0], first.points[-1])
    assert np.allclose(points[-1], second.points[0])
    assert 0.0 < diagnostic["confidence"] < 1.0


def test_long_gap_is_rejected_not_invented() -> None:
    first = segment(0, 0.0, 20.0)
    second = segment(1, 80.0, 100.0)
    diagnostic, keys, points = BRIDGE.evaluate_gap(first, second, [], args())
    assert diagnostic["accepted_as_low_confidence_hypothesis"] is False
    assert "order_gap_exceeds_gate" in diagnostic["rejection_reasons"]
    assert keys is None and points is None


def test_tangent_uses_neighbourhood_not_only_two_points() -> None:
    current = segment(0, 0.0, 20.0)
    tangent = BRIDGE.robust_endpoint_tangent(current, True, 12)
    expected = np.array([0.02, 1.0])
    expected /= np.linalg.norm(expected)
    assert np.allclose(tangent, expected, atol=1e-3)


def test_hermite_bridge_has_endpoint_tangent_direction() -> None:
    p0 = np.array([0.0, 0.0])
    p1 = np.array([4.0, 8.0])
    t0 = np.array([0.0, 1.0])
    t1 = np.array([1.0, 1.0])
    curve = BRIDGE.hermite_bridge(p0, p1, t0, t1, 100)
    assert BRIDGE.vector_angle_deg(curve[1] - curve[0], t0) < 2.0
    assert BRIDGE.vector_angle_deg(curve[-1] - curve[-2], t1) < 2.0
