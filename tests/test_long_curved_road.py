from __future__ import annotations

import numpy as np

from scripts import run_long_curved_road as long_road


def local_curve(side: str, lower: float, upper: float, offset: float) -> long_road.LocalCurve:
    return long_road.LocalCurve(
        method="bspline",
        side=side,
        s_min=lower,
        s_max=upper,
        evaluate_function=lambda values, value=offset: np.zeros_like(values) + value,
        derivative_function=lambda values: np.zeros_like(values),
    )


def window(index: int, lower: float, upper: float, offset: float = 0.0):
    curves = {
        "bspline": {
            "left": local_curve("left", lower, upper, offset),
            "right": local_curve("right", lower, upper, offset + 4.0),
        }
    }
    return {
        "window_index": index,
        "start_frame": index * 10,
        "end_frame": index * 10 + 14,
        "curves": curves,
    }


def test_segment_windows_keeps_continuous_overlaps_together() -> None:
    windows = [window(0, 0.0, 20.0), window(1, 15.0, 35.0, 0.1)]
    chains, audit = long_road.segment_windows(windows, 1.5, 20.0)

    assert len(chains) == 1
    assert len(chains[0]) == 2
    assert all(not row["starts_new_road_segment"] for row in audit)


def test_segment_windows_breaks_at_uncovered_support() -> None:
    windows = [window(0, 0.0, 20.0), window(1, 25.0, 45.0)]
    chains, audit = long_road.segment_windows(windows, 1.5, 20.0)

    assert [len(chain) for chain in chains] == [1, 1]
    assert all(row["starts_new_road_segment"] for row in audit)
    assert any("no_support_overlap" in row["break_reasons"] for row in audit)


def test_segment_windows_breaks_at_large_position_gap() -> None:
    windows = [window(0, 0.0, 20.0), window(1, 15.0, 35.0, 2.0)]
    chains, audit = long_road.segment_windows(windows, 1.5, 20.0)

    assert [len(chain) for chain in chains] == [1, 1]
    assert any("position_gap" in row["break_reasons"] for row in audit)


def test_blended_curve_is_finite_across_shared_support() -> None:
    first = long_road.LocalCurve(
        method="polynomial",
        side="left",
        s_min=0.0,
        s_max=20.0,
        evaluate_function=lambda values: 0.1 * values,
        derivative_function=lambda values: np.zeros_like(values) + 0.1,
    )
    second = long_road.LocalCurve(
        method="polynomial",
        side="left",
        s_min=15.0,
        s_max=35.0,
        evaluate_function=lambda values: 0.1 * values + 0.05,
        derivative_function=lambda values: np.zeros_like(values) + 0.1,
    )
    blend = long_road.BlendedCurve(
        method="polynomial",
        side="left",
        curves=[first, second],
        s_min=0.0,
        s_max=35.0,
    )

    values = blend.evaluate(np.linspace(0.0, 35.0, 200))
    assert np.all(np.isfinite(values))
    assert np.max(np.abs(np.diff(values))) < 0.1
