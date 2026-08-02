from __future__ import annotations

import numpy as np
import pytest

from scripts import fit_first5_two_curves as curve


def synthetic_frames() -> list[dict[str, object]]:
    rng = np.random.default_rng(20260802)
    frames = []
    for frame_id in range(5):
        z = np.linspace(3.0, 45.0, 58)
        common = 0.002 * (z - 22.0) ** 2
        left = np.column_stack(
            [common - 1.8 + rng.normal(0.0, 0.035, len(z)), z]
        )
        right = np.column_stack(
            [common + 1.8 + rng.normal(0.0, 0.035, len(z)), z]
        )
        if frame_id == 2:
            left[25, 0] += 1.0
        frames.append({"frame_id": frame_id, "lanes": [left, right]})
    return frames


def test_two_sides_fit_without_crossing() -> None:
    frames = synthetic_frames()
    fits = {
        side: curve.fit_robust_spline(
            side,
            curve.side_frame_points(frames, side),
            bin_size_m=0.5,
            smoothing_per_point_m2=0.04,
            huber_delta_m=0.20,
            irls_iterations=4,
            curve_samples=400,
        )
        for side in curve.SIDES
    }
    check = curve.no_crossing_check(fits)
    assert check["curves_cross"] is False
    assert check["minimum_right_minus_left_m"] > 3.0
    for side in curve.SIDES:
        metrics = curve.fit_metrics(
            fits[side], curve.side_frame_points(frames, side)
        )
        assert metrics["raw_to_curve_rmse_m"] < 0.12
        assert len(fits[side].curve_points) == 400


def test_nonempty_output_is_rejected(tmp_path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    (output / "old.txt").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(FileExistsError):
        curve.require_empty_output(output)
    assert (output / "old.txt").read_text(encoding="utf-8") == "do not overwrite"
