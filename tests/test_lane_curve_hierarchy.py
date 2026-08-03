from __future__ import annotations

from argparse import Namespace

import numpy as np

from scripts import analyze_lane_curve_hierarchy as hierarchy


def frames20() -> list[dict[str, object]]:
    rng = np.random.default_rng(20260803)
    frames = []
    for frame_id in range(20):
        z = np.linspace(-10.0, 48.0, 64)
        center = 0.0015 * (z - 18.0) ** 2
        left = np.column_stack(
            [center - 3.5 + rng.normal(0.0, 0.025, len(z)), z]
        )
        right = np.column_stack(
            [center + 3.5 + rng.normal(0.0, 0.025, len(z)), z]
        )
        frames.append({"frame_id": frame_id, "lanes": [left, right]})
    return frames


def arguments() -> Namespace:
    return Namespace(
        segment_size=5,
        feature_points_per_segment=8,
        bin_size_m=0.5,
        huber_delta_m=0.2,
        irls_iterations=4,
        curve_samples=300,
    )


def test_segment_feature_refusion_is_compact_and_close() -> None:
    frames = frames20()
    direct = hierarchy.fit_sides(frames, 0.16, arguments())
    refused, segment_rows, feature_rows = hierarchy.fit_hierarchy(
        frames, 0.16, arguments()
    )
    fidelity = hierarchy.curve_fidelity(direct, refused)
    assert len(segment_rows) == 8
    assert len(feature_rows) == 64
    assert all(row["symmetric_chamfer_mean_m"] < 0.25 for row in fidelity)
    assert not hierarchy.lane_curve.no_crossing_check(refused)["curves_cross"]


def test_polynomial_lofo_and_pose_geometry_are_reported() -> None:
    frames = frames20()
    result = hierarchy.polynomial_lofo(frames, 2, arguments())
    assert result["count"] > 0
    assert result["rmse_m"] < 0.10
    record = {
        "frames": [
            {
                "frame_id": frame_id,
                "camera_origin_reference_xz_m": [0.0, float(frame_id)],
            }
            for frame_id in range(20)
        ]
    }
    geometry = hierarchy.pose_geometry(record)
    assert geometry["path_length_m"] == 19.0
    assert geometry["maximum_deviation_from_endpoint_chord_m"] == 0.0
