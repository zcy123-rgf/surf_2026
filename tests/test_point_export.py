from __future__ import annotations

import numpy as np
import pytest

from surf_bev.point_export import (
    build_point_decision_rows,
    build_point_sets_document,
    rows_for_method,
)


def sample():
    points = np.asarray(
        [[-1.2, 4.0], [-1.1, 8.0], [1.0, 5.0], [1.1, 9.0]],
        dtype=np.float64,
    )
    ranges = [
        {
            "frame_index": 0,
            "frame_id": 12,
            "side": "left",
            "lane_index": 2,
            "start": 0,
            "end": 2,
        },
        {
            "frame_index": 1,
            "frame_id": 13,
            "side": "right",
            "lane_index": 4,
            "start": 2,
            "end": 4,
        },
    ]
    masks = {
        "raw": np.ones(4, dtype=bool),
        "denoised": np.asarray([True, False, False, True]),
    }
    return points, ranges, masks


def test_decision_rows_are_traceable_to_frame_side_and_source_point():
    points, ranges, masks = sample()
    rows = build_point_decision_rows(points, ranges, masks)
    assert [row["point_id"] for row in rows] == [0, 1, 2, 3]
    assert rows[1] == {
        "point_id": 1,
        "frame_index": 0,
        "frame_id": 12,
        "side": "left",
        "lane_index": 2,
        "point_index_in_lane": 1,
        "x_m": -1.1,
        "z_m": 8.0,
        "raw": 1,
        "denoised": 0,
    }
    kept = rows_for_method(rows, "denoised", kept=True)
    rejected = rows_for_method(rows, "denoised", kept=False)
    assert [row["point_id"] for row in kept] == [0, 3]
    assert [row["point_id"] for row in rejected] == [1, 2]


def test_json_document_contains_exact_kept_and_rejected_coordinates():
    points, ranges, masks = sample()
    document = build_point_sets_document(
        points,
        ranges,
        masks,
        coordinate_system="test X/Z metres",
        metadata={"reference_frame_id": 13},
    )
    method = document["methods"]["denoised"]
    assert method["kept_points"] == 2
    assert method["rejected_points"] == 2
    assert method["frames"][0]["lanes"][0]["kept_points_xz_m"] == [
        [-1.2, 4.0]
    ]
    assert method["frames"][0]["lanes"][0]["rejected_points_xz_m"] == [
        [-1.1, 8.0]
    ]


def test_export_rejects_misaligned_ranges_or_masks():
    points, ranges, masks = sample()
    bad_ranges = [{**ranges[0], "start": 1}]
    with pytest.raises(ValueError, match="cover"):
        build_point_decision_rows(points, bad_ranges, masks)
    with pytest.raises(ValueError, match="entries"):
        build_point_decision_rows(points, ranges, {"bad": np.ones(3)})
