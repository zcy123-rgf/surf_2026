from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "run_weekly_lane_hierarchy.py"
SPEC = importlib.util.spec_from_file_location("weekly_lane_hierarchy", MODULE_PATH)
assert SPEC and SPEC.loader
weekly = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = weekly
SPEC.loader.exec_module(weekly)


def straight_path() -> weekly.FrenetPath:
    points = np.column_stack([np.zeros(11), np.linspace(0.0, 50.0, 11)])
    return weekly.FrenetPath(points_xz=points, cumulative_s=np.linspace(0.0, 50.0, 11))


def test_frenet_round_trip_on_straight_path() -> None:
    path = straight_path()
    original = np.array([[-2.0, -5.0], [1.5, 7.0], [3.0, 55.0]])
    sd = weekly.xz_to_frenet(original, path)
    reconstructed = weekly.frenet_to_xz(sd, path)
    np.testing.assert_allclose(reconstructed, original, atol=1e-9)


def test_sparse_reconstruction_is_exact_for_linear_offset() -> None:
    path = straight_path()
    progress = np.linspace(-5.0, 55.0, 30)
    offsets = 1.0 + 0.02 * progress
    fit = weekly.fit_scalar_spline(
        "right",
        [(0, np.column_stack([progress, offsets]))],
        bin_size_m=2.0,
        smoothing_per_point_m2=1e-8,
        huber_delta_m=0.2,
        irls_iterations=2,
    )
    metrics, anchors = weekly.sparse_reconstruction_trial(fit, 4, path, 200)
    assert len(anchors) == 4
    assert metrics["max_reconstruction_error_m"] < 1e-5


def test_curve_width_check_detects_non_crossing_lanes() -> None:
    progress = np.linspace(0.0, 50.0, 30)
    args = type(
        "Args",
        (),
        {"bin_size_m": 2.0, "huber_delta_m": 0.2, "irls_iterations": 2},
    )()
    groups = []
    for index in range(6):
        jitter = index * 0.001
        groups.append(
            {
                "group_id": index,
                "lanes_sd": [
                    np.column_stack([progress, np.full_like(progress, -1.8 + jitter)]),
                    np.column_stack([progress, np.full_like(progress, 1.8 + jitter)]),
                ],
            }
        )
    fits = weekly.fit_both_sides(groups, 0.01, args)
    result = weekly.curve_width_check(fits)
    assert result["curves_cross"] is False
    assert 3.5 < result["median_right_minus_left_m"] < 3.7


def test_piecewise_sparse_blend_preserves_overlapping_local_curves() -> None:
    first_s = np.linspace(0.0, 20.0, 8)
    second_s = np.linspace(10.0, 30.0, 8)
    groups = [
        {
            "group_id": 0,
            "lanes_sd": [
                np.column_stack([first_s, -2.0 + 0.01 * first_s]),
                np.column_stack([first_s, 2.0 + 0.01 * first_s]),
            ],
        },
        {
            "group_id": 1,
            "lanes_sd": [
                np.column_stack([second_s, -2.0 + 0.01 * second_s]),
                np.column_stack([second_s, 2.0 + 0.01 * second_s]),
            ],
        },
    ]
    fits = weekly.build_piecewise_sparse_blends(groups)
    progress = np.linspace(0.0, 30.0, 100)
    np.testing.assert_allclose(
        fits["left"].evaluate(progress), -2.0 + 0.01 * progress, atol=1e-10
    )
    np.testing.assert_allclose(
        fits["right"].evaluate(progress), 2.0 + 0.01 * progress, atol=1e-10
    )


def test_windows_wrapper_uses_new_output_and_registered_design() -> None:
    text = (ROOT / "scripts" / "run_weekly_meeting_complete_windows.ps1").read_text(
        encoding="utf-8"
    )
    assert "weekly_lane_hierarchy_$Stamp" in text
    assert "hierarchy_overlap_selection_*" in text
    assert "selection_status -ne \"selected\"" in text
    assert "meeting_result_bundle.zip" in text
    assert "manual_annotation_package = $AnnotationZip" in text


def test_full_synthetic_ten_window_run(tmp_path, monkeypatch) -> None:
    frame_count = 105
    poses = tmp_path / "00.txt"
    pose_rows = []
    for frame_id in range(frame_count):
        pose_rows.append(
            "1 0 0 0 0 1 0 0 0 0 1 " + str(frame_id * 0.5)
        )
    poses.write_text("\n".join(pose_rows) + "\n", encoding="utf-8")
    calib = tmp_path / "calib.txt"
    calib.write_text(
        "P2: 700 0 600 0 0 700 180 0 0 0 1 0\n", encoding="utf-8"
    )

    local_z = np.linspace(3.0, 30.0, 12)
    point_frames = []
    for frame_id in range(frame_count):
        bend = 0.0005 * (frame_id + local_z) ** 2
        left = np.column_stack([-1.8 + bend, local_z])
        right = np.column_stack([1.8 + bend, local_z])
        point_frames.append(
            {
                "frame_id": frame_id,
                "eligible_for_two_curve_fit": True,
                "selected_lanes_local_ground_xz_m": [left.tolist(), right.tolist()],
            }
        )
    points = tmp_path / "selected_lane_points.json"
    points.write_text(json.dumps({"frames": point_frames}), encoding="utf-8")

    blocks = []
    for index in range(10):
        start = index * 10
        blocks.append(
            {
                "block_index": index,
                "start_frame": start,
                "end_frame": start + 14,
                "valid_frames": 15,
            }
        )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "status": "selected",
                "dataset": "KITTI Odometry Sequence synthetic",
                "block_count": 10,
                "block_size": 15,
                "block_stride": 10,
                "minimum_valid_frames_per_block_required": 5,
                "selected": {
                    "start_frame": 0,
                    "end_frame": 104,
                    "selected_lane_points_json": str(points),
                },
                "blocks": blocks,
                "package_zip": str(tmp_path / "manual.zip"),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_weekly_lane_hierarchy.py",
            "--selection-json",
            str(selection),
            "--poses",
            str(poses),
            "--calib",
            str(calib),
            "--output-dir",
            str(output),
            "--maximum-cv-folds",
            "5",
            "--curve-samples",
            "120",
        ],
    )
    weekly.main()
    status = json.loads((output / "STATUS.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    assert status["dataset"] == "KITTI Odometry Sequence synthetic"
    assert status["windows_completed"] == 10
    assert status["manual_evaluation_status"] == "pending_manual_annotation"
    assert (output / "01_q1_two_curves" / "final_two_curves.png").is_file()
    assert (output / "03_q3_sparse_refusion" / "sparse_anchors.csv").is_file()
    assert (
        output / "03_q3_sparse_refusion" / "legacy_global_fidelity_to_direct.csv"
    ).is_file()
    assert (output / "MEETING_SUMMARY.md").is_file()
