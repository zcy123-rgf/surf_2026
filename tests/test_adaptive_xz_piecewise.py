import argparse
import json
from pathlib import Path

import numpy as np

from scripts import fit_adaptive_xz_piecewise as adaptive
from surf_bev.geometry import transform_lane_points_by_pose


def synthetic_route(frame_count: int = 45) -> tuple[np.ndarray, np.ndarray]:
    step_headings = np.zeros(frame_count - 1, dtype=np.float64)
    step_headings[14:29] = np.linspace(0.0, np.pi / 2.0, 15)
    step_headings[29:] = np.pi / 2.0
    positions = [np.array([0.0, 0.0], dtype=np.float64)]
    for heading in step_headings:
        positions.append(
            positions[-1] + np.array([np.sin(heading), np.cos(heading)])
        )
    positions_xz = np.asarray(positions)
    pose_headings = np.concatenate([step_headings, step_headings[-1:]])
    poses = []
    for position, heading in zip(positions_xz, pose_headings):
        cosine = np.cos(heading)
        sine = np.sin(heading)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = np.array(
            [
                [cosine, 0.0, sine],
                [0.0, 1.0, 0.0],
                [-sine, 0.0, cosine],
            ]
        )
        pose[[0, 2], 3] = position
        poses.append(pose)
    return positions_xz, np.asarray(poses)


def write_synthetic_inputs(root: Path) -> tuple[Path, Path, Path]:
    route, poses = synthetic_route()
    pose_path = root / "01.txt"
    pose_path.write_text(
        "\n".join(
            " ".join(f"{value:.12e}" for value in pose[:3, :4].reshape(-1))
            for pose in poses
        ),
        encoding="utf-8",
    )
    calib_path = root / "calib.txt"
    calib_path.write_text(
        "P2: 7.000000e+02 0 6.000000e+02 0 0 7.000000e+02 1.800000e+02 0 0 0 1 0\n",
        encoding="utf-8",
    )

    pose0 = poses[0]
    frames = []
    for frame_id in range(len(route)):
        lower = max(0, frame_id - 5)
        upper = min(len(route), frame_id + 9)
        centre = route[lower:upper]
        if len(centre) < 5:
            raise AssertionError("Synthetic support must have at least five points.")
        local_steps = np.gradient(centre, axis=0)
        tangent = local_steps / np.maximum(
            np.linalg.norm(local_steps, axis=1, keepdims=True), 1e-9
        )
        right_normal = np.column_stack([tangent[:, 1], -tangent[:, 0]])
        left_global = centre - 1.8 * right_normal
        right_global = centre + 1.8 * right_normal
        left_local = transform_lane_points_by_pose(
            left_global, pose0, poses[frame_id], camera_height=1.65, pitch_deg=0.0
        )
        right_local = transform_lane_points_by_pose(
            right_global, pose0, poses[frame_id], camera_height=1.65, pitch_deg=0.0
        )
        frames.append(
            {
                "frame_id": frame_id,
                "candidate_count": 2,
                "eligible_for_two_curve_fit": True,
                "selected_lanes_local_ground_xz_m": [
                    left_local.tolist(),
                    right_local.tolist(),
                ],
            }
        )
    points_path = root / "selected_lane_points.json"
    points_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "coordinate_system": "per-frame image-camera ground X/Z in metres",
                "camera_height_m": 1.65,
                "pitch_deg": 0.0,
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )
    return points_path, pose_path, calib_path


def test_window_classification_uses_pose_heading_change() -> None:
    route, _ = synthetic_route()
    straight = adaptive.trajectory_turn_statistics(route[:15])
    curve = adaptive.trajectory_turn_statistics(route[14:30])
    assert adaptive.classify_window(
        straight["trajectory_net_heading_change_deg"], 1.5, 3.0
    ) == "straight"
    assert adaptive.classify_window(
        curve["trajectory_net_heading_change_deg"], 1.5, 3.0
    ) == "curve"


def test_micro_turn_noise_is_diagnostic_not_classification_signal() -> None:
    z = np.arange(31, dtype=np.float64)
    x = 0.03 * (np.arange(31) % 2)
    stats = adaptive.trajectory_turn_statistics(np.column_stack([x, z]))
    assert stats["total_absolute_heading_change_deg"] > 1.5
    assert abs(stats["trajectory_net_heading_change_deg"]) < 1.5
    assert adaptive.classify_window(
        stats["trajectory_net_heading_change_deg"], 1.5, 3.0
    ) == "straight"


def test_transition_selection_uses_heldout_rmse_and_simplicity_tie() -> None:
    polynomial = "parametric_polynomial"
    spline = "parametric_cubic_bspline"
    selected, _ = adaptive.select_transition_model(
        {
            polynomial: {"heldout_fold_count": 4, "rmse_m": 0.101},
            spline: {"heldout_fold_count": 4, "rmse_m": 0.098},
        },
        [polynomial, spline],
        tie_m=0.005,
    )
    assert selected == polynomial
    selected, _ = adaptive.select_transition_model(
        {
            polynomial: {"heldout_fold_count": 4, "rmse_m": 0.120},
            spline: {"heldout_fold_count": 4, "rmse_m": 0.098},
        },
        [polynomial, spline],
        tie_m=0.005,
    )
    assert selected == spline


def test_curve_comparison_is_not_hardwired_to_bspline() -> None:
    polynomial = "parametric_polynomial"
    spline = "parametric_cubic_bspline"
    metrics = {
        polynomial: {"heldout_fold_count": 5, "rmse_m": 0.20},
        spline: {"heldout_fold_count": 5, "rmse_m": 0.21},
    }
    selected, reason = adaptive.select_compared_model(
        metrics, [polynomial, spline], tie_m=0.02, context="curve"
    )
    assert selected == polynomial
    assert "curve" in reason


def dummy_window_result(window_id: int, key_start: float) -> adaptive.WindowResult:
    points = np.column_stack(
        [np.full(20, 1.8), np.linspace(key_start, key_start + 10.0, 20)]
    )
    fit = adaptive.ParametricFit(
        model="parametric_polynomial",
        degree=2,
        aggregate_xz=points,
        aggregate_support_frames=np.ones(20, dtype=np.int64),
        parameter=np.linspace(0.0, 1.0, 20),
        curve_xz=points,
        residuals_m=np.zeros(20),
        coefficients={},
        huber_delta_m=0.2,
        irls_iterations=3,
    )
    return adaptive.WindowResult(
        spec=adaptive.WindowSpec(window_id, window_id * 10, window_id * 10 + 14, window_id * 10 + 14, False),
        classification="straight",
        net_heading_change_deg=0.0,
        total_absolute_heading_change_deg=0.0,
        trajectory_path_length_m=14.0,
        side="left",
        selected_model="parametric_polynomial",
        valid_frame_ids=list(range(window_id * 10, window_id * 10 + 15)),
        fit_local=fit,
        curve_common_xz=points,
        aggregate_common_xz=points,
        sample_route_keys=np.linspace(key_start, key_start + 10.0, 20),
    )


def test_nonconsecutive_windows_become_separate_output_segments() -> None:
    first = dummy_window_result(0, 0.0)
    third = dummy_window_result(2, 20.0)
    segments, rows = adaptive.split_continuity_segments(
        [first, third], p95_gate_m=0.75, angle_gate_deg=30.0
    )
    assert len(segments) == 2
    assert first.segment_id == 0
    assert third.segment_id == 1
    assert rows[0]["passes_continuity_gate"] is False
    assert "nonconsecutive_window_ids" in rows[0]["break_reason"]


def test_small_route_key_gap_uses_endpoint_continuity_instead_of_false_break() -> None:
    first = dummy_window_result(0, 0.0)
    second = dummy_window_result(1, 10.05)
    segments, rows = adaptive.split_continuity_segments(
        [first, second],
        p95_gate_m=0.75,
        angle_gate_deg=30.0,
        route_gap_gate_m=0.50,
    )
    assert len(segments) == 1
    assert rows[0]["passes_continuity_gate"] is True
    assert rows[0]["continuity_mode"] == "small_route_gap_endpoint_check"
    assert np.isclose(rows[0]["route_key_gap_m"], 0.05)


def test_pose_polyline_order_is_monotonic_on_acute_curve() -> None:
    angle = np.linspace(0.0, 0.85 * np.pi, 50)
    trajectory = np.column_stack(
        [10.0 - 10.0 * np.cos(angle), 10.0 * np.sin(angle)]
    )
    tangent = np.gradient(trajectory, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    right_normal = np.column_stack([tangent[:, 1], -tangent[:, 0]])
    lane = trajectory + 1.8 * right_normal
    keys = adaptive.polyline_order_key(lane, trajectory)
    assert np.all(np.diff(keys) > -1e-6)
    assert keys[-1] - keys[0] > 20.0


def test_internal_support_gap_splits_blended_output() -> None:
    keys = np.array([0.0, 0.5, 1.0, 7.0, 7.5])
    points = np.column_stack([np.full(len(keys), 1.8), keys])
    pieces = adaptive.split_blended_support(
        keys,
        points,
        np.ones(len(keys), dtype=np.int64),
        maximum_order_gap_frames=3.0,
        maximum_node_gap_m=5.0,
    )
    assert len(pieces) == 2
    assert "internal_order_support_gap" in pieces[1][3]
    assert "internal_spatial_node_gap" in pieces[1][3]


def test_isolated_single_node_piece_is_not_exported_as_a_curve() -> None:
    keys = np.array([0.0, 0.5, 1.0, 2.0])
    points = np.array([[1.8, 0.0], [1.8, 0.5], [1.8, 1.0], [20.0, 2.0]])
    pieces = adaptive.split_blended_support(
        keys,
        points,
        np.ones(len(keys), dtype=np.int64),
        maximum_order_gap_frames=3.0,
        maximum_node_gap_m=5.0,
    )
    assert len(pieces) == 1
    assert len(pieces[0][0]) == 3


def test_coverage_and_partial_side_slots_are_auditable() -> None:
    assert adaptive.longest_missing_run(0, 14, [0, 1, 2, 6, 7, 8, 9]) == 5
    partial = {
        "selected_lanes_local_ground_xz_m": [
            [],
            [[2.0, 3.0], [2.0, 4.0], [2.0, 5.0], [2.0, 6.0]],
        ]
    }
    assert adaptive.valid_local_lane(partial, 0) is None
    assert adaptive.valid_local_lane(partial, 1).shape == (4, 2)


def test_robust_models_return_metric_xz_curves() -> None:
    z = np.linspace(0.0, 30.0, 50)
    straight = np.column_stack([1.8 + 0.01 * z, z])
    straight[20] += np.array([2.0, -1.0])
    support = np.full(len(straight), 5, dtype=np.int64)
    polynomial = adaptive.fit_parametric_polynomial(
        straight, support, 2, 0.20, 5, 120
    )
    assert polynomial.curve_xz.shape == (120, 2)
    assert np.median(polynomial.residuals_m) < 0.05

    angle = np.linspace(0.0, np.pi / 2.0, 70)
    curved = np.column_stack([12.0 - 12.0 * np.cos(angle), 12.0 * np.sin(angle)])
    spline = adaptive.fit_parametric_bspline(
        curved, np.full(len(curved), 5), 0.0025, 0.20, 4, 140
    )
    assert spline.curve_xz.shape == (140, 2)
    assert np.percentile(spline.residuals_m, 95) < 0.05


def test_partial_fixed_slots_are_consumed_independently() -> None:
    z = np.linspace(3.0, 20.0, 8)
    left = np.column_stack([np.full_like(z, -1.8), z]).tolist()
    right = np.column_stack([np.full_like(z, 1.8), z]).tolist()
    selected = {
        0: {"frame_id": 0, "selected_lanes_local_ground_xz_m": [left, []]},
        1: {"frame_id": 1, "selected_lanes_local_ground_xz_m": [[], right]},
        2: {"frame_id": 2, "selected_lanes_local_ground_xz_m": [left, []]},
        3: {"frame_id": 3, "selected_lanes_local_ground_xz_m": [[], right]},
    }
    poses = [np.eye(4, dtype=np.float64) for _ in range(4)]
    spec = adaptive.WindowSpec(0, 0, 3, 3, False)

    left_observations = adaptive.align_window_observations(
        spec, 0, selected, poses, camera_height=1.65, pitch_deg=0.0
    )
    right_observations = adaptive.align_window_observations(
        spec, 1, selected, poses, camera_height=1.65, pitch_deg=0.0
    )

    assert [frame_id for frame_id, _ in left_observations] == [0, 2]
    assert [frame_id for frame_id, _ in right_observations] == [1, 3]
    assert all(points.shape == (8, 2) for _, points in left_observations)
    assert all(points.shape == (8, 2) for _, points in right_observations)


def test_synthetic_straight_curve_straight_run(tmp_path: Path) -> None:
    points, poses, calib = write_synthetic_inputs(tmp_path)
    output = tmp_path / "output"
    args = argparse.Namespace(
        selected_lane_points=points,
        scan_json=None,
        poses=poses,
        calib=calib,
        sequence_id="01",
        start_frame=0,
        end_frame=44,
        reference_frame=44,
        core_end_frame=29,
        window_length=15,
        window_stride=10,
        straight_max_heading_deg=1.5,
        curve_min_heading_deg=3.0,
        transition_rmse_tie_m=0.005,
        curve_model_policy="compare",
        curve_rmse_tie_m=0.02,
        curvature_windows_csv=None,
        polynomial_degree=2,
        bin_size_m=0.50,
        smoothing_per_point_m2=0.01,
        huber_delta_m=0.20,
        irls_iterations=3,
        curve_samples=100,
        minimum_valid_frames_per_side=4,
        maximum_missing_run_frames=3,
        maximum_cv_folds=2,
        blend_key_bin_width=0.25,
        blend_maximum_route_distance_m=12.0,
        continuity_p95_gate_m=0.75,
        continuity_angle_gate_deg=30.0,
        continuity_route_gap_gate_m=0.50,
        maximum_internal_order_gap_frames=3.0,
        maximum_internal_node_gap_m=5.0,
        camera_height=None,
        pitch_deg=None,
        output_dir=output,
    )
    result = adaptive.run(args)
    assert result["status"] == "complete"
    plan = (output / "window_plan.csv").read_text(encoding="utf-8-sig")
    assert "parametric_polynomial" in plan
    assert "parametric_cubic_bspline" in plan
    assert (output / "adaptive_piecewise_xz_overview.png").stat().st_size > 1000
    blended_text = (output / "blended_lane_nodes.csv").read_text(
        encoding="utf-8-sig"
    )
    assert len(blended_text) > 100
    assert "segment_id" in blended_text.splitlines()[0]
    assert result["classification"]["total_absolute_heading_change_is_diagnostic_only"]
    assert result["experiment_sections"] == {
        "core_frames": [0, 29],
        "exit_extension_frames": [30, 44],
    }

    generated_text = "\n".join(
        path.read_text(encoding="utf-8-sig")
        for path in output.iterdir()
        if path.suffix.lower() in {".json", ".csv", ".md"}
    ).lower()
    assert "frenet" not in generated_text
    assert result["parameter_policy"].endswith("not spatial coordinates")
