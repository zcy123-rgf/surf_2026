import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_full_point_pipeline", ROOT / "scripts" / "run_full_point_pipeline.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)

RESULT = ROOT / "results" / "kitti00_first5_lane_bev_fusion"
CALIB = RESULT / "metadata" / "calib.txt"
POSES = RESULT / "metadata" / "poses_00_first5.txt"
IMAGES = RESULT / "01_original_frames"
PROVENANCE = RESULT / "audit" / "kitti_first5_provenance.json"


def test_p2_camera_offset_is_not_discarded():
    calibration = RUNNER.parse_kitti_projection(CALIB)
    assert calibration["key"] == "P2"
    projection = calibration["projection"]
    transform = calibration["T_cam_image_cam0"]
    expected = np.linalg.solve(projection[:, :3], projection[:, 3])
    np.testing.assert_allclose(transform[:3, 3], expected, atol=1e-12)
    assert abs(transform[0, 3]) > 0.01


def test_first_five_poses_align_to_frame_four():
    calibration = RUNNER.parse_kitti_projection(CALIB)
    poses_cam0 = RUNNER.load_kitti_poses(POSES)
    poses = RUNNER.poses_for_image_camera(
        poses_cam0, calibration["T_cam0_cam_image"]
    )
    reference = poses[4]
    relatives = [np.linalg.inv(reference) @ poses[index] for index in range(5)]
    origins_z = np.asarray([pose[2, 3] for pose in relatives])
    np.testing.assert_allclose(relatives[-1], np.eye(4), atol=1e-9)
    assert np.all(np.diff(origins_z) > 0.8)
    assert -3.6 < origins_z[0] < -3.2
    assert abs(origins_z[-1]) < 1e-9


def test_official_first_five_image_hashes_match_audit():
    paths = [IMAGES / f"frame_{index:06d}.png" for index in range(5)]
    result = RUNNER.verify_provenance(PROVENANCE, paths, list(range(5)))
    assert result is not None
    assert result["verified_frames"] == [0, 1, 2, 3, 4]


def test_extended_fusion_range_keeps_points_behind_reference():
    points = np.asarray([[0.0, -3.4], [0.0, 10.0], [0.0, 55.0]])
    kept = RUNNER.filter_metric_points(points, (-15.0, 15.0), (-10.0, 50.0))
    np.testing.assert_allclose(kept, points[:2])
