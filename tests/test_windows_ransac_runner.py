from __future__ import annotations

import json
from pathlib import Path

import scripts.evaluate_denoise_methods as denoise


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_ransac_improvement_windows.ps1"
POSE_RUNNER = ROOT / "scripts" / "run_kitti00_first5_windows.ps1"
ANNOTATION = ROOT / "annotations" / "kitti00_first5_manual_annotations.json"
DATASET_HELPER = ROOT / "scripts" / "kitti00_workstation_input.ps1"
FROM_SCRATCH_SCRIPTS = [
    RUNNER,
    POSE_RUNNER,
    ROOT / "scripts" / "evaluate_denoise_methods.py",
    ROOT / "scripts" / "optimize_ransac_parameters.py",
    ROOT / "scripts" / "optimize_ransac_safety_expansion.py",
    ROOT / "scripts" / "audit_ransac_shortlist.py",
    ROOT / "scripts" / "optimize_temporal_denoise.py",
]


def test_windows_runner_never_reads_historical_results() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "results\\kitti00" not in text
    assert "F:\\BaiduNetdiskDownload\\kitti\\odometry" in text
    assert "Resolve-Kitti00WorkstationInput" in text
    assert "input_data\\kitti00_first5" not in text
    assert "workstation_outputs\\ransac_improvement_" in text
    assert "run_full_point_pipeline.py" in text
    assert "evaluate_denoise_methods.py" in text

    pose_text = POSE_RUNNER.read_text(encoding="utf-8")
    assert "results\\kitti00" not in pose_text
    assert "F:\\BaiduNetdiskDownload\\kitti\\odometry" in pose_text
    assert "Resolve-Kitti00WorkstationInput" in pose_text
    assert "workstation_outputs\\kitti00_first5_$Stamp" in pose_text


def test_external_kitti_input_is_required_and_hash_checked() -> None:
    assert ANNOTATION.is_file()
    assert not (ROOT / "workstation_input" / "kitti00_first5").exists()
    helper = DATASET_HELPER.read_text(encoding="utf-8")
    assert "data_odometry_color\\dataset\\sequences\\00\\image_2" in helper
    assert "data_odometry_calib\\dataset\\sequences\\00\\calib.txt" in helper
    assert "data_odometry_poses\\dataset\\poses\\00.txt" in helper
    assert "Sequence 00 image_2 should contain 4541 PNG files" in helper
    for expected_hash in (
        "af34528f0edd14ddf8ee7ac9017f4b7f540e04e3d50cecce006245d950ea1d6c",
        "cc8a7a3c3fa653b5edec3ac81dd7bda4e5391330c82070d8bc6c3e2831572263",
        "18504a8bf4e71a2273cc31289f0c3fdd1a46942fdd59125a14bb14da58457522",
        "0766814389f2650b556913b07f3b1e2344ad9f1e6e9defef7ccceddd22e2562a",
        "b0d2c7364bdfb75b099d48303ff5e296b1004aa2b5969d81383b8c36c62b2c49",
    ):
        assert expected_hash in helper


def test_from_scratch_scripts_do_not_read_generated_result_roots() -> None:
    forbidden = ("results/kitti00", "denoise_experiments")
    for script in FROM_SCRATCH_SCRIPTS:
        text = script.read_text(encoding="utf-8").replace("\\", "/")
        assert all(item not in text for item in forbidden), script


def test_loader_accepts_from_scratch_pipeline_schema(tmp_path: Path) -> None:
    source = tmp_path / "detected_lane_points.json"
    source.write_text(
        json.dumps(
            {
                "detector": {
                    "frames": [
                        {
                            "frame_id": frame_id,
                            "candidate_lanes_image_xy_px": [],
                            "selected": [
                                {
                                    "side": "left",
                                    "image_xy_px": [[10, 100], [20, 50]],
                                },
                                {
                                    "side": "right",
                                    "image_xy_px": [[90, 100], [80, 50]],
                                },
                            ],
                        }
                        for frame_id in range(5)
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    lanes = denoise.load_image_lanes(source)
    assert len(lanes) == 5
    assert all(len(frame) == 2 for frame in lanes)
    assert lanes[0][0].shape == (2, 2)
