from __future__ import annotations

import json
from pathlib import Path

import scripts.evaluate_denoise_methods as denoise


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_ransac_improvement_windows.ps1"
FROM_SCRATCH_SCRIPTS = [
    RUNNER,
    ROOT / "scripts" / "evaluate_denoise_methods.py",
    ROOT / "scripts" / "optimize_ransac_parameters.py",
    ROOT / "scripts" / "optimize_ransac_safety_expansion.py",
    ROOT / "scripts" / "audit_ransac_shortlist.py",
    ROOT / "scripts" / "optimize_temporal_denoise.py",
]


def test_windows_runner_never_reads_historical_results() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "results\\kitti00" not in text
    assert "input_data\\kitti00_first5" in text
    assert "workstation_outputs\\ransac_improvement_" in text
    assert "run_full_point_pipeline.py" in text
    assert "evaluate_denoise_methods.py" in text


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
