"""End-to-end output smoke test without loading the large CLRNet checkpoint."""

import importlib.util
import json
import sys
import tempfile
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


def main() -> None:
    saved = json.loads(
        (
            RESULT
            / "07_manual_comparison"
            / "data"
            / "clrnet_lanes_kitti_ordered.json"
        ).read_text(encoding="utf-8")
    )

    class SavedPointDetector:
        def __init__(self, **_kwargs):
            self.index = 0

        def detect(self, image):
            item = saved["items"][self.index]
            self.index += 1
            return {
                "lanes": [
                    np.asarray(lane, dtype=np.float64)
                    for lane in item["lanes_image_xy_px"]
                ],
                "image_rgb": image[:, :, ::-1],
            }

    RUNNER.CLRNetLaneDetector = SavedPointDetector
    with tempfile.TemporaryDirectory(prefix="surf_full_point_smoke_") as temp:
        temp_dir = Path(temp)
        checkpoint = temp_dir / "fake_checkpoint.pth"
        checkpoint.write_bytes(b"smoke-test-only")
        output = temp_dir / "fresh_output"
        previous_argv = sys.argv
        sys.argv = [
            "run_full_point_pipeline.py",
            "--image-dir",
            str(RESULT / "01_original_frames"),
            "--image-pattern",
            "frame_{frame_id:06d}.png",
            "--calib",
            str(RESULT / "metadata" / "calib.txt"),
            "--poses",
            str(RESULT / "metadata" / "poses_00_first5.txt"),
            "--provenance",
            str(RESULT / "audit" / "kitti_first5_provenance.json"),
            "--output-dir",
            str(output),
            "--clrnet-checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
        ]
        try:
            RUNNER.main()
        finally:
            sys.argv = previous_argv

        required = [
            output / "04_pose_aligned_points" / "five_frame_metric_pose_fusion.png",
            output / "05_fusion_without_denoise" / "accumulated_points_by_frame.png",
            output / "05_fusion_without_denoise" / "weighted_score_heatmap.png",
            output / "00_metadata" / "pose_alignment.csv",
            output / "00_metadata" / "aligned_lane_points.json",
            output / "00_metadata" / "audit.json",
            output / "00_metadata" / "manifest.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        assert not missing, missing
        status = json.loads(
            (output / "00_metadata" / "run_status.json").read_text(
                encoding="utf-8"
            )
        )
        assert status["status"] == "complete"
        audit = json.loads(
            (output / "00_metadata" / "audit.json").read_text(encoding="utf-8")
        )
        assert audit["pose_alignment"]["reference_frame"] == 4
        assert len(audit["pose_alignment"]["frames"]) == 5
        print("Fresh-output end-to-end smoke test passed.")


if __name__ == "__main__":
    main()
