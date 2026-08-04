"""Run CLRNet, point IPM and pose alignment on KITTI 00 frames 0-4."""

from __future__ import annotations

from common import (
    CALIB,
    CLRNET_ROOT,
    IMAGE_DIR,
    POSES,
    new_output,
    record_latest_pipeline,
    require_directory,
    require_file,
    run_main,
)
from scripts import run_full_point_pipeline


if __name__ == "__main__":
    require_directory(IMAGE_DIR, "KITTI 00 image directory")
    require_file(CALIB, "KITTI 00 calibration")
    require_file(POSES, "KITTI 00 poses")
    require_directory(CLRNET_ROOT / "clrnet", "CLRNet source")
    require_file(CLRNET_ROOT / "weights" / "culane_r18.pth", "CLRNet checkpoint")

    output = new_output("01_pipeline")
    run_main(
        run_full_point_pipeline.main,
        [
            "--image-dir",
            str(IMAGE_DIR),
            "--image-pattern",
            "{frame_id:06d}.png",
            "--calib",
            str(CALIB),
            "--poses",
            str(POSES),
            "--output-dir",
            str(output),
            "--frame-ids",
            "0,1,2,3,4",
            "--reference-id",
            "4",
            "--local-x-range=-10,10",
            "--local-z-range=3,50",
            "--fusion-x-range=-15,15",
            "--fusion-z-range=-10,50",
            "--weights",
            "0.2,0.4,0.6,0.8,1.0",
            "--clrnet-root",
            str(CLRNET_ROOT),
            "--device",
            "cuda",
        ],
    )
    files = record_latest_pipeline(output)
    print("Latest pipeline output:", files["output"])
    print("Detected image points:", files["detected"])
    print("Pose-aligned metric points:", files["aligned"])
