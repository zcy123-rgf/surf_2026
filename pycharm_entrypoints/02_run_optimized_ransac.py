"""Run the fixed improved-RANSAC reference on the latest pipeline points."""

from __future__ import annotations

from common import (
    CALIB,
    POSES,
    PROJECT_ROOT,
    latest_pipeline_files,
    new_output,
    require_file,
    run_main,
)
from scripts import run_selected_ransac_reference


if __name__ == "__main__":
    files = latest_pipeline_files()
    manual = PROJECT_ROOT / "annotations" / "kitti00_first5_manual_annotations.json"
    require_file(CALIB, "KITTI 00 calibration")
    require_file(POSES, "KITTI 00 poses")
    require_file(manual, "manual pseudo-reference")

    output = new_output("02_ransac")
    run_main(
        run_selected_ransac_reference.main,
        [
            "--clrnet-json",
            str(files["detected"]),
            "--manual-json",
            str(manual),
            "--calib",
            str(CALIB),
            "--poses",
            str(POSES),
            "--output-dir",
            str(output),
        ],
    )
    print("RANSAC output:", output)
