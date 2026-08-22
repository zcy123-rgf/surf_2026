"""Run the independently safety-audited RANSAC configuration without retuning.

This is a small production-style entry point for the RANSAC-only reference.
It consumes five-frame CLRNet image-point JSON, projects and pose-aligns the
points with the same functions used by the audited search, applies the fixed
configuration selected by the safety-expanded audit, and writes a new result
directory.  It does not run B-splines, temporal denoisers, or a parameter grid.

The configuration is *not* claimed to be KITTI lane-line accurate.  It passed
the recorded anti-over-deletion and injected-outlier constraints on the five
frame sample; far-range gradual drift remains a known failure mode.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.evaluate_denoise_methods as base  # noqa: E402
from surf_bev.optimized_ransac import (  # noqa: E402
    SideAwarePolynomialRansac,
)


SELECTED_CONFIG = {
    "degree": 3,
    "base_threshold": 0.35,
    "distance_slope": 0.015,
    "threshold_cap": 0.80,
    "iterations": 1000,
    "minimum_sample_z_span": 10.0,
    "score_mode": "balanced",
    "local_refinements": 1,
}

CONFIG_SOURCE = (
    "kitti00_first5_ransac_safety_expanded_20260727: "
    "d3_b0.350_s0.015_c0.80_i1000_z10.0_qbalanced_r1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clrnet-json", type=Path, required=True)
    parser.add_argument(
        "--manual-json",
        type=Path,
        default=ROOT / "annotations" / "kitti00_first5_manual_annotations.json",
    )
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=base.SEED + 999_999)
    return parser.parse_args()


def decision_rows(
    points: np.ndarray,
    ranges: list[dict[str, object]],
    keep: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in ranges:
        start, end = int(item["start"]), int(item["end"])
        for local_index, global_index in enumerate(range(start, end)):
            rows.append(
                {
                    "frame_id": int(item["frame_id"]),
                    "side": str(item["side"]),
                    "point_index_in_lane": local_index,
                    "x_right_reference_m": float(points[global_index, 0]),
                    "z_forward_reference_m": float(points[global_index, 1]),
                    "kept": bool(keep[global_index]),
                }
            )
    return rows


def kept_points_document(
    points: np.ndarray,
    ranges: list[dict[str, object]],
    keep: np.ndarray,
) -> dict[str, object]:
    frames: dict[int, dict[str, object]] = {}
    for item in ranges:
        frame_id = int(item["frame_id"])
        side = str(item["side"])
        start, end = int(item["start"]), int(item["end"])
        frame = frames.setdefault(frame_id, {"frame_id": frame_id})
        frame[side] = points[start:end][keep[start:end]].tolist()
    return {
        "coordinate_system": (
            "reference image-camera ground X/Z metres; X right, Z forward; "
            "reference frame 4"
        ),
        "method": "fixed safety-audited side-aware polynomial RANSAC",
        "configuration": SELECTED_CONFIG,
        "frames": [frames[index] for index in sorted(frames)],
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    base.require_empty_output(args.output_dir)
    required = [args.clrnet_json, args.manual_json, args.calib, args.poses]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))
    calibration = base.parse_projection(args.calib)
    poses_cam0 = base.load_kitti_poses(args.poses)
    poses = [pose @ calibration["cam0_from_image"] for pose in poses_cam0]
    clrnet_lanes = base.load_image_lanes(args.clrnet_json)
    manual_lanes = base.load_manual_lanes(args.manual_json)
    _, points, ranges = base.align_image_lanes(
        clrnet_lanes, calibration["K"], poses
    )
    _, manual_points, _ = base.align_image_lanes(
        manual_lanes, calibration["K"], poses
    )
    evaluator = SideAwarePolynomialRansac(points, ranges, args.seed)
    keep, details = evaluator.apply(SELECTED_CONFIG)
    frames = base.frames_from_mask(points, ranges, keep)
    _, fusion = base.lane_mask(frames, *base.FUSION_RANGE)
    metrics = base.method_metrics(
        "selected_safety_ransac", points, ranges, keep, manual_points, fusion
    )

    base.imwrite(
        args.output_dir / "01_points" / "points_by_frame.png",
        base.draw_points(frames),
    )
    base.plot_points(
        args.output_dir / "01_points" / "metric_plot.png",
        frames,
        "Fixed safety-audited side-aware cubic RANSAC",
    )
    base.imwrite(
        args.output_dir / "02_fusion" / "weighted_heatmap.png",
        fusion["heatmap"],
    )
    base.imwrite(
        args.output_dir / "02_fusion" / "weighted_binary.png",
        fusion["binary"],
    )
    base.write_csv(
        args.output_dir / "03_data" / "point_decisions.csv",
        decision_rows(points, ranges, keep),
    )
    (args.output_dir / "03_data" / "denoised_points.json").write_text(
        json.dumps(
            kept_points_document(points, ranges, keep),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    audit = {
        "status": "complete",
        "method_family": "RANSAC only; no B-spline or temporal denoiser",
        "configuration_source": CONFIG_SOURCE,
        "configuration": SELECTED_CONFIG,
        "seed": int(args.seed),
        "inputs": {
            "clrnet_json": {
                "path": str(args.clrnet_json.resolve()),
                "sha256": base.sha256(args.clrnet_json),
            },
            "manual_json": {
                "path": str(args.manual_json.resolve()),
                "sha256": base.sha256(args.manual_json),
            },
            "calib": {
                "path": str(args.calib.resolve()),
                "sha256": base.sha256(args.calib),
            },
            "poses": {
                "path": str(args.poses.resolve()),
                "sha256": base.sha256(args.poses),
            },
        },
        "points": {
            "input": int(len(points)),
            "kept": int(np.count_nonzero(keep)),
            "retention": float(np.mean(keep)),
        },
        "metrics": metrics,
        "model_details": details,
        "limitations": [
            "This five-frame sample has no official lane-line ground truth.",
            "Manual annotations are only an anti-over-deletion pseudo-reference.",
            "Injected-outlier tests are not KITTI lane-line accuracy.",
            "Far-range gradual drift had low recall in the independent audit.",
        ],
    }
    audit_path = args.output_dir / "00_audit" / "result.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
