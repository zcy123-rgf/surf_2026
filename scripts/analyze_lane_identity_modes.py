"""Compare frame-local ego-adjacent selection with pose-temporal association.

The comparison deliberately does not call a CLRNet candidate index a lane ID.
Candidate indices are frame-local.  Continuity is measured after transforming
the previous selected metric curve into the current image-camera frame with
KITTI poses.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scan_clrnet_lane_counts import (  # noqa: E402
    load_calibration,
    symmetric_curve_distance_m,
)
from surf_bev.geometry import transform_lane_points_by_pose  # noqa: E402


SIDES = ("left", "right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ego-json", type=Path, required=True)
    parser.add_argument("--temporal-json", type=Path, required=True)
    parser.add_argument(
        "--comparison-name",
        choices=("temporal_ego", "temporal_joint"),
        default="temporal_ego",
        help="Name of the method stored in --temporal-json.",
    )
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--continuity-threshold-m", type=float, default=1.50)
    parser.add_argument("--camera-height", type=float, default=1.65)
    parser.add_argument("--pitch-deg", type=float, default=0.0)
    parser.add_argument("--maximum-overlay-count", type=int, default=12)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def image_camera_poses(poses_path: Path, calib_path: Path) -> list[np.ndarray]:
    rows = np.loadtxt(poses_path, dtype=np.float64).reshape(-1, 3, 4)
    calibration = load_calibration(calib_path)
    poses = []
    for row in rows:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :] = row
        poses.append(pose @ calibration["cam0_from_image"])
    return poses


def valid_lanes(record: dict[str, object]) -> list[np.ndarray] | None:
    if not bool(record.get("eligible_for_two_curve_fit", False)):
        return None
    raw = record.get("selected_lanes_local_ground_xz_m", [])
    if not isinstance(raw, list) or len(raw) != 2:
        return None
    lanes = [np.asarray(item, dtype=np.float64).reshape(-1, 2) for item in raw]
    if min(map(len, lanes), default=0) < 2:
        return None
    return lanes


def longest_consecutive_run(frame_ids: list[int]) -> int:
    if not frame_ids:
        return 0
    best = current = 1
    for previous, current_id in zip(frame_ids, frame_ids[1:]):
        current = current + 1 if current_id == previous + 1 else 1
        best = max(best, current)
    return best


def continuity_rows(
    method: str,
    records: dict[int, dict[str, object]],
    poses: list[np.ndarray],
    camera_height: float,
    pitch_deg: float,
    threshold: float,
) -> list[dict[str, object]]:
    output = []
    valid_ids = sorted(frame_id for frame_id, row in records.items() if valid_lanes(row))
    for previous_id, frame_id in zip(valid_ids, valid_ids[1:]):
        if frame_id != previous_id + 1:
            continue
        previous = valid_lanes(records[previous_id])
        current = valid_lanes(records[frame_id])
        assert previous is not None and current is not None
        costs = []
        for side_index in range(2):
            predicted = transform_lane_points_by_pose(
                previous[side_index],
                poses[previous_id],
                poses[frame_id],
                camera_height=camera_height,
                pitch_deg=pitch_deg,
            )
            costs.append(symmetric_curve_distance_m(predicted, current[side_index]))
        output.append(
            {
                "method": method,
                "previous_frame": previous_id,
                "frame_id": frame_id,
                "left_continuity_m": costs[0],
                "right_continuity_m": costs[1],
                "mean_continuity_m": float(np.mean(costs)),
                "maximum_continuity_m": float(np.max(costs)),
                "continuity_gate_failed": bool(np.max(costs) > threshold),
            }
        )
    return output


def summarize_mode(
    method: str,
    records: dict[int, dict[str, object]],
    continuity: list[dict[str, object]],
) -> dict[str, object]:
    ids = sorted(records)
    valid_ids = [frame_id for frame_id in ids if valid_lanes(records[frame_id])]
    costs = np.asarray(
        [row["mean_continuity_m"] for row in continuity], dtype=np.float64
    )
    maximum = np.asarray(
        [row["maximum_continuity_m"] for row in continuity], dtype=np.float64
    )
    return {
        "method": method,
        "requested_frames": len(ids),
        "valid_two_lane_frames": len(valid_ids),
        "valid_two_lane_rate": len(valid_ids) / len(ids) if ids else 0.0,
        "longest_consecutive_valid_run": longest_consecutive_run(valid_ids),
        "frames_with_more_than_two_candidates": sum(
            int(records[item].get("candidate_count", 0)) > 2 for item in ids
        ),
        "adjacent_valid_transitions": len(continuity),
        "continuity_mean_m": float(np.mean(costs)) if len(costs) else None,
        "continuity_median_m": float(np.median(costs)) if len(costs) else None,
        "continuity_p90_m": float(np.percentile(costs, 90)) if len(costs) else None,
        "continuity_max_m": float(np.max(maximum)) if len(maximum) else None,
        "continuity_gate_failure_count": sum(
            bool(row["continuity_gate_failed"]) for row in continuity
        ),
        "temporal_gap_reset_count": sum(
            bool(records[item].get("temporal_gap_reset", False)) for item in ids
        ),
    }


def disagreement_rows(
    ego: dict[int, dict[str, object]], temporal: dict[int, dict[str, object]], threshold: float
) -> list[dict[str, object]]:
    output = []
    for frame_id in sorted(set(ego) & set(temporal)):
        first = valid_lanes(ego[frame_id])
        second = valid_lanes(temporal[frame_id])
        row: dict[str, object] = {
            "frame_id": frame_id,
            "ego_valid": first is not None,
            "temporal_valid": second is not None,
            "comparison_valid": second is not None,
            "both_valid": first is not None and second is not None,
        }
        if first is not None and second is not None:
            costs = [
                symmetric_curve_distance_m(first[index], second[index])
                for index in range(2)
            ]
            row.update(
                {
                    "left_method_disagreement_m": costs[0],
                    "right_method_disagreement_m": costs[1],
                    "maximum_method_disagreement_m": max(costs),
                    "methods_disagree_above_gate": max(costs) > threshold,
                }
            )
        else:
            row.update(
                {
                    "left_method_disagreement_m": None,
                    "right_method_disagreement_m": None,
                    "maximum_method_disagreement_m": None,
                    "methods_disagree_above_gate": first is not second,
                }
            )
        output.append(row)
    return output


def read_image(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"Cannot encode {path}")
    encoded.tofile(path)


def comparison_overlays(
    ego_json: Path,
    temporal_json: Path,
    comparison_name: str,
    rows: list[dict[str, object]],
    output_dir: Path,
    maximum_count: int,
) -> list[int]:
    ranked = sorted(
        rows,
        key=lambda row: (
            bool(row["methods_disagree_above_gate"]),
            float(row["maximum_method_disagreement_m"] or -1.0),
        ),
        reverse=True,
    )[:maximum_count]
    exported = []
    for row in ranked:
        frame_id = int(row["frame_id"])
        ego_image = read_image(
            ego_json.parent / "selected_pairs" / f"frame_{frame_id:06d}.png"
        )
        temporal_image = read_image(
            temporal_json.parent / "selected_pairs" / f"frame_{frame_id:06d}.png"
        )
        if ego_image is None or temporal_image is None:
            continue
        height = min(ego_image.shape[0], temporal_image.shape[0])
        ego_image = ego_image[:height]
        temporal_image = temporal_image[:height]
        canvas = np.hstack([ego_image, temporal_image])
        cv2.putText(canvas, "ego_adjacent", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(canvas, "ego_adjacent", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        offset = ego_image.shape[1] + 15
        cv2.putText(canvas, comparison_name, (offset, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(canvas, comparison_name, (offset, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        write_image(output_dir / f"frame_{frame_id:06d}.png", canvas)
        exported.append(frame_id)
    return exported


def plot_continuity(path: Path, rows: list[dict[str, object]], threshold: float) -> None:
    figure, axis = plt.subplots(figsize=(10.5, 4.6))
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    palette = ("#555555", "#1f77b4", "#2ca02c")
    for method, color in zip(methods, palette):
        selected = [row for row in rows if row["method"] == method]
        axis.plot(
            [row["frame_id"] for row in selected],
            [row["mean_continuity_m"] for row in selected],
            marker=".", markersize=3, linewidth=1.0, color=color, label=method,
        )
    axis.axhline(threshold, color="#d62728", linestyle="--", linewidth=1.0, label="continuity gate")
    axis.set_xlabel("frame")
    axis.set_ylabel("pose-aligned consecutive-frame distance [m]")
    axis.grid(True, linewidth=0.4, alpha=0.35)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.continuity_threshold_m <= 0 or args.maximum_overlay_count < 0:
        raise ValueError("Threshold must be positive and overlay count non-negative.")
    require_empty_output(args.output_dir)
    ego_doc = load_json(args.ego_json)
    temporal_doc = load_json(args.temporal_json)
    ego = {int(row["frame_id"]): row for row in ego_doc["frames"]}
    temporal = {int(row["frame_id"]): row for row in temporal_doc["frames"]}
    if set(ego) != set(temporal):
        raise ValueError("The two modes must cover exactly the same requested frame IDs.")
    poses = image_camera_poses(args.poses, args.calib)
    if max(ego) >= len(poses):
        raise ValueError("Pose file does not cover the compared frames.")

    all_continuity = []
    summaries = []
    for method, records in (("ego_adjacent", ego), (args.comparison_name, temporal)):
        rows = continuity_rows(
            method, records, poses, args.camera_height, args.pitch_deg,
            args.continuity_threshold_m,
        )
        all_continuity.extend(rows)
        summaries.append(summarize_mode(method, records, rows))
    disagreements = disagreement_rows(ego, temporal, args.continuity_threshold_m)
    exported = comparison_overlays(
        args.ego_json, args.temporal_json, args.comparison_name, disagreements,
        args.output_dir / "03_method_overlays", args.maximum_overlay_count,
    )
    write_csv(args.output_dir / "01_metrics" / "mode_summary.csv", summaries)
    write_csv(args.output_dir / "01_metrics" / "continuity_by_frame.csv", all_continuity)
    write_csv(args.output_dir / "01_metrics" / "method_disagreement_by_frame.csv", disagreements)
    plot_continuity(
        args.output_dir / "02_figures" / "continuity_comparison.png",
        all_continuity,
        args.continuity_threshold_m,
    )

    ego_summary, temporal_summary = summaries
    result = {
        "status": "complete",
        "claim_scope": (
            "pose-aligned temporal continuity and valid-frame coverage; not semantic "
            "lane identity accuracy and not official ground truth"
        ),
        "frame_range": [min(ego), max(ego)],
        "mode_summaries": summaries,
        "comparison": {
            "comparison_name": args.comparison_name,
            "comparison_minus_ego_valid_rate": (
                temporal_summary["valid_two_lane_rate"] - ego_summary["valid_two_lane_rate"]
            ),
            "comparison_minus_ego_continuity_p90_m": (
                None if temporal_summary["continuity_p90_m"] is None or ego_summary["continuity_p90_m"] is None
                else temporal_summary["continuity_p90_m"] - ego_summary["continuity_p90_m"]
            ),
            "temporal_minus_ego_valid_rate": (
                temporal_summary["valid_two_lane_rate"] - ego_summary["valid_two_lane_rate"]
            ),
            "temporal_minus_ego_continuity_p90_m": (
                None if temporal_summary["continuity_p90_m"] is None or ego_summary["continuity_p90_m"] is None
                else temporal_summary["continuity_p90_m"] - ego_summary["continuity_p90_m"]
            ),
            "frames_where_methods_disagree_above_gate": sum(
                bool(row["methods_disagree_above_gate"]) for row in disagreements
            ),
            "exported_overlay_frames": exported,
        },
        "interpretation_rule": (
            f"Prefer {args.comparison_name} only when it reduces pose-aligned continuity error "
            "without an unacceptable valid-frame loss; inspect exported disagreements."
        ),
        "warnings": [
            "CLRNet candidate indices are frame-local and are not persistent lane IDs.",
            "Project ego_left/ego_right IDs are created by pose-temporal association.",
            "A continuous sidewalk edge can also pass a temporal continuity test.",
            "Semantic validation is a separate stage.",
        ],
        "previous_outputs_modified": False,
    }
    (args.output_dir / "RESULT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "README.md").write_text(
        "# Lane identity mode comparison\n\n"
        "This output compares frame-local `ego_adjacent` with project-created "
        f"pose-temporal `{args.comparison_name}`. It measures continuity, not semantic truth.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "output_dir": str(args.output_dir.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
