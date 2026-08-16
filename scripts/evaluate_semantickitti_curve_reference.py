"""Evaluate fitted lane curves against sparse SemanticKITTI raw class-60 points.

SemanticKITTI class 60 is an official sparse lane-marking semantic label, not a
left/right lane polyline.  This script first associates each class-60 point to
one fixed left/right anchor set built from the temporal CLRNet selections.
Every fitted method is then evaluated against that same fixed reference set.
The association is therefore method-independent, but still detection-assisted;
the report never calls it official left/right ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from surf_bev.geometry import transform_lane_points_by_pose  # noqa: E402


SIDES = ("left", "right")
THRESHOLDS_M = (0.3, 0.5, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-lane-points", type=Path, required=True)
    parser.add_argument("--velodyne-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--reference-id", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        help="name,left_curve_csv,right_curve_csv; repeat for each fitted method",
    )
    parser.add_argument(
        "--method-result",
        action="append",
        default=[],
        help="name,RESULT.json; optional held-out consistency source",
    )
    parser.add_argument("--lane-semantic-id", type=int, default=60)
    parser.add_argument("--x-range", default="-20,20")
    parser.add_argument("--z-range", default="3,50")
    parser.add_argument("--association-gate-m", type=float, default=1.50)
    parser.add_argument("--association-margin-m", type=float, default=0.15)
    parser.add_argument("--curve-support-radius-m", type=float, default=3.0)
    parser.add_argument("--minimum-reference-points-per-side", type=int, default=20)
    parser.add_argument("--camera-height", type=float, default=1.65)
    parser.add_argument("--pitch-deg", type=float, default=0.0)
    return parser.parse_args()


def parse_range(value: str, name: str) -> tuple[float, float]:
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 2 or parts[0] >= parts[1]:
        raise ValueError(f"{name} requires two increasing values")
    return parts[0], parts[1]


def parse_methods(values: list[str]) -> dict[str, dict[str, Path]]:
    output: dict[str, dict[str, Path]] = {}
    for value in values:
        parts = [item.strip() for item in value.split(",")]
        if len(parts) != 3 or not parts[0]:
            raise ValueError("--method syntax: name,left_curve_csv,right_curve_csv")
        name = parts[0]
        if name in output:
            raise ValueError(f"Duplicate method name: {name}")
        output[name] = {"left": Path(parts[1]), "right": Path(parts[2])}
    return output


def parse_method_results(values: list[str]) -> dict[str, Path]:
    output = {}
    for value in values:
        name, separator, raw_path = value.partition(",")
        if not separator or not name.strip() or not raw_path.strip():
            raise ValueError("--method-result syntax: name,RESULT.json")
        output[name.strip()] = Path(raw_path.strip())
    return output


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


def read_calibration(path: Path) -> dict[str, np.ndarray]:
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, values = line.split(":", 1)
            entries[key.strip()] = np.fromstring(values, sep=" ", dtype=np.float64)
    if "P2" not in entries or entries["P2"].size != 12:
        raise ValueError("Calibration has no valid P2")
    tr_key = next((key for key in ("Tr", "Tr_velo_to_cam", "Tr_velo_cam") if key in entries), None)
    if tr_key is None or entries[tr_key].size != 12:
        raise ValueError("Calibration has no valid Velodyne-to-camera Tr")
    projection = entries["P2"].reshape(3, 4)
    intrinsic = projection[:, :3]
    image_from_cam0 = np.eye(4, dtype=np.float64)
    image_from_cam0[:3, 3] = np.linalg.solve(intrinsic, projection[:, 3])
    cam0_from_velodyne = np.eye(4, dtype=np.float64)
    cam0_from_velodyne[:3, :] = entries[tr_key].reshape(3, 4)
    return {
        "image_from_cam0": image_from_cam0,
        "cam0_from_image": np.linalg.inv(image_from_cam0),
        "image_from_velodyne": image_from_cam0 @ cam0_from_velodyne,
        "tr_key": tr_key,
        "P2": projection,
    }


def load_image_poses(path: Path, calibration: dict[str, np.ndarray]) -> list[np.ndarray]:
    rows = np.loadtxt(path, dtype=np.float64).reshape(-1, 3, 4)
    output = []
    for row in rows:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :] = row
        output.append(pose @ calibration["cam0_from_image"])
    return output


def load_curve(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = list(csv.DictReader(path.open("r", newline="", encoding="utf-8-sig")))
    if not rows:
        raise ValueError(f"Curve CSV is empty: {path}")
    points = np.asarray(
        [
            [float(row["x_right_reference_m"]), float(row["z_forward_reference_m"])]
            for row in rows
        ],
        dtype=np.float64,
    )
    return points[np.isfinite(points).all(axis=1)]


def align_selected_anchors(
    document: dict[str, object],
    poses: list[np.ndarray],
    reference_id: int,
    frame_start: int,
    frame_end: int,
    camera_height: float,
    pitch_deg: float,
) -> dict[str, np.ndarray]:
    reference_pose = poses[reference_id]
    buckets: dict[str, list[np.ndarray]] = {side: [] for side in SIDES}
    for row in document["frames"]:
        frame_id = int(row["frame_id"])
        if not frame_start <= frame_id <= frame_end:
            continue
        if not bool(row.get("eligible_for_two_curve_fit", False)):
            continue
        lanes = row.get("selected_lanes_local_ground_xz_m", [])
        if len(lanes) != 2:
            continue
        for side_index, side in enumerate(SIDES):
            aligned = transform_lane_points_by_pose(
                np.asarray(lanes[side_index], dtype=np.float64),
                poses[frame_id],
                reference_pose,
                camera_height=camera_height,
                pitch_deg=pitch_deg,
            )
            aligned = aligned[np.isfinite(aligned).all(axis=1)]
            if len(aligned):
                buckets[side].append(aligned)
    result = {
        side: np.vstack(buckets[side]) if buckets[side] else np.empty((0, 2))
        for side in SIDES
    }
    if min(map(len, result.values()), default=0) < 2:
        raise ValueError("Temporal selected points do not provide two usable anchor sets.")
    return result


def semantic_reference_points(
    velodyne_dir: Path,
    label_dir: Path,
    poses: list[np.ndarray],
    calibration: dict[str, np.ndarray],
    reference_id: int,
    frame_start: int,
    frame_end: int,
    semantic_id: int,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    reference_from_world = np.linalg.inv(poses[reference_id])
    points = []
    audit = []
    for frame_id in range(frame_start, frame_end + 1):
        scan_path = velodyne_dir / f"{frame_id:06d}.bin"
        label_path = label_dir / f"{frame_id:06d}.label"
        if not scan_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Missing frame {frame_id}: {scan_path} or {label_path}")
        scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
        labels = np.fromfile(label_path, dtype=np.uint32)
        if len(scan) != len(labels):
            raise ValueError(f"Point/label length mismatch at frame {frame_id}")
        mask = (labels & np.uint32(0xFFFF)) == semantic_id
        selected = scan[mask, :3].astype(np.float64)
        if len(selected):
            homogeneous = np.column_stack([selected, np.ones(len(selected))])
            local_image = (calibration["image_from_velodyne"] @ homogeneous.T).T
            local_keep = (
                np.isfinite(local_image[:, :3]).all(axis=1)
                & (local_image[:, 0] >= x_range[0])
                & (local_image[:, 0] <= x_range[1])
                & (local_image[:, 2] >= z_range[0])
                & (local_image[:, 2] <= z_range[1])
            )
            local_image = local_image[local_keep]
            aligned = (
                reference_from_world @ poses[frame_id] @ local_image.T
            ).T if len(local_image) else np.empty((0, 4))
            aligned_xz = aligned[:, [0, 2]]
            if len(aligned_xz):
                points.append(aligned_xz)
        else:
            local_keep = np.zeros(0, dtype=bool)
            aligned_xz = np.empty((0, 2))
        audit.append(
            {
                "frame_id": frame_id,
                "velodyne_points": len(scan),
                "raw_class_60_points": int(np.count_nonzero(mask)),
                "class_60_points_in_local_roi": int(np.count_nonzero(local_keep)),
                "class_60_points_exported_to_reference": len(aligned_xz),
            }
        )
    return (np.vstack(points) if points else np.empty((0, 2))), audit


def associate_reference(
    reference: np.ndarray,
    anchors: dict[str, np.ndarray],
    gate: float,
    margin: float,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]], dict[str, object]]:
    trees = {side: cKDTree(anchors[side]) for side in SIDES}
    distances = np.column_stack([trees[side].query(reference)[0] for side in SIDES])
    best = np.argmin(distances, axis=1)
    ordered = np.sort(distances, axis=1)
    accepted = (ordered[:, 0] <= gate) & ((ordered[:, 1] - ordered[:, 0]) >= margin)
    output = {
        side: reference[accepted & (best == side_index)]
        for side_index, side in enumerate(SIDES)
    }
    rows = []
    for index, point in enumerate(reference):
        side = SIDES[int(best[index])] if accepted[index] else "unassigned"
        rows.append(
            {
                "x_right_reference_m": point[0],
                "z_forward_reference_m": point[1],
                "assigned_side": side,
                "left_anchor_distance_m": distances[index, 0],
                "right_anchor_distance_m": distances[index, 1],
                "nearest_anchor_distance_m": ordered[index, 0],
                "anchor_distance_margin_m": ordered[index, 1] - ordered[index, 0],
                "accepted": bool(accepted[index]),
            }
        )
    summary = {
        "raw_reference_points_in_roi": len(reference),
        "assigned_left": len(output["left"]),
        "assigned_right": len(output["right"]),
        "unassigned": int(np.count_nonzero(~accepted)),
        "association_gate_m": gate,
        "association_margin_m": margin,
    }
    return output, rows, summary


def stats(values: np.ndarray, prefix: str) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {f"{prefix}_{name}": None for name in ("mean_m", "rmse_m", "median_m", "p90_m", "max_m")}
    return {
        f"{prefix}_mean_m": float(np.mean(values)),
        f"{prefix}_rmse_m": float(np.sqrt(np.mean(values ** 2))),
        f"{prefix}_median_m": float(np.median(values)),
        f"{prefix}_p90_m": float(np.percentile(values, 90)),
        f"{prefix}_max_m": float(np.max(values)),
    }


def evaluate_method(
    name: str,
    curves: dict[str, np.ndarray],
    references: dict[str, np.ndarray],
    support_radius: float,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    rows = []
    distances_out: dict[str, np.ndarray] = {}
    for side in SIDES:
        prediction = curves[side]
        reference = references[side]
        if not len(reference) or not len(prediction):
            ref_to_prediction = np.empty(0)
            prediction_to_ref = np.empty(0)
            support = np.zeros(len(prediction), dtype=bool)
        else:
            ref_to_prediction = cKDTree(prediction).query(reference)[0]
            all_prediction_to_ref = cKDTree(reference).query(prediction)[0]
            support = all_prediction_to_ref <= support_radius
            prediction_to_ref = all_prediction_to_ref[support]
        combined = np.concatenate([ref_to_prediction, prediction_to_ref])
        row: dict[str, object] = {
            "method": name,
            "side": side,
            "reference_points": len(reference),
            "prediction_points": len(prediction),
            "supported_prediction_points": int(np.count_nonzero(support)),
            "curve_support_radius_m": support_radius,
            **stats(ref_to_prediction, "reference_to_prediction"),
            **stats(prediction_to_ref, "supported_prediction_to_reference"),
            **stats(combined, "symmetric_supported"),
        }
        for threshold in THRESHOLDS_M:
            suffix = str(threshold).replace(".", "p") + "m"
            recall = float(np.mean(ref_to_prediction <= threshold)) if len(ref_to_prediction) else None
            precision = float(np.mean(prediction_to_ref <= threshold)) if len(prediction_to_ref) else None
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision is not None and recall is not None and precision + recall > 0
                else None
            )
            row[f"reference_coverage_at_{suffix}"] = recall
            row[f"supported_curve_precision_at_{suffix}"] = precision
            row[f"symmetric_f1_at_{suffix}"] = f1
        rows.append(row)
        distances_out[side] = ref_to_prediction
    return rows, distances_out


def heldout_rows(paths: dict[str, Path]) -> list[dict[str, object]]:
    output = []
    for method, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        result = json.loads(path.read_text(encoding="utf-8-sig"))
        candidates = result.get("models") or result.get("cross_validation") or []
        numeric = [
            row for row in candidates
            if isinstance(row, dict) and row.get("mean_fold_rmse_m") is not None
        ]
        selected = min(numeric, key=lambda row: float(row["mean_fold_rmse_m"])) if numeric else {}
        output.append(
            {
                "method": method,
                "valid_frame_count": result.get("valid_frame_count"),
                "selected_parameter": result.get("selected_degree", result.get("selected_smoothing_per_point_m2")),
                "best_heldout_mean_fold_rmse_m": selected.get("mean_fold_rmse_m"),
                "best_heldout_fold_standard_error_m": selected.get("fold_rmse_standard_error_m"),
                "interpretation": "cross-frame consistency against held-out CLRNet/IPM points; not absolute accuracy",
                "source": str(path.resolve()),
            }
        )
    return output


def plot_overlay(
    path: Path,
    anchors: dict[str, np.ndarray],
    references: dict[str, np.ndarray],
    methods: dict[str, dict[str, np.ndarray]],
) -> None:
    colors = {"left": "#1f77b4", "right": "#d62728"}
    styles = ["-", "--", ":", "-."]
    figure, axis = plt.subplots(figsize=(8.0, 9.5))
    for side in SIDES:
        anchor = anchors[side]
        reference = references[side]
        if len(anchor):
            axis.scatter(anchor[:, 0], anchor[:, 1], s=1, alpha=0.06, color=colors[side])
        if len(reference):
            axis.scatter(reference[:, 0], reference[:, 1], s=10, marker="x", alpha=0.75, color=colors[side], label=f"class-60 {side} reference")
    for method_index, (method, curves) in enumerate(methods.items()):
        for side in SIDES:
            curve = curves[side]
            axis.plot(curve[:, 0], curve[:, 1], styles[method_index % len(styles)], linewidth=1.8, color=colors[side], label=f"{method} {side}")
    axis.set_xlabel("X right in selected reference frame [m]")
    axis.set_ylabel("Z forward in selected reference frame [m]")
    axis.grid(True, linewidth=0.4, alpha=0.35)
    axis.legend(fontsize=8, ncol=2)
    axis.set_aspect("equal", adjustable="datalim")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    x_range = parse_range(args.x_range, "--x-range")
    z_range = parse_range(args.z_range, "--z-range")
    methods_paths = parse_methods(args.method)
    method_result_paths = parse_method_results(args.method_result)
    if args.frame_start < 0 or args.frame_end < args.frame_start:
        raise ValueError("Invalid frame range")
    reference_id = args.reference_id if args.reference_id is not None else args.frame_end
    if not args.frame_start <= reference_id <= args.frame_end:
        raise ValueError("reference-id must lie inside the evaluated frame range")
    for value in (args.association_gate_m, args.association_margin_m, args.curve_support_radius_m):
        if value <= 0:
            raise ValueError("Association and support parameters must be positive")
    require_empty_output(args.output_dir)

    required = [args.selected_lane_points, args.velodyne_dir, args.label_dir, args.poses, args.calib]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs: " + ", ".join(missing))
    calibration = read_calibration(args.calib)
    poses = load_image_poses(args.poses, calibration)
    if args.frame_end >= len(poses):
        raise ValueError("Pose file does not cover the evaluated frame range")
    selected_document = json.loads(args.selected_lane_points.read_text(encoding="utf-8-sig"))
    anchors = align_selected_anchors(
        selected_document, poses, reference_id, args.frame_start, args.frame_end,
        args.camera_height, args.pitch_deg,
    )
    raw_reference, frame_audit = semantic_reference_points(
        args.velodyne_dir, args.label_dir, poses, calibration, reference_id,
        args.frame_start, args.frame_end, args.lane_semantic_id, x_range, z_range,
    )
    if not len(raw_reference):
        references = {side: np.empty((0, 2)) for side in SIDES}
        association_rows = []
        association_summary = {
            "raw_reference_points_in_roi": 0,
            "assigned_left": 0,
            "assigned_right": 0,
            "unassigned": 0,
            "association_gate_m": args.association_gate_m,
            "association_margin_m": args.association_margin_m,
        }
    else:
        references, association_rows, association_summary = associate_reference(
            raw_reference, anchors, args.association_gate_m, args.association_margin_m
        )

    methods = {
        name: {side: load_curve(paths[side]) for side in SIDES}
        for name, paths in methods_paths.items()
    }
    metric_rows = []
    method_distances = {}
    for name, curves in methods.items():
        rows, distances = evaluate_method(
            name, curves, references, args.curve_support_radius_m
        )
        metric_rows.extend(rows)
        method_distances[name] = distances
    method_summary = []
    for name in methods:
        values = np.concatenate(
            [method_distances[name][side] for side in SIDES if len(method_distances[name][side])]
        ) if any(len(method_distances[name][side]) for side in SIDES) else np.empty(0)
        method_summary.append(
            {
                "method": name,
                "assigned_reference_points": len(references["left"]) + len(references["right"]),
                **stats(values, "reference_to_prediction"),
                **{
                    f"reference_coverage_at_{str(threshold).replace('.', 'p')}m": (
                        float(np.mean(values <= threshold)) if len(values) else None
                    )
                    for threshold in THRESHOLDS_M
                },
            }
        )
    heldout = heldout_rows(method_result_paths)
    enough = all(
        len(references[side]) >= args.minimum_reference_points_per_side for side in SIDES
    )
    status = "complete" if enough else "complete_with_insufficient_semantic_coverage"

    write_csv(args.output_dir / "00_audit" / "semantic_frame_coverage.csv", frame_audit)
    write_csv(args.output_dir / "00_audit" / "reference_association.csv", association_rows)
    write_csv(args.output_dir / "01_metrics" / "semantic_metrics_by_side.csv", metric_rows)
    write_csv(args.output_dir / "01_metrics" / "semantic_method_summary.csv", method_summary)
    write_csv(args.output_dir / "01_metrics" / "heldout_consistency_fallback.csv", heldout)
    plot_overlay(
        args.output_dir / "02_figures" / "semantic_reference_overlay.png",
        anchors, references, methods,
    )
    result = {
        "status": status,
        "dataset": f"KITTI Odometry frames {args.frame_start}-{args.frame_end} with SemanticKITTI labels",
        "reference_frame": reference_id,
        "coordinate_system": "metric X/Z in the selected P2 image-camera reference frame",
        "raw_semantic_id": args.lane_semantic_id,
        "reference_claim": (
            "official sparse SemanticKITTI lane-marking points, associated to project "
            "left/right anchors; not official continuous left/right lane polylines"
        ),
        "association": association_summary,
        "minimum_reference_points_per_side": args.minimum_reference_points_per_side,
        "semantic_coverage_gate_passed": enough,
        "method_summary": method_summary,
        "heldout_fallback": heldout,
        "primary_ranking_metric": (
            "reference_to_prediction_rmse_m when both sides pass the semantic coverage gate; "
            "otherwise use held-out consistency and label it non-absolute"
        ),
        "calibration": {
            "projection": "P2",
            "velodyne_to_camera_key": calibration["tr_key"],
        },
        "warnings": [
            "SemanticKITTI raw class 60 has no ego-left/ego-right instance ID.",
            "Left/right association is fixed once from temporal CLRNet anchors and reused by every fitted method.",
            "Reference-to-prediction distance is the primary sparse-reference accuracy diagnostic.",
            "Prediction-to-reference distance is support-limited because LiDAR lane-marking sampling is sparse.",
            "Held-out metrics measure consistency with CLRNet/IPM points, not official accuracy.",
        ],
        "previous_outputs_modified": False,
    }
    (args.output_dir / "RESULT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "README.md").write_text(
        "# SemanticKITTI sparse lane-marking evaluation\n\n"
        "Use `01_metrics/semantic_method_summary.csv` only when "
        "`semantic_coverage_gate_passed` is true. Otherwise use the held-out "
        "fallback and describe it as cross-frame consistency.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "output_dir": str(args.output_dir.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
