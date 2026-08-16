"""Audit why requested curved-road frames fail the two-lane metric gate."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan",
        action="append",
        required=True,
        help="METHOD,PATH_TO_SCAN_JSON; repeat for every selection mode.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_scan(specification: str) -> tuple[str, Path, list[dict[str, object]]]:
    parts = specification.split(",", 1)
    if len(parts) != 2 or not parts[0].strip():
        raise ValueError("--scan must be METHOD,PATH")
    method, raw_path = parts[0].strip(), parts[1].strip()
    path = Path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = document.get("counts", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No scan rows in {path}")
    return method, path, rows


def failure_category(row: dict[str, object]) -> str:
    if bool(row.get("metric_bev_point_gate", False)):
        return "valid_two_lane_metric_frame"
    if int(row.get("candidate_count", 0)) < 2:
        return "fewer_than_two_clrnet_candidates"
    if not bool(row.get("pair_selection_gate", False)):
        reason = str(row.get("pair_selection_reason", "unknown"))
        if "distance_gate_failed" in reason:
            return "temporal_distance_gate_failed"
        if "one_side" in reason or "one_side_of_camera_center" in reason:
            return "image_side_pairing_failed"
        return "other_pair_selection_failure"
    return "selected_pair_has_too_few_ipm_points"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaded = [load_scan(specification) for specification in args.scan]
    frame_sets = [{int(row["frame_id"]) for row in rows} for _, _, rows in loaded]
    if any(values != frame_sets[0] for values in frame_sets[1:]):
        raise ValueError("All scans must cover the same frame IDs.")

    summary_rows = []
    frame_rows = []
    valid_by_method: dict[str, set[int]] = {}
    for method, path, rows in loaded:
        categories = Counter(failure_category(row) for row in rows)
        reasons = Counter(str(row.get("pair_selection_reason")) for row in rows)
        valid = {
            int(row["frame_id"])
            for row in rows
            if bool(row.get("metric_bev_point_gate", False))
        }
        valid_by_method[method] = valid
        summary_rows.append(
            {
                "method": method,
                "requested_frames": len(rows),
                "valid_two_lane_metric_frames": len(valid),
                "fewer_than_two_clrnet_candidates": categories[
                    "fewer_than_two_clrnet_candidates"
                ],
                "image_side_pairing_failed": categories["image_side_pairing_failed"],
                "temporal_distance_gate_failed": categories[
                    "temporal_distance_gate_failed"
                ],
                "other_pair_selection_failure": categories[
                    "other_pair_selection_failure"
                ],
                "selected_pair_has_too_few_ipm_points": categories[
                    "selected_pair_has_too_few_ipm_points"
                ],
                "frames_with_more_than_two_candidates": sum(
                    int(row.get("candidate_count", 0)) > 2 for row in rows
                ),
                "scan_json": str(path.resolve()),
            }
        )
        for row in rows:
            frame_rows.append(
                {
                    "method": method,
                    "frame_id": int(row["frame_id"]),
                    "candidate_count": int(row.get("candidate_count", 0)),
                    "pair_selection_reason": row.get("pair_selection_reason"),
                    "pair_selection_gate": bool(row.get("pair_selection_gate", False)),
                    "metric_bev_point_gate": bool(
                        row.get("metric_bev_point_gate", False)
                    ),
                    "selected_projected_point_counts": row.get(
                        "selected_projected_point_counts"
                    ),
                    "failure_category": failure_category(row),
                }
            )

    baseline_name = "temporal_ego" if "temporal_ego" in valid_by_method else loaded[0][0]
    baseline = valid_by_method[baseline_name]
    comparisons = {}
    for method, valid in valid_by_method.items():
        if method == baseline_name:
            continue
        comparisons[method] = {
            "baseline": baseline_name,
            "newly_recovered_frames": sorted(valid - baseline),
            "newly_lost_frames": sorted(baseline - valid),
            "valid_frame_change": len(valid) - len(baseline),
        }

    write_csv(args.output_dir / "failure_summary.csv", summary_rows)
    write_csv(args.output_dir / "failure_by_frame.csv", frame_rows)
    result = {
        "status": "complete",
        "summary": summary_rows,
        "comparisons": comparisons,
        "interpretation": (
            "Only the largest failure category should drive the next coverage change; "
            "do not relax the temporal gate when CLRNet produced fewer than two candidates."
        ),
        "previous_outputs_modified": False,
    }
    (args.output_dir / "RESULT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
