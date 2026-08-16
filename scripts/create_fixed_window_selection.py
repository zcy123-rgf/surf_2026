"""Create an auditable fixed-window selection document from a completed scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-json", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-valid-frames", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.end_frame < args.start_frame:
        raise ValueError("end-frame must be >= start-frame")
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    scan = json.loads(args.scan_json.read_text(encoding="utf-8-sig"))
    by_id = {int(row["frame_id"]): row for row in scan["counts"]}
    requested = list(range(args.start_frame, args.end_frame + 1))
    missing = [frame_id for frame_id in requested if frame_id not in by_id]
    if missing:
        raise ValueError(f"Scan is missing requested frames, first missing: {missing[:5]}")
    valid = [
        frame_id for frame_id in requested
        if bool(by_id[frame_id].get("eligible_for_two_curve_fit", False))
    ]
    status = "selected" if len(valid) >= args.minimum_valid_frames else "insufficient_valid_frames"
    document = {
        "status": status,
        "dataset": args.dataset_name,
        "selection_policy": "fixed window supplied by the reviewed qly curvature selection",
        "selected_rank": 1,
        "selected": {
            "start_frame": args.start_frame,
            "end_frame": args.end_frame,
            "requested_frame_count": len(requested),
            "valid_frame_count": len(valid),
            "valid_frame_ids": valid,
            "scan_json": str(args.scan_json.resolve()),
            "selected_lane_points_json": str(
                (args.scan_json.parent / "selected_lane_points.json").resolve()
            ),
        },
        "gate": {
            "minimum_valid_frames": args.minimum_valid_frames,
            "passed": status == "selected",
        },
        "previous_outputs_modified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if status != "selected":
        raise RuntimeError(
            f"Only {len(valid)} valid frames; minimum is {args.minimum_valid_frames}."
        )
    print(json.dumps({"status": status, "valid_frames": len(valid)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
