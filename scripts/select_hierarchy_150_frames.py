"""Choose a continuous span for ten fixed-size lane-fusion windows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from scan_clrnet_lane_counts import pose_window_stats  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-name", default="KITTI Odometry Sequence 00"
    )
    parser.add_argument("--scan-json", type=Path, action="append", required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--image-pattern", default="{frame_id:06d}.png")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=15)
    parser.add_argument("--block-count", type=int, default=10)
    parser.add_argument(
        "--block-stride",
        type=int,
        default=None,
        help="Frame step between block starts; defaults to non-overlapping block-size.",
    )
    parser.add_argument("--minimum-valid-frames-per-block", type=int, default=5)
    parser.add_argument(
        "--selected-rank",
        type=int,
        default=1,
        help="One-based candidate rank to materialize as selection.json.",
    )
    parser.add_argument(
        "--top-candidate-count",
        type=int,
        default=100,
        help="Maximum ranked candidates to export for later review or reruns.",
    )
    parser.add_argument(
        "--ranking-mode",
        choices=("coverage", "sustained_curve", "straight_curve_straight"),
        default="coverage",
    )
    parser.add_argument(
        "--minimum-trajectory-turn-deg-per-block",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--maximum-straight-turn-deg-per-block", type=float, default=1.5
    )
    parser.add_argument(
        "--minimum-curve-turn-deg-per-block", type=float, default=3.0
    )
    parser.add_argument("--minimum-middle-curve-blocks", type=int, default=2)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        if not rows:
            return
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def candidate_spans(
    rows: list[dict[str, object]],
    poses: np.ndarray,
    block_size: int,
    block_count: int,
    block_stride: int,
    minimum_valid: int,
    scan_json: Path,
    minimum_trajectory_turn_deg_per_block: float = 1.0,
    maximum_straight_turn_deg_per_block: float = 1.5,
    minimum_curve_turn_deg_per_block: float = 3.0,
    minimum_middle_curve_blocks: int = 2,
) -> list[dict[str, object]]:
    by_id = {int(row["frame_id"]): row for row in rows}
    ordered_ids = sorted(by_id)
    total_frames = block_size + (block_count - 1) * block_stride
    if not ordered_ids:
        return []
    output = []
    for start in range(ordered_ids[0], ordered_ids[-1] - total_frames + 2):
        frame_ids = list(range(start, start + total_frames))
        if any(frame_id not in by_id for frame_id in frame_ids):
            continue
        block_counts = []
        block_trajectory_turns = []
        for block_index in range(block_count):
            block_start = start + block_index * block_stride
            block_ids = list(range(block_start, block_start + block_size))
            block_counts.append(
                sum(
                    bool(by_id[frame_id]["eligible_for_two_curve_fit"])
                    for frame_id in block_ids
                )
            )
            block_stats = pose_window_stats(poses, block_ids)
            block_trajectory_turns.append(
                abs(float(block_stats["trajectory_net_heading_change_deg"]))
            )
        stats = pose_window_stats(poses, frame_ids)
        sustained_turn_block_count = sum(
            turn >= minimum_trajectory_turn_deg_per_block
            for turn in block_trajectory_turns
        )
        edge_block_count = max(1, min(2, block_count // 3))
        entry_turns = block_trajectory_turns[:edge_block_count]
        exit_turns = block_trajectory_turns[-edge_block_count:]
        middle_turns = block_trajectory_turns[
            edge_block_count : block_count - edge_block_count
        ]
        entry_mean_turn = float(np.mean(entry_turns))
        exit_mean_turn = float(np.mean(exit_turns))
        middle_curve_block_count = sum(
            turn >= minimum_curve_turn_deg_per_block for turn in middle_turns
        )
        middle_peak_turn = float(max(middle_turns, default=0.0))
        transition_gate = bool(
            entry_mean_turn <= maximum_straight_turn_deg_per_block
            and exit_mean_turn <= maximum_straight_turn_deg_per_block
            and middle_curve_block_count >= minimum_middle_curve_blocks
        )
        transition_score = float(
            sum(middle_turns) - sum(entry_turns) - sum(exit_turns)
        )
        output.append(
            {
                "start_frame": start,
                "end_frame": frame_ids[-1],
                "minimum_valid_frames_in_a_block": min(block_counts),
                "total_valid_frames": sum(block_counts),
                "all_blocks_meet_minimum": min(block_counts) >= minimum_valid,
                "block_valid_counts": block_counts,
                "block_trajectory_turn_deg": block_trajectory_turns,
                "sustained_turn_block_count": sustained_turn_block_count,
                "minimum_trajectory_turn_deg_per_block": (
                    minimum_trajectory_turn_deg_per_block
                ),
                "straight_curve_straight_gate": transition_gate,
                "entry_mean_trajectory_turn_deg": entry_mean_turn,
                "middle_peak_trajectory_turn_deg": middle_peak_turn,
                "middle_curve_block_count": middle_curve_block_count,
                "exit_mean_trajectory_turn_deg": exit_mean_turn,
                "maximum_straight_turn_deg_per_block": (
                    maximum_straight_turn_deg_per_block
                ),
                "minimum_curve_turn_deg_per_block": (
                    minimum_curve_turn_deg_per_block
                ),
                "minimum_middle_curve_blocks": minimum_middle_curve_blocks,
                "straight_curve_straight_score": transition_score,
                **stats,
                "scan_json": str(scan_json.resolve()),
                "selected_lane_points_json": str(
                    (scan_json.parent / "selected_lane_points.json").resolve()
                ),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    if args.block_size < 5 or args.block_count < 2:
        raise ValueError("block-size must be >=5 and block-count must be >=2.")
    block_stride = args.block_stride or args.block_size
    if not 1 <= block_stride <= args.block_size:
        raise ValueError("block-stride must be between 1 and block-size.")
    if not 1 <= args.minimum_valid_frames_per_block <= args.block_size:
        raise ValueError("minimum-valid-frames-per-block is outside the block size.")
    if args.selected_rank < 1 or args.top_candidate_count < 1:
        raise ValueError("selected-rank and top-candidate-count must be positive.")
    if args.selected_rank > args.top_candidate_count:
        raise ValueError("selected-rank cannot exceed top-candidate-count.")
    if args.minimum_trajectory_turn_deg_per_block <= 0:
        raise ValueError("minimum trajectory turn per block must be positive.")
    if (
        args.maximum_straight_turn_deg_per_block <= 0
        or args.minimum_curve_turn_deg_per_block <= 0
        or args.minimum_middle_curve_blocks < 1
    ):
        raise ValueError("Straight/curve transition thresholds must be positive.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory must be new or empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    required = [*args.scan_json, args.poses, args.image_dir]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs: " + ", ".join(missing))

    poses = np.loadtxt(args.poses, dtype=np.float64).reshape(-1, 3, 4)
    candidates: list[dict[str, object]] = []
    for scan_json in args.scan_json:
        document = json.loads(scan_json.read_text(encoding="utf-8"))
        candidates.extend(
            candidate_spans(
                document["counts"],
                poses,
                args.block_size,
                args.block_count,
                block_stride,
                args.minimum_valid_frames_per_block,
                scan_json,
                args.minimum_trajectory_turn_deg_per_block,
                args.maximum_straight_turn_deg_per_block,
                args.minimum_curve_turn_deg_per_block,
                args.minimum_middle_curve_blocks,
            )
        )
    if not candidates:
        raise ValueError("No complete fixed-window candidate exists in the scans.")

    if args.ranking_mode == "straight_curve_straight":
        ranking_key = lambda item: (  # noqa: E731
            bool(item["all_blocks_meet_minimum"]),
            bool(item["straight_curve_straight_gate"]),
            int(item["minimum_valid_frames_in_a_block"]),
            float(item["straight_curve_straight_score"]),
            int(item["total_valid_frames"]),
        )
    elif args.ranking_mode == "sustained_curve":
        ranking_key = lambda item: (  # noqa: E731
            bool(item["all_blocks_meet_minimum"]),
            int(item["sustained_turn_block_count"]),
            int(item["minimum_valid_frames_in_a_block"]),
            float(item["maximum_deviation_from_chord_m"]),
            int(item["total_valid_frames"]),
            float(item["path_displacement_ratio"]),
        )
    else:
        ranking_key = lambda item: (  # noqa: E731
            bool(item["all_blocks_meet_minimum"]),
            int(item["minimum_valid_frames_in_a_block"]),
            int(item["total_valid_frames"]),
            float(item["total_absolute_heading_change_deg"]),
        )
    ranked = sorted(candidates, key=ranking_key, reverse=True)
    if args.selected_rank > len(ranked):
        raise ValueError(
            f"selected-rank {args.selected_rank} exceeds {len(ranked)} candidates."
        )
    selected = ranked[args.selected_rank - 1]
    exported = ranked[: args.top_candidate_count]
    write_csv(
        args.output_dir / "candidate_options.csv",
        [
            {
                "rank": rank,
                "start_frame": item["start_frame"],
                "end_frame": item["end_frame"],
                "all_blocks_meet_minimum": item["all_blocks_meet_minimum"],
                "minimum_valid_frames_in_a_block": item[
                    "minimum_valid_frames_in_a_block"
                ],
                "total_valid_frame_uses": item["total_valid_frames"],
                "block_valid_counts": ";".join(
                    map(str, item["block_valid_counts"])
                ),
                "sustained_turn_block_count": item[
                    "sustained_turn_block_count"
                ],
                "straight_curve_straight_gate": item[
                    "straight_curve_straight_gate"
                ],
                "entry_mean_trajectory_turn_deg": item[
                    "entry_mean_trajectory_turn_deg"
                ],
                "middle_peak_trajectory_turn_deg": item[
                    "middle_peak_trajectory_turn_deg"
                ],
                "middle_curve_block_count": item["middle_curve_block_count"],
                "exit_mean_trajectory_turn_deg": item[
                    "exit_mean_trajectory_turn_deg"
                ],
                "straight_curve_straight_score": item[
                    "straight_curve_straight_score"
                ],
                "block_trajectory_turn_deg": ";".join(
                    f"{value:.6f}"
                    for value in item["block_trajectory_turn_deg"]
                ),
                "path_length_m": item["path_length_m"],
                "displacement_m": item["displacement_m"],
                "path_displacement_ratio": item["path_displacement_ratio"],
                "maximum_deviation_from_chord_m": item[
                    "maximum_deviation_from_chord_m"
                ],
                "trajectory_net_heading_change_deg": item[
                    "trajectory_net_heading_change_deg"
                ],
                "net_heading_change_deg": item["net_heading_change_deg"],
                "total_absolute_heading_change_deg": item[
                    "total_absolute_heading_change_deg"
                ],
                "scan_json": item["scan_json"],
                "selected_lane_points_json": item[
                    "selected_lane_points_json"
                ],
            }
            for rank, item in enumerate(exported, start=1)
        ],
    )
    start = int(selected["start_frame"])
    block_rows = []
    manual_frame_ids = []
    for block_index, valid_count in enumerate(selected["block_valid_counts"]):
        block_start = start + block_index * block_stride
        block_end = block_start + args.block_size - 1
        manual_ids = [block_start, block_start + args.block_size // 2, block_end]
        manual_frame_ids.extend(manual_ids)
        block_rows.append(
            {
                "block_index": block_index,
                "start_frame": block_start,
                "end_frame": block_end,
                "valid_frames": valid_count,
                "manual_annotation_frames": ";".join(map(str, manual_ids)),
            }
        )
    write_csv(args.output_dir / "block_coverage.csv", block_rows)

    image_output = args.output_dir / "manual_annotation_package" / "images"
    image_output.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for frame_id in manual_frame_ids:
        source = args.image_dir / args.image_pattern.format(frame_id=frame_id)
        if not source.is_file():
            raise FileNotFoundError(f"Missing manual-annotation image: {source}")
        target = image_output / f"frame_{frame_id:06d}.png"
        shutil.copy2(source, target)
        manifest_rows.append(
            {
                "frame_id": frame_id,
                "source": str(source.resolve()),
                "file": target.name,
                "sha256": sha256(target),
            }
        )
    write_csv(
        args.output_dir / "manual_annotation_package" / "manifest.csv",
        manifest_rows,
    )
    package_zip = args.output_dir / "manual_annotation_package.zip"
    with zipfile.ZipFile(package_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((args.output_dir / "manual_annotation_package").rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    path.relative_to(args.output_dir / "manual_annotation_package"),
                )

    report = {
        "status": (
            "selected"
            if bool(selected["all_blocks_meet_minimum"])
            else "selected_but_below_requested_coverage"
        ),
        "dataset": args.dataset_name,
        "claim_scope": (
            "frame-availability audit for ten fixed-size 15-frame windows; "
            "not a lane-identity or accuracy result"
        ),
        "block_size": args.block_size,
        "block_count": args.block_count,
        "block_stride": block_stride,
        "overlap_frames_between_adjacent_blocks": args.block_size - block_stride,
        "total_unique_frames": args.block_size + (args.block_count - 1) * block_stride,
        "minimum_valid_frames_per_block_required": args.minimum_valid_frames_per_block,
        "selected_rank": args.selected_rank,
        "available_candidate_count": len(ranked),
        "exported_candidate_count": len(exported),
        "candidate_options_csv": str(
            (args.output_dir / "candidate_options.csv").resolve()
        ),
        "ranking_mode": args.ranking_mode,
        "minimum_trajectory_turn_deg_per_block": (
            args.minimum_trajectory_turn_deg_per_block
        ),
        "maximum_straight_turn_deg_per_block": (
            args.maximum_straight_turn_deg_per_block
        ),
        "minimum_curve_turn_deg_per_block": (
            args.minimum_curve_turn_deg_per_block
        ),
        "minimum_middle_curve_blocks": args.minimum_middle_curve_blocks,
        "selected": selected,
        "blocks": block_rows,
        "manual_annotation_frame_ids": manual_frame_ids,
        "manual_annotation_policy": (
            "first, middle and last frame of every 15-frame block; label only "
            "visible ego-left/ego-right boundaries and mark ambiguity instead "
            "of interpolating through occlusion"
        ),
        "package_zip": str(package_zip.resolve()),
        "top_candidates": exported,
        "warnings": [
            "Five valid frames per block is a feasibility floor, not proof of accuracy.",
            "All selected candidate identities still require visual inspection.",
            "Manual annotations are pseudo-ground-truth, not official KITTI truth.",
        ],
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
