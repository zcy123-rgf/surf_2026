"""Choose a continuous 150-frame span for ten 15-frame fusion blocks."""

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
    parser.add_argument("--scan-json", type=Path, action="append", required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--image-pattern", default="{frame_id:06d}.png")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=15)
    parser.add_argument("--block-count", type=int, default=10)
    parser.add_argument("--minimum-valid-frames-per-block", type=int, default=5)
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
    minimum_valid: int,
    scan_json: Path,
) -> list[dict[str, object]]:
    by_id = {int(row["frame_id"]): row for row in rows}
    ordered_ids = sorted(by_id)
    total_frames = block_size * block_count
    if not ordered_ids:
        return []
    output = []
    for start in range(ordered_ids[0], ordered_ids[-1] - total_frames + 2):
        frame_ids = list(range(start, start + total_frames))
        if any(frame_id not in by_id for frame_id in frame_ids):
            continue
        block_counts = []
        for block_index in range(block_count):
            block_start = start + block_index * block_size
            block_ids = range(block_start, block_start + block_size)
            block_counts.append(
                sum(
                    bool(by_id[frame_id]["eligible_for_two_curve_fit"])
                    for frame_id in block_ids
                )
            )
        stats = pose_window_stats(poses, frame_ids)
        output.append(
            {
                "start_frame": start,
                "end_frame": frame_ids[-1],
                "minimum_valid_frames_in_a_block": min(block_counts),
                "total_valid_frames": sum(block_counts),
                "all_blocks_meet_minimum": min(block_counts) >= minimum_valid,
                "block_valid_counts": block_counts,
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
    if not 1 <= args.minimum_valid_frames_per_block <= args.block_size:
        raise ValueError("minimum-valid-frames-per-block is outside the block size.")
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
                args.minimum_valid_frames_per_block,
                scan_json,
            )
        )
    if not candidates:
        raise ValueError("No complete 150-frame candidate exists in the scans.")

    ranked = sorted(
        candidates,
        key=lambda item: (
            bool(item["all_blocks_meet_minimum"]),
            int(item["minimum_valid_frames_in_a_block"]),
            int(item["total_valid_frames"]),
            float(item["total_absolute_heading_change_deg"]),
        ),
        reverse=True,
    )
    selected = ranked[0]
    start = int(selected["start_frame"])
    block_rows = []
    manual_frame_ids = []
    for block_index, valid_count in enumerate(selected["block_valid_counts"]):
        block_start = start + block_index * args.block_size
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
        "claim_scope": (
            "frame-availability audit for ten fixed non-overlapping 15-frame "
            "blocks; not a lane-identity or accuracy result"
        ),
        "block_size": args.block_size,
        "block_count": args.block_count,
        "total_frames": args.block_size * args.block_count,
        "minimum_valid_frames_per_block_required": args.minimum_valid_frames_per_block,
        "selected": selected,
        "blocks": block_rows,
        "manual_annotation_frame_ids": manual_frame_ids,
        "manual_annotation_policy": (
            "first, middle and last frame of every 15-frame block; label only "
            "visible ego-left/ego-right boundaries and mark ambiguity instead "
            "of interpolating through occlusion"
        ),
        "package_zip": str(package_zip.resolve()),
        "top_candidates": ranked[:20],
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
