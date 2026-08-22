"""Create auditable cross-Sequence plots from the final SURF batch CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def as_float(row: dict[str, str], name: str) -> float:
    value = row.get(name, "").strip()
    return float(value) if value else float("nan")


def as_int(row: dict[str, str], name: str) -> int:
    value = row.get(name, "").strip()
    return int(float(value)) if value else 0


def run(batch_summary: Path, output_dir: Path) -> dict[str, object]:
    with batch_summary.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    completed = [row for row in rows if row.get("status", "").startswith("complete")]
    if not completed:
        raise ValueError("The batch summary has no completed Sequence.")

    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [row["sequence_id"] for row in completed]
    x = np.arange(len(labels), dtype=float)
    left = np.array([as_float(row, "left_observation_fraction") for row in completed])
    right = np.array([as_float(row, "right_observation_fraction") for row in completed])
    both = np.array([as_float(row, "both_observation_fraction") for row in completed])
    fitting = np.array([as_float(row, "completed_fit_fraction") for row in completed])
    continuity = np.array([as_float(row, "continuity_pass_fraction") for row in completed])
    straight_poly = np.array([as_int(row, "straight_polynomial_fits") for row in completed])
    transition_spline = np.array([as_int(row, "transition_bspline_fits") for row in completed])
    curve_poly = np.array([as_int(row, "curve_polynomial_fits") for row in completed])
    curve_spline = np.array([as_int(row, "curve_bspline_fits") for row in completed])

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)
    width = 0.24
    axes[0].bar(x - width, left, width, label="left observation")
    axes[0].bar(x, right, width, label="right observation")
    axes[0].bar(x + width, both, width, label="both sides in one frame")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("frame fraction")
    axes[0].set_title("CLRNet/IPM observation coverage by KITTI Odometry Sequence")
    axes[0].legend(ncol=3, fontsize=9)

    axes[1].bar(x - width / 2, fitting, width, label="completed window-side fits")
    axes[1].bar(x + width / 2, continuity, width, label="passed interfaces")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("fraction")
    axes[1].set_title("Fitting coverage and continuity diagnostics")
    axes[1].legend(ncol=2, fontsize=9)

    axes[2].bar(x, straight_poly, label="straight polynomial")
    axes[2].bar(x, transition_spline, bottom=straight_poly, label="transition B-spline")
    base = straight_poly + transition_spline
    axes[2].bar(x, curve_poly, bottom=base, label="curve polynomial")
    axes[2].bar(x, curve_spline, bottom=base + curve_poly, label="curve B-spline")
    axes[2].set_ylabel("selected window-side fits")
    axes[2].set_title("Selected curve models")
    axes[2].legend(ncol=2, fontsize=9)

    for axis in axes:
        axis.set_xticks(x, labels)
        axis.set_xlabel("Sequence")
        axis.grid(axis="y", alpha=0.25)
    figure_path = output_dir / "ALL_SEQUENCES_SUMMARY.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    frames = np.array([as_int(row, "frame_count") for row in completed], dtype=float)
    total_frames = int(frames.sum())

    def weighted(values: np.ndarray) -> float | None:
        valid = np.isfinite(values) & (frames > 0)
        if not np.any(valid):
            return None
        return float(np.average(values[valid], weights=frames[valid]))

    aggregate: dict[str, object] = {
        "status": "complete",
        "completed_sequences": labels,
        "completed_sequence_count": len(completed),
        "failed_sequence_count": len(rows) - len(completed),
        "completed_sequence_frames": total_frames,
        "frame_weighted_observation_fraction": {
            "left": weighted(left),
            "right": weighted(right),
            "both": weighted(both),
        },
        "selected_model_totals": {
            "straight_polynomial": int(straight_poly.sum()),
            "transition_bspline": int(transition_spline.sum()),
            "curve_polynomial": int(curve_poly.sum()),
            "curve_bspline": int(curve_spline.sum()),
        },
        "accepted_low_confidence_bridges": sum(
            as_int(row, "accepted_low_confidence_bridges") for row in completed
        ),
        "scope": (
            "Coverage and consistency of CLRNet/IPM-derived observations; "
            "not official lane-position accuracy."
        ),
        "figure": str(figure_path),
    }
    aggregate_path = output_dir / "ALL_SEQUENCES_AGGREGATE.json"
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return aggregate


def main() -> None:
    args = parse_args()
    result = run(args.batch_summary, args.output_dir)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

