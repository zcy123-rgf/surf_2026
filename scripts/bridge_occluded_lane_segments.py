#!/usr/bin/env python3
"""Create auditable low-confidence hypotheses across short lane occlusions.

Observed fitted segments are never modified.  A bridge is considered only
between two sufficiently supported segments of the same project side.  Its
endpoint tangents are estimated from neighbourhoods, not from two isolated
points.  Accepted bridges are cubic Hermite curves and are exported with an
explicit ``hypothesis`` source and confidence; rejected gaps remain open.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SIDES = ("left", "right")
SIDE_COLOURS = {"left": "#2b6cb0", "right": "#d63230"}


@dataclass(frozen=True)
class Segment:
    side: str
    segment_id: int
    keys: np.ndarray
    points: np.ndarray
    supports: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bridge supported pre/post-occlusion lane segments in metric X/Z."
    )
    parser.add_argument("--blended-nodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-order-gap", type=float, default=30.0)
    parser.add_argument("--maximum-endpoint-gap-m", type=float, default=35.0)
    parser.add_argument("--maximum-tangent-angle-deg", type=float, default=45.0)
    parser.add_argument("--maximum-chord-tangent-angle-deg", type=float, default=60.0)
    parser.add_argument("--maximum-bridge-curvature-1pm", type=float, default=0.20)
    parser.add_argument("--minimum-segment-nodes", type=int, default=10)
    parser.add_argument("--tangent-fit-nodes", type=int, default=12)
    parser.add_argument("--bridge-samples", type=int, default=80)
    parser.add_argument("--lane-width-min-m", type=float, default=1.5)
    parser.add_argument("--lane-width-max-m", type=float, default=8.0)
    parser.add_argument("--lane-width-p90-deviation-m", type=float, default=1.5)
    return parser.parse_args()


def require_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_segments(path: Path) -> dict[str, list[Segment]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in read_rows(path):
        grouped.setdefault((row["side"], int(row["segment_id"])), []).append(row)
    result = {side: [] for side in SIDES}
    for (side, segment_id), rows in grouped.items():
        ordered = sorted(rows, key=lambda row: float(row["ordering_key"]))
        result.setdefault(side, []).append(
            Segment(
                side=side,
                segment_id=segment_id,
                keys=np.asarray([float(row["ordering_key"]) for row in ordered]),
                points=np.asarray(
                    [
                        [float(row["x_common_m"]), float(row["z_common_m"])]
                        for row in ordered
                    ],
                    dtype=np.float64,
                ),
                supports=np.asarray(
                    [int(row["contributing_sample_count"]) for row in ordered],
                    dtype=np.int64,
                ),
            )
        )
    for side in result:
        result[side].sort(key=lambda item: float(item.keys[0]))
    return result


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        raise ValueError("Cannot normalize a zero tangent.")
    return vector / norm


def vector_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    dot = float(np.clip(np.dot(unit(first), unit(second)), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def robust_endpoint_tangent(
    segment: Segment, at_end: bool, node_count: int
) -> np.ndarray:
    count = min(max(3, node_count), len(segment.keys))
    keys = segment.keys[-count:] if at_end else segment.keys[:count]
    points = segment.points[-count:] if at_end else segment.points[:count]
    centred = keys - float(np.mean(keys))
    denominator = float(np.dot(centred, centred))
    if denominator <= 1e-9:
        return unit(points[-1] - points[0])
    tangent = np.asarray(
        [
            float(np.dot(centred, points[:, axis] - np.mean(points[:, axis])))
            / denominator
            for axis in range(2)
        ]
    )
    return unit(tangent)


def hermite_bridge(
    start: np.ndarray,
    end: np.ndarray,
    start_tangent: np.ndarray,
    end_tangent: np.ndarray,
    sample_count: int,
) -> np.ndarray:
    chord_length = float(np.linalg.norm(end - start))
    scale = chord_length
    t = np.linspace(0.0, 1.0, sample_count)
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return (
        h00[:, None] * start
        + h10[:, None] * scale * unit(start_tangent)
        + h01[:, None] * end
        + h11[:, None] * scale * unit(end_tangent)
    )


def discrete_curvature(points: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return np.empty(0)
    first = np.gradient(points, axis=0)
    second = np.gradient(first, axis=0)
    numerator = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    denominator = np.maximum(np.linalg.norm(first, axis=1) ** 3, 1e-9)
    return numerator / denominator


def interpolate_other_side(
    segments: list[Segment], query_keys: np.ndarray
) -> np.ndarray | None:
    for segment in segments:
        if query_keys[0] >= segment.keys[0] and query_keys[-1] <= segment.keys[-1]:
            return np.column_stack(
                [
                    np.interp(query_keys, segment.keys, segment.points[:, axis])
                    for axis in range(2)
                ]
            )
    return None


def evaluate_gap(
    first: Segment,
    second: Segment,
    other_segments: list[Segment],
    args: argparse.Namespace,
) -> tuple[dict[str, object], np.ndarray | None, np.ndarray | None]:
    order_gap = float(second.keys[0] - first.keys[-1])
    endpoint_gap = float(np.linalg.norm(second.points[0] - first.points[-1]))
    reasons: list[str] = []
    if len(first.points) < args.minimum_segment_nodes or len(second.points) < args.minimum_segment_nodes:
        reasons.append("insufficient_segment_support")
    if order_gap <= 0:
        reasons.append("segments_overlap_or_reverse")
    if order_gap > args.maximum_order_gap:
        reasons.append("order_gap_exceeds_gate")
    if endpoint_gap > args.maximum_endpoint_gap_m:
        reasons.append("endpoint_gap_exceeds_gate")
    bridge: np.ndarray | None = None
    bridge_keys: np.ndarray | None = None
    tangent_angle = float("nan")
    chord_start_angle = float("nan")
    chord_end_angle = float("nan")
    maximum_curvature = float("nan")
    width_median = float("nan")
    width_p90_deviation = float("nan")
    if not reasons:
        start_tangent = robust_endpoint_tangent(first, True, args.tangent_fit_nodes)
        end_tangent = robust_endpoint_tangent(second, False, args.tangent_fit_nodes)
        chord = second.points[0] - first.points[-1]
        tangent_angle = vector_angle_deg(start_tangent, end_tangent)
        chord_start_angle = vector_angle_deg(start_tangent, chord)
        chord_end_angle = vector_angle_deg(end_tangent, chord)
        if tangent_angle > args.maximum_tangent_angle_deg:
            reasons.append("endpoint_tangent_mismatch")
        if max(chord_start_angle, chord_end_angle) > args.maximum_chord_tangent_angle_deg:
            reasons.append("tangent_chord_mismatch")
        if not reasons:
            bridge = hermite_bridge(
                first.points[-1],
                second.points[0],
                start_tangent,
                end_tangent,
                args.bridge_samples,
            )
            bridge_keys = np.linspace(first.keys[-1], second.keys[0], len(bridge))
            curvature = np.abs(discrete_curvature(bridge))
            maximum_curvature = float(np.max(curvature)) if len(curvature) else 0.0
            if maximum_curvature > args.maximum_bridge_curvature_1pm:
                reasons.append("bridge_curvature_exceeds_gate")
            other = interpolate_other_side(other_segments, bridge_keys)
            if other is not None:
                widths = np.linalg.norm(bridge - other, axis=1)
                width_median = float(np.median(widths))
                reference_width = float(np.median(np.r_[widths[:8], widths[-8:]]))
                width_p90_deviation = float(
                    np.percentile(np.abs(widths - reference_width), 90)
                )
                if not args.lane_width_min_m <= width_median <= args.lane_width_max_m:
                    reasons.append("lane_width_outside_gate")
                if width_p90_deviation > args.lane_width_p90_deviation_m:
                    reasons.append("lane_width_variation_exceeds_gate")
    accepted = not reasons and bridge is not None and bridge_keys is not None
    if not accepted:
        bridge = None
        bridge_keys = None
    confidence = 0.0
    if accepted:
        confidence = float(
            math.exp(-order_gap / max(args.maximum_order_gap, 1e-6))
            * math.exp(-tangent_angle / max(args.maximum_tangent_angle_deg, 1e-6))
        )
    diagnostic = {
        "side": first.side,
        "first_segment_id": first.segment_id,
        "second_segment_id": second.segment_id,
        "first_key_end": float(first.keys[-1]),
        "second_key_start": float(second.keys[0]),
        "order_gap": order_gap,
        "endpoint_gap_m": endpoint_gap,
        "endpoint_tangent_angle_deg": tangent_angle,
        "start_tangent_to_chord_deg": chord_start_angle,
        "end_tangent_to_chord_deg": chord_end_angle,
        "maximum_bridge_curvature_1pm": maximum_curvature,
        "lane_width_median_m": width_median,
        "lane_width_p90_deviation_m": width_p90_deviation,
        "accepted_as_low_confidence_hypothesis": accepted,
        "confidence": confidence,
        "rejection_reasons": ";".join(reasons),
    }
    return diagnostic, bridge_keys, bridge


def plot_result(
    path: Path,
    segments: dict[str, list[Segment]],
    accepted: list[tuple[str, np.ndarray, np.ndarray]],
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 8.0), dpi=180)
    for side in SIDES:
        for index, segment in enumerate(segments.get(side, [])):
            ax.plot(
                segment.points[:, 0],
                segment.points[:, 1],
                color=SIDE_COLOURS[side],
                linewidth=2.0,
                label=f"{side} observed" if index == 0 else None,
            )
    labelled = set()
    for side, _, points in accepted:
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=SIDE_COLOURS[side],
            linestyle="--",
            linewidth=2.0,
            alpha=0.75,
            label=f"{side} hypothesis bridge" if side not in labelled else None,
        )
        labelled.add(side)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("X right in common reference [m]")
    ax.set_ylabel("Z forward in common reference [m]")
    ax.set_title("Observed lane segments and auditable occlusion hypotheses")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, object]:
    require_empty_output(args.output_dir)
    segments = load_segments(args.blended_nodes)
    diagnostics: list[dict[str, object]] = []
    accepted: list[tuple[str, np.ndarray, np.ndarray]] = []
    bridge_rows: list[dict[str, object]] = []
    for side in SIDES:
        other_side = "right" if side == "left" else "left"
        for first, second in zip(segments.get(side, [])[:-1], segments.get(side, [])[1:]):
            diagnostic, keys, points = evaluate_gap(
                first, second, segments.get(other_side, []), args
            )
            bridge_id = len(diagnostics)
            diagnostic["bridge_id"] = bridge_id
            diagnostics.append(diagnostic)
            if keys is None or points is None:
                continue
            accepted.append((side, keys, points))
            for node_id, (key, point) in enumerate(zip(keys, points)):
                bridge_rows.append(
                    {
                        "side": side,
                        "bridge_id": bridge_id,
                        "node_id": node_id,
                        "ordering_key": float(key),
                        "x_common_m": float(point[0]),
                        "z_common_m": float(point[1]),
                        "source": "low_confidence_occlusion_hypothesis",
                        "confidence": diagnostic["confidence"],
                    }
                )
    combined_rows: list[dict[str, object]] = []
    for side in SIDES:
        for segment in segments.get(side, []):
            for node_id, (key, point, support) in enumerate(
                zip(segment.keys, segment.points, segment.supports)
            ):
                combined_rows.append(
                    {
                        "side": side,
                        "source_id": f"observed_segment_{segment.segment_id}",
                        "node_id": node_id,
                        "ordering_key": float(key),
                        "x_common_m": float(point[0]),
                        "z_common_m": float(point[1]),
                        "source": "observed_fitted_segment",
                        "confidence": 1.0,
                        "supporting_sample_count": int(support),
                    }
                )
    for row in bridge_rows:
        combined_rows.append(
            {
                "side": row["side"],
                "source_id": f"hypothesis_bridge_{row['bridge_id']}",
                "node_id": row["node_id"],
                "ordering_key": row["ordering_key"],
                "x_common_m": row["x_common_m"],
                "z_common_m": row["z_common_m"],
                "source": row["source"],
                "confidence": row["confidence"],
                "supporting_sample_count": 0,
            }
        )
    combined_rows.sort(key=lambda row: (row["side"], float(row["ordering_key"]), row["source"]))
    write_rows(args.output_dir / "occlusion_bridge_diagnostics.csv", diagnostics)
    write_rows(args.output_dir / "occlusion_bridge_nodes.csv", bridge_rows)
    write_rows(args.output_dir / "final_lane_nodes_with_hypotheses.csv", combined_rows)
    plot_result(args.output_dir / "occlusion_bridge_overview.png", segments, accepted)
    result = {
        "status": "complete",
        "coordinate_system": "metric Cartesian common-reference X/Z",
        "observed_segments": {
            side: len(segments.get(side, [])) for side in SIDES
        },
        "candidate_gaps": len(diagnostics),
        "accepted_hypothesis_bridges": len(accepted),
        "rejected_gaps": len(diagnostics) - len(accepted),
        "policy": {
            "observed_segments_are_never_modified": True,
            "endpoint_tangents_use_multiple_neighbour_nodes": True,
            "bridge_type": "cubic Hermite, G1 at both endpoints",
            "bridge_semantics": "low-confidence hypothesis, never an observation",
        },
        "warning": (
            "An accepted bridge is a geometry hypothesis inspired by pose-compensated "
            "temporal map memory. It is not detected lane evidence and must be drawn dashed."
        ),
    }
    (args.output_dir / "BRIDGE_RESULT.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
