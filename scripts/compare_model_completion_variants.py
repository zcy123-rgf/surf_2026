#!/usr/bin/env python3
"""Compare adaptive B-spline and polynomial-only lane maps before/after bridging.

The adaptive route uses the registered polynomial on straight windows and the
selected cubic B-spline on transition/curved windows.  The polynomial-only
route uses the registered quadratic parametric polynomial on every fittable
window.  Both routes are completed with the same audited Hermite bridge gates.
Observed curves are never modified and bridge samples remain explicit
low-confidence hypotheses.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import bridge_occluded_lane_segments as bridge


POLYNOMIAL = "parametric_polynomial"
BSPLINE = "parametric_cubic_bspline"
SIDES = ("left", "right")
SIDE_COLORS = {"left": "#1769aa", "right": "#d73027"}
OBS_COLORS = {"left": "#8fc5eb", "right": "#f3a29d"}
METRICS = {
    "Mean": "heldout_mean_m",
    "RMSE": "heldout_rmse_m",
    "Median": "heldout_median_m",
    "P90": "heldout_p90_m",
    "P95": "heldout_p95_m",
    "Max": "heldout_max_m",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Produce four maps and fixed-metric comparisons for Sequence 01 "
            "frames 851-1005."
        )
    )
    parser.add_argument("--workstation-outputs", type=Path)
    parser.add_argument("--bspline-result-dir", type=Path)
    parser.add_argument("--polynomial-result-dir", type=Path)
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
        raise FileExistsError(f"Output directory must be new or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def is_true(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def inspect_result_dir(path: Path) -> dict[str, object]:
    required = [
        path / "RESULT.json",
        path / "model_comparison.csv",
        path / "blended_lane_nodes.csv",
    ]
    if any(not item.is_file() for item in required):
        raise FileNotFoundError(f"Result directory is incomplete: {path}")
    result = read_json(path / "RESULT.json")
    rows = read_rows(path / "model_comparison.csv")
    selected = [row for row in rows if is_true(row.get("selected_for_output"))]
    selected_counts = Counter(row["model"] for row in selected)
    frames = result.get("requested_frames", result.get("frames"))
    provenance = result.get("provenance", {})
    return {
        "path": path.resolve(),
        "result": result,
        "frames": [int(value) for value in frames] if frames else None,
        "selected_model_counts": dict(selected_counts),
        "selected_lane_points_sha256": provenance.get("selected_lane_points_sha256"),
        "mtime": (path / "RESULT.json").stat().st_mtime,
    }


def discover_results(root: Path) -> tuple[Path, Path, list[dict[str, object]]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Workstation output root does not exist: {root}")
    candidates = []
    for result_json in root.rglob("RESULT.json"):
        path = result_json.parent
        try:
            info = inspect_result_dir(path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if info["frames"] != [851, 1005]:
            continue
        candidates.append(info)

    polynomial = [
        info
        for info in candidates
        if set(info["selected_model_counts"]) == {POLYNOMIAL}
        and int(info["selected_model_counts"].get(POLYNOMIAL, 0)) >= 20
    ]
    adaptive = [
        info
        for info in candidates
        if int(info["selected_model_counts"].get(BSPLINE, 0)) > 0
        and int(info["selected_model_counts"].get(POLYNOMIAL, 0)) > 0
    ]
    if not polynomial or not adaptive:
        summary = [
            {
                "path": str(info["path"]),
                "frames": info["frames"],
                "selected_model_counts": info["selected_model_counts"],
            }
            for info in candidates
        ]
        raise RuntimeError(
            "Could not auto-discover both required results. Pass explicit "
            "--bspline-result-dir and --polynomial-result-dir. Candidates: "
            + json.dumps(summary, ensure_ascii=False)
        )
    polynomial.sort(key=lambda item: float(item["mtime"]), reverse=True)
    adaptive.sort(key=lambda item: float(item["mtime"]), reverse=True)
    return Path(adaptive[0]["path"]), Path(polynomial[0]["path"]), candidates


def choose_inputs(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object]]:
    if bool(args.bspline_result_dir) != bool(args.polynomial_result_dir):
        raise ValueError("Provide both explicit result directories, or neither.")
    if args.bspline_result_dir:
        bspline_dir = args.bspline_result_dir
        polynomial_dir = args.polynomial_result_dir
    else:
        if args.workstation_outputs is None:
            raise ValueError(
                "Provide --workstation-outputs or both explicit result directories."
            )
        bspline_dir, polynomial_dir, _ = discover_results(args.workstation_outputs)

    adaptive = inspect_result_dir(bspline_dir)
    polynomial = inspect_result_dir(polynomial_dir)
    if adaptive["frames"] != [851, 1005] or polynomial["frames"] != [851, 1005]:
        raise ValueError("Both results must cover Sequence 01 frames 851-1005.")
    if BSPLINE not in adaptive["selected_model_counts"]:
        raise ValueError("The adaptive result does not contain selected B-spline fits.")
    if set(polynomial["selected_model_counts"]) != {POLYNOMIAL}:
        raise ValueError("The polynomial result is not polynomial-only.")
    first_hash = adaptive.get("selected_lane_points_sha256")
    second_hash = polynomial.get("selected_lane_points_sha256")
    if first_hash and second_hash and first_hash != second_hash:
        raise ValueError(
            "The two results do not use the same selected-lane-point input. "
            f"adaptive={first_hash}, polynomial={second_hash}"
        )
    return adaptive, polynomial


def bridge_namespace(
    blended_nodes: Path, output_dir: Path, args: argparse.Namespace
) -> argparse.Namespace:
    return argparse.Namespace(
        blended_nodes=blended_nodes,
        output_dir=output_dir,
        maximum_order_gap=args.maximum_order_gap,
        maximum_endpoint_gap_m=args.maximum_endpoint_gap_m,
        maximum_tangent_angle_deg=args.maximum_tangent_angle_deg,
        maximum_chord_tangent_angle_deg=args.maximum_chord_tangent_angle_deg,
        maximum_bridge_curvature_1pm=args.maximum_bridge_curvature_1pm,
        minimum_segment_nodes=args.minimum_segment_nodes,
        tangent_fit_nodes=args.tangent_fit_nodes,
        bridge_samples=args.bridge_samples,
        lane_width_min_m=args.lane_width_min_m,
        lane_width_max_m=args.lane_width_max_m,
        lane_width_p90_deviation_m=args.lane_width_p90_deviation_m,
    )


def grouped_nodes(path: Path) -> dict[tuple[str, int], np.ndarray]:
    grouped: dict[tuple[str, int], list[tuple[float, float, float]]] = defaultdict(list)
    for row in read_rows(path):
        grouped[(row["side"], int(row["segment_id"]))].append(
            (
                float(row["ordering_key"]),
                float(row["x_common_m"]),
                float(row["z_common_m"]),
            )
        )
    return {
        key: np.asarray([[x, z] for _, x, z in sorted(values)])
        for key, values in grouped.items()
    }


def grouped_bridges(path: Path) -> dict[tuple[str, int], np.ndarray]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    grouped: dict[tuple[str, int], list[tuple[int, float, float]]] = defaultdict(list)
    for row in read_rows(path):
        grouped[(row["side"], int(row["bridge_id"]))].append(
            (
                int(row["node_id"]),
                float(row["x_common_m"]),
                float(row["z_common_m"]),
            )
        )
    return {
        key: np.asarray([[x, z] for _, x, z in sorted(values)])
        for key, values in grouped.items()
    }


def optional_xy(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    output: dict[str, list[list[float]]] = defaultdict(list)
    for row in read_rows(path):
        side = row.get("side", "trajectory")
        output[side].append([float(row["x_common_m"]), float(row["z_common_m"])])
    return {key: np.asarray(values) for key, values in output.items()}


def all_plot_points(*collections) -> np.ndarray:
    arrays = []
    for collection in collections:
        arrays.extend(value for value in collection.values() if len(value))
    return np.vstack(arrays)


def plot_lane_map(
    ax,
    nodes: dict[tuple[str, int], np.ndarray],
    bridges: dict[tuple[str, int], np.ndarray],
    observations: dict[str, np.ndarray],
    trajectory: dict[str, np.ndarray],
    title: str,
    show_bridge: bool,
    limits: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    for side in SIDES:
        values = observations.get(side)
        if values is not None:
            ax.scatter(
                values[:, 0],
                values[:, 1],
                s=4,
                color=OBS_COLORS[side],
                alpha=0.18,
                linewidths=0,
                zorder=1,
            )
    route = trajectory.get("trajectory")
    if route is not None:
        ax.plot(route[:, 0], route[:, 1], color="#555555", linewidth=1.0, zorder=2)
    labelled = set()
    for (side, _segment_id), points in sorted(nodes.items()):
        label = None if side in labelled else f"{side} observed fit"
        labelled.add(side)
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=SIDE_COLORS[side],
            linewidth=2.4,
            label=label,
            zorder=3,
        )
    if show_bridge:
        bridge_labelled = set()
        for (side, _bridge_id), points in sorted(bridges.items()):
            label = None if side in bridge_labelled else f"{side} low-confidence bridge"
            bridge_labelled.add(side)
            ax.plot(
                points[:, 0],
                points[:, 1],
                color=SIDE_COLORS[side],
                linewidth=2.3,
                linestyle="--",
                alpha=0.82,
                label=label,
                zorder=4,
            )
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("X right in common reference [m]")
    ax.set_ylabel("Z forward in common reference [m]")
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.22)
    ax.legend(loc="best", fontsize=7.5)


def save_individual_and_four_way(
    output_dir: Path,
    adaptive_nodes,
    adaptive_bridges,
    polynomial_nodes,
    polynomial_bridges,
    observations,
    trajectory,
) -> None:
    points = all_plot_points(
        adaptive_nodes,
        adaptive_bridges,
        polynomial_nodes,
        polynomial_bridges,
        observations,
        trajectory,
    )
    x_min, z_min = np.min(points, axis=0)
    x_max, z_max = np.max(points, axis=0)
    x_pad = max(5.0, (x_max - x_min) * 0.05)
    z_pad = max(5.0, (z_max - z_min) * 0.05)
    limits = ((x_min - x_pad, x_max + x_pad), (z_min - z_pad, z_max + z_pad))

    variants = [
        (
            "01_adaptive_bspline_original",
            adaptive_nodes,
            adaptive_bridges,
            "Adaptive route: polynomial straight / B-spline curve - observed only",
            False,
        ),
        (
            "02_adaptive_bspline_completed",
            adaptive_nodes,
            adaptive_bridges,
            "Adaptive route with audited low-confidence bridge",
            True,
        ),
        (
            "03_polynomial_original",
            polynomial_nodes,
            polynomial_bridges,
            "Polynomial-only route - observed only",
            False,
        ),
        (
            "04_polynomial_completed",
            polynomial_nodes,
            polynomial_bridges,
            "Polynomial-only route with audited low-confidence bridge",
            True,
        ),
    ]
    for name, nodes, bridges, title, show_bridge in variants:
        fig, ax = plt.subplots(figsize=(8.0, 7.2), constrained_layout=True)
        plot_lane_map(
            ax, nodes, bridges, observations, trajectory, title, show_bridge, limits
        )
        fig.savefig(output_dir / f"{name}.png", dpi=300, facecolor="white")
        fig.savefig(output_dir / f"{name}.svg", facecolor="white")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 11.0), constrained_layout=True)
    for ax, (_name, nodes, bridges, title, show_bridge) in zip(axes.flat, variants):
        plot_lane_map(
            ax, nodes, bridges, observations, trajectory, title, show_bridge, limits
        )
    fig.suptitle(
        "Sequence 01 frames 851-1005: model and occlusion-completion comparison",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(output_dir / "05_four_way_lane_map_comparison.png", dpi=300, facecolor="white")
    fig.savefig(output_dir / "05_four_way_lane_map_comparison.svg", facecolor="white")
    fig.savefig(output_dir / "05_four_way_lane_map_comparison.pdf", facecolor="white")
    plt.close(fig)


def load_metric_pairs(path: Path) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in read_rows(path):
        classification = row["classification"]
        if classification not in {"transition", "curve"}:
            continue
        key = (int(row["window_id"]), row["side"], classification)
        grouped[key][row["model"]] = row
    output = []
    for (window_id, side, classification), models in grouped.items():
        if POLYNOMIAL not in models or BSPLINE not in models:
            continue
        row: dict[str, object] = {
            "window_id": window_id,
            "side": side,
            "classification": classification,
        }
        for label, field in METRICS.items():
            key = label.lower()
            row[f"polynomial_{key}_m"] = float(models[POLYNOMIAL][field])
            row[f"bspline_{key}_m"] = float(models[BSPLINE][field])
        output.append(row)
    return sorted(output, key=lambda row: (row["classification"], row["window_id"], row["side"]))


def metric_class_summary(pairs, classification):
    rows = [row for row in pairs if row["classification"] == classification]
    result: dict[str, object] = {"paired_window_sides": len(rows)}
    for label in METRICS:
        key = label.lower()
        polynomial = np.asarray([float(row[f"polynomial_{key}_m"]) for row in rows])
        bspline = np.asarray([float(row[f"bspline_{key}_m"]) for row in rows])
        result[label] = {
            "polynomial_macro_mean_m": float(np.mean(polynomial)),
            "bspline_macro_mean_m": float(np.mean(bspline)),
            "bspline_lower_pair_count": int(np.sum(bspline < polynomial)),
            "relative_change_percent": float(
                (np.mean(bspline) / np.mean(polynomial) - 1.0) * 100.0
            ),
        }
    return result


def plot_rmse_slope(ax, pairs, classification, title):
    rows = [row for row in pairs if row["classification"] == classification]
    polynomial = np.asarray([float(row["polynomial_rmse_m"]) for row in rows])
    bspline = np.asarray([float(row["bspline_rmse_m"]) for row in rows])
    for first, second in zip(polynomial, bspline):
        ax.plot(
            [0, 1],
            [first, second],
            color="#A9A9A9" if second < first else "#B0192E",
            linewidth=1.0,
            alpha=0.72,
        )
        ax.scatter(0, first, s=25, color="#2367A4", zorder=2)
        ax.scatter(1, second, s=25, color="#E26749", zorder=2)
    first_mean = float(np.mean(polynomial))
    second_mean = float(np.mean(bspline))
    ax.plot(
        [0, 1],
        [first_mean, second_mean],
        color="#111111",
        linewidth=3,
        marker="D",
        markersize=7,
        zorder=3,
    )
    wins = int(np.sum(bspline < polynomial))
    ax.text(
        0.5,
        0.98,
        f"B-spline lower RMSE: {wins}/{len(rows)}\n"
        f"{first_mean:.3f} -> {second_mean:.3f} m",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#CFCFCF"},
    )
    ax.set_title(title, fontweight="bold")
    ax.set_xticks([0, 1], ["Polynomial", "Cubic B-spline"])
    ax.set_ylabel("Held-out-frame RMSE [m]")
    ax.set_xlim(-0.28, 1.28)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def plot_fixed_metrics(ax, summary):
    names = list(METRICS)
    polynomial = np.asarray([summary[name]["polynomial_macro_mean_m"] for name in names])
    bspline = np.asarray([summary[name]["bspline_macro_mean_m"] for name in names])
    ratio = bspline / polynomial * 100.0
    wins = [summary[name]["bspline_lower_pair_count"] for name in names]
    positions = np.arange(len(names))
    colors = ["#009AA6" if value < 100.0 else "#B0192E" for value in ratio]
    bars = ax.barh(positions, ratio, color=colors, height=0.66)
    ax.axvline(100.0, color="#444444", linewidth=1.2, linestyle="--")
    pair_count = int(summary["paired_window_sides"])
    for bar, p_value, b_value, value, win_count in zip(
        bars, polynomial, bspline, ratio, wins
    ):
        ax.text(
            3.0,
            bar.get_y() + bar.get_height() / 2,
            f"{p_value:.3f} -> {b_value:.3f} m | lower {win_count}/{pair_count}",
            ha="left",
            va="center",
            fontsize=7.8,
            color="white",
            fontweight="bold",
        )
        ax.text(
            value + 1.4,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontsize=8.2,
            fontweight="bold",
        )
    ax.set_title("C. Fixed metrics on curved windows", fontweight="bold")
    ax.set_xlabel("B-spline / polynomial macro mean [%]")
    ax.set_yticks(positions, names)
    ax.invert_yaxis()
    ax.set_xlim(0, max(125.0, float(np.max(ratio)) + 13.0))
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def save_fit_metric_figure(output_dir: Path, pairs, summary) -> None:
    fig = plt.figure(figsize=(16.0, 5.4), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.35, 1.75])
    plot_rmse_slope(fig.add_subplot(grid[0, 0]), pairs, "transition", "A. Transition windows")
    plot_rmse_slope(fig.add_subplot(grid[0, 1]), pairs, "curve", "B. Curved windows")
    plot_fixed_metrics(fig.add_subplot(grid[0, 2]), summary["curve"])
    fig.suptitle(
        "Same-window model comparison on held-out observations",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(output_dir / "06_model_fit_metric_comparison.png", dpi=300, facecolor="white")
    fig.savefig(output_dir / "06_model_fit_metric_comparison.svg", facecolor="white")
    fig.savefig(output_dir / "06_model_fit_metric_comparison.pdf", facecolor="white")
    plt.close(fig)


def polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def completion_summary(
    variant: str,
    nodes: dict[tuple[str, int], np.ndarray],
    bridges: dict[tuple[str, int], np.ndarray],
    bridge_dir: Path,
) -> dict[str, object]:
    diagnostics = read_rows(bridge_dir / "occlusion_bridge_diagnostics.csv")
    accepted = [row for row in diagnostics if is_true(row["accepted_as_low_confidence_hypothesis"])]
    observed_length = sum(polyline_length(points) for points in nodes.values())
    bridge_length = sum(polyline_length(points) for points in bridges.values())
    confidence = [float(row["confidence"]) for row in accepted]
    return {
        "variant": variant,
        "observed_segments_left": sum(side == "left" for side, _ in nodes),
        "observed_segments_right": sum(side == "right" for side, _ in nodes),
        "observed_segments_total": len(nodes),
        "candidate_gaps": len(diagnostics),
        "accepted_bridges": len(accepted),
        "rejected_gaps": len(diagnostics) - len(accepted),
        "observed_lane_geometry_length_m": observed_length,
        "accepted_bridge_length_m": bridge_length,
        "represented_geometry_length_m": observed_length + bridge_length,
        "bridge_fraction_percent": (
            bridge_length / (observed_length + bridge_length) * 100.0
            if observed_length + bridge_length > 0
            else 0.0
        ),
        "mean_accepted_bridge_confidence": float(np.mean(confidence)) if confidence else math.nan,
    }


def save_completion_audit_figure(output_dir: Path, rows: list[dict[str, object]]) -> None:
    labels = ["Adaptive B-spline route", "Polynomial-only route"]
    observed = np.asarray([float(row["observed_lane_geometry_length_m"]) for row in rows])
    bridge_length = np.asarray([float(row["accepted_bridge_length_m"]) for row in rows])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10.5, 4.5), constrained_layout=True)
    ax.barh(y, observed, color="#6B7A8F", label="observation-supported fitted geometry")
    ax.barh(
        y,
        bridge_length,
        left=observed,
        color="#E26749",
        hatch="//",
        label="low-confidence Hermite hypothesis",
    )
    for index, row in enumerate(rows):
        ax.text(
            observed[index] + bridge_length[index] + 2.0,
            index,
            f"bridge {bridge_length[index]:.1f} m; "
            f"confidence {float(row['mean_accepted_bridge_confidence']):.3f}",
            va="center",
            fontsize=9.5,
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Sum of left/right lane-geometry length [m]")
    ax.set_title(
        "Occlusion completion audit: supported geometry remains separate from hypotheses",
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.25)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=True,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_dir / "07_completion_audit_comparison.png", dpi=300, facecolor="white")
    fig.savefig(output_dir / "07_completion_audit_comparison.svg", facecolor="white")
    fig.savefig(output_dir / "07_completion_audit_comparison.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    require_empty_output(args.output_dir)
    adaptive, polynomial = choose_inputs(args)

    bridge_root = args.output_dir / "bridge_data"
    adaptive_bridge_dir = bridge_root / "adaptive_bspline"
    polynomial_bridge_dir = bridge_root / "polynomial_only"
    bridge_root.mkdir(parents=True, exist_ok=True)
    adaptive_bridge_result = bridge.run(
        bridge_namespace(
            Path(adaptive["path"]) / "blended_lane_nodes.csv",
            adaptive_bridge_dir,
            args,
        )
    )
    polynomial_bridge_result = bridge.run(
        bridge_namespace(
            Path(polynomial["path"]) / "blended_lane_nodes.csv",
            polynomial_bridge_dir,
            args,
        )
    )

    adaptive_nodes = grouped_nodes(Path(adaptive["path"]) / "blended_lane_nodes.csv")
    polynomial_nodes = grouped_nodes(Path(polynomial["path"]) / "blended_lane_nodes.csv")
    adaptive_bridges = grouped_bridges(adaptive_bridge_dir / "occlusion_bridge_nodes.csv")
    polynomial_bridges = grouped_bridges(polynomial_bridge_dir / "occlusion_bridge_nodes.csv")
    observations = optional_xy(Path(adaptive["path"]) / "window_aggregate_points.csv")
    trajectory = optional_xy(Path(adaptive["path"]) / "trajectory_common_xz.csv")
    save_individual_and_four_way(
        args.output_dir,
        adaptive_nodes,
        adaptive_bridges,
        polynomial_nodes,
        polynomial_bridges,
        observations,
        trajectory,
    )

    pairs = load_metric_pairs(Path(polynomial["path"]) / "model_comparison.csv")
    metric_summary = {
        "pairing": "same window, same side, same held-out observations",
        "interpretation": (
            "internal consistency against held-out CLRNet/IPM observations; "
            "not official lane ground-truth accuracy"
        ),
        "transition": metric_class_summary(pairs, "transition"),
        "curve": metric_class_summary(pairs, "curve"),
    }
    write_rows(args.output_dir / "model_metric_pairs.csv", pairs)
    (args.output_dir / "model_metric_summary.json").write_text(
        json.dumps(metric_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_fit_metric_figure(args.output_dir, pairs, metric_summary)

    completion_rows = [
        completion_summary(
            "adaptive_bspline_route", adaptive_nodes, adaptive_bridges, adaptive_bridge_dir
        ),
        completion_summary(
            "polynomial_only_route", polynomial_nodes, polynomial_bridges, polynomial_bridge_dir
        ),
    ]
    write_rows(args.output_dir / "completion_audit_metrics.csv", completion_rows)
    save_completion_audit_figure(args.output_dir, completion_rows)

    status = {
        "status": "complete",
        "dataset": "KITTI Odometry Sequence 01",
        "frames": [851, 1005],
        "adaptive_bspline_result": str(adaptive["path"]),
        "polynomial_only_result": str(polynomial["path"]),
        "same_selected_lane_points_sha256": (
            adaptive.get("selected_lane_points_sha256")
            and adaptive.get("selected_lane_points_sha256")
            == polynomial.get("selected_lane_points_sha256")
        ),
        "adaptive_selected_model_counts": adaptive["selected_model_counts"],
        "polynomial_selected_model_counts": polynomial["selected_model_counts"],
        "adaptive_bridge_result": adaptive_bridge_result,
        "polynomial_bridge_result": polynomial_bridge_result,
        "historical_outputs_modified": False,
        "interpretation": [
            "solid curves are supported fitted outputs",
            "dashed bridges are low-confidence geometry hypotheses",
            "bridge samples are excluded from held-out model-error metrics",
            "model errors measure consistency with held-out CLRNet/IPM observations, not official lane accuracy",
        ],
    }
    (args.output_dir / "STATUS.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "README_RESULT.md").write_text(
        "\n".join(
            [
                "# Four-way lane-model and occlusion-completion comparison",
                "",
                "- Solid curves: observation-supported fitted geometry.",
                "- Dashed curves: accepted low-confidence cubic Hermite hypotheses.",
                "- The adaptive route uses a polynomial on straight windows and a cubic B-spline on transition/curved windows.",
                "- The polynomial-only route uses a quadratic parametric polynomial on every fittable window.",
                "- Completion never modifies observed segments and never enters the held-out error metrics.",
                "- All model-error metrics are internal consistency against held-out CLRNet/IPM observations, not official lane ground truth.",
                "",
                "Open `05_four_way_lane_map_comparison.png`, `06_model_fit_metric_comparison.png`, and `07_completion_audit_comparison.png` first.",
            ]
        ),
        encoding="utf-8",
    )

    zip_path = args.output_dir.with_suffix(".zip")
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=args.output_dir)
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(args.output_dir),
                "zip": str(zip_path),
                "adaptive_bridges": adaptive_bridge_result["accepted_hypothesis_bridges"],
                "polynomial_bridges": polynomial_bridge_result["accepted_hypothesis_bridges"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
