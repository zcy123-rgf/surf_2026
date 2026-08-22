#!/usr/bin/env python3
"""Estimate road-trajectory curvature from KITTI poses.

The saved geometry remains in metric Cartesian X/Z.  Arc length is used only
as the independent variable needed to calculate heading change per metre.
Thresholds are registered explicitly and are raised when a straight seed
reveals a larger pose-noise floor.  The script never overwrites a non-empty
output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


STATE_COLOURS = {
    "straight": "#3b7d3b",
    "enter_transition": "#d98c2b",
    "curve": "#b8322d",
    "exit_transition": "#7967ad",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pose-derived curvature and straight/curve hysteresis audit."
    )
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--sequence-id", default="unknown")
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--straight-seed-ranges", default="")
    parser.add_argument("--resample-step-m", type=float, default=0.5)
    parser.add_argument("--smoothing-window-m", type=float, default=15.0)
    parser.add_argument("--minimum-straight-threshold-1pm", type=float, default=0.0015)
    parser.add_argument("--minimum-curve-threshold-1pm", type=float, default=0.0030)
    parser.add_argument("--straight-noise-sigma", type=float, default=3.0)
    parser.add_argument("--curve-noise-sigma", type=float, default=6.0)
    parser.add_argument("--minimum-state-run-frames", type=int, default=5)
    parser.add_argument("--window-length", type=int, default=15)
    parser.add_argument("--window-stride", type=int, default=10)
    return parser.parse_args()


def require_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def load_pose_positions(path: Path) -> np.ndarray:
    raw = np.loadtxt(path, dtype=np.float64)
    raw = np.atleast_2d(raw)
    if raw.shape[1] != 12:
        raise ValueError(f"KITTI pose rows must have 12 values, got {raw.shape}.")
    poses = raw.reshape(-1, 3, 4)
    return poses[:, :, 3][:, [0, 2]]


def cumulative_distance(points: np.ndarray) -> np.ndarray:
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def odd_window(sample_count: int, requested: int) -> int:
    value = max(5, requested)
    if value % 2 == 0:
        value += 1
    maximum = sample_count if sample_count % 2 == 1 else sample_count - 1
    return max(3, min(value, maximum))


def estimate_curvature(
    positions_xz: np.ndarray,
    resample_step_m: float,
    smoothing_window_m: float,
) -> dict[str, np.ndarray | float | int]:
    original_s = cumulative_distance(positions_xz)
    keep = np.concatenate([[True], np.diff(original_s) > 1e-6])
    unique_s = original_s[keep]
    unique_points = positions_xz[keep]
    if len(unique_points) < 7 or unique_s[-1] <= 0:
        raise ValueError("Too little non-stationary trajectory support for curvature.")
    sample_count = max(7, int(math.floor(unique_s[-1] / resample_step_m)) + 1)
    uniform_s = np.linspace(0.0, float(unique_s[-1]), sample_count)
    step = float(uniform_s[1] - uniform_s[0])
    x = np.interp(uniform_s, unique_s, unique_points[:, 0])
    z = np.interp(uniform_s, unique_s, unique_points[:, 1])
    window = odd_window(sample_count, int(round(smoothing_window_m / step)))
    degree = min(3, window - 2)
    x_smooth = savgol_filter(x, window, degree, mode="interp")
    z_smooth = savgol_filter(z, window, degree, mode="interp")
    dx = savgol_filter(x, window, degree, deriv=1, delta=step, mode="interp")
    dz = savgol_filter(z, window, degree, deriv=1, delta=step, mode="interp")
    ddx = savgol_filter(x, window, degree, deriv=2, delta=step, mode="interp")
    ddz = savgol_filter(z, window, degree, deriv=2, delta=step, mode="interp")
    denominator = np.maximum((dx * dx + dz * dz) ** 1.5, 1e-9)
    signed_kappa = (dx * ddz - dz * ddx) / denominator
    frame_kappa = np.interp(original_s, uniform_s, signed_kappa)
    heading = np.unwrap(np.arctan2(dx, dz))
    frame_heading = np.interp(original_s, uniform_s, heading)
    return {
        "frame_s_m": original_s,
        "frame_signed_curvature_1pm": frame_kappa,
        "frame_heading_rad": frame_heading,
        "uniform_s_m": uniform_s,
        "uniform_x_m": x_smooth,
        "uniform_z_m": z_smooth,
        "uniform_signed_curvature_1pm": signed_kappa,
        "effective_resample_step_m": step,
        "savgol_window_samples": window,
        "savgol_degree": degree,
    }


def parse_ranges(text: str, minimum: int, maximum: int) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    if not text.strip():
        return output
    for token in text.split(","):
        pieces = token.strip().split("-")
        if len(pieces) != 2:
            raise ValueError(f"Bad frame range: {token}")
        start, end = map(int, pieces)
        if start > end or start < minimum or end > maximum:
            raise ValueError(f"Straight seed range is outside requested frames: {token}")
        output.append((start, end))
    return output


def seed_mask(frame_ids: np.ndarray, ranges: Iterable[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(len(frame_ids), dtype=bool)
    for start, end in ranges:
        mask |= (frame_ids >= start) & (frame_ids <= end)
    return mask


def calibrate_thresholds(
    absolute_curvature: np.ndarray,
    baseline_mask: np.ndarray,
    minimum_straight: float,
    minimum_curve: float,
    straight_sigma: float,
    curve_sigma: float,
) -> dict[str, float | int | str]:
    if baseline_mask.any():
        baseline = absolute_curvature[baseline_mask]
        source = "registered_straight_seed_ranges"
    else:
        cutoff = float(np.percentile(absolute_curvature, 25))
        baseline = absolute_curvature[absolute_curvature <= cutoff]
        source = "lowest_curvature_quartile_fallback"
    centre = float(np.median(baseline))
    sigma = float(1.4826 * np.median(np.abs(baseline - centre)))
    straight_threshold = max(minimum_straight, centre + straight_sigma * sigma)
    curve_threshold = max(
        minimum_curve,
        centre + curve_sigma * sigma,
        1.5 * straight_threshold,
    )
    return {
        "baseline_source": source,
        "baseline_sample_count": int(len(baseline)),
        "baseline_median_abs_curvature_1pm": centre,
        "baseline_robust_sigma_1pm": sigma,
        "straight_threshold_abs_curvature_1pm": float(straight_threshold),
        "curve_threshold_abs_curvature_1pm": float(curve_threshold),
    }


def hysteresis_states(
    absolute_curvature: np.ndarray,
    straight_threshold: float,
    curve_threshold: float,
    minimum_run: int,
) -> list[str]:
    states = ["straight"] * len(absolute_curvature)
    active_curve = False
    pending_start: int | None = None
    for index, value in enumerate(absolute_curvature):
        if not active_curve:
            if value >= curve_threshold:
                if pending_start is None:
                    pending_start = index
                states[index] = "enter_transition"
                if index - pending_start + 1 >= minimum_run:
                    for back in range(pending_start, index + 1):
                        states[back] = "curve"
                    active_curve = True
                    pending_start = None
            elif value > straight_threshold:
                states[index] = "enter_transition"
                pending_start = None
            else:
                states[index] = "straight"
                pending_start = None
        else:
            if value <= straight_threshold:
                if pending_start is None:
                    pending_start = index
                states[index] = "exit_transition"
                if index - pending_start + 1 >= minimum_run:
                    for back in range(pending_start, index + 1):
                        states[back] = "straight"
                    active_curve = False
                    pending_start = None
            elif value < curve_threshold:
                states[index] = "exit_transition"
                pending_start = None
            else:
                states[index] = "curve"
                pending_start = None
    return states


def contiguous_segments(frame_ids: np.ndarray, states: list[str]) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    start = 0
    for index in range(1, len(states) + 1):
        if index == len(states) or states[index] != states[start]:
            segments.append(
                {
                    "segment_id": len(segments),
                    "state": states[start],
                    "start_frame": int(frame_ids[start]),
                    "end_frame": int(frame_ids[index - 1]),
                    "frame_count": int(index - start),
                }
            )
            start = index
    return segments


def windows(start: int, end: int, length: int, stride: int) -> list[tuple[int, int]]:
    if end - start + 1 < length:
        return []
    starts = list(range(start, end - length + 2, stride))
    terminal = end - length + 1
    # Match the fitting stage exactly: always add a terminal anchor when the
    # regular stride does not land on the requested final frame.
    if starts[-1] != terminal:
        starts.append(terminal)
    return [(value, value + length - 1) for value in starts]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_result(
    path: Path,
    frame_ids: np.ndarray,
    positions: np.ndarray,
    signed_curvature: np.ndarray,
    states: list[str],
    straight_threshold: float,
    curve_threshold: float,
    sequence_id: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=170)
    ax = axes[0]
    for state, colour in STATE_COLOURS.items():
        mask = np.asarray(states) == state
        ax.scatter(positions[mask, 0], positions[mask, 1], s=12, c=colour, label=state)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(f"Sequence {sequence_id} pose trajectory states")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(frame_ids, np.abs(signed_curvature), color="#2b6cb0", linewidth=1.5)
    ax.axhline(straight_threshold, color=STATE_COLOURS["straight"], linestyle="--", label="straight threshold")
    ax.axhline(curve_threshold, color=STATE_COLOURS["curve"], linestyle="--", label="curve threshold")
    ax.set_xlabel("frame")
    ax.set_ylabel("absolute curvature [1/m]")
    ax.set_title("Pose-derived curvature with registered hysteresis")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.end_frame < args.start_frame:
        raise ValueError("End frame must not precede start frame.")
    if min(
        args.resample_step_m,
        args.smoothing_window_m,
        args.minimum_straight_threshold_1pm,
        args.minimum_curve_threshold_1pm,
    ) <= 0:
        raise ValueError("Curvature distances and thresholds must be positive.")
    if args.minimum_curve_threshold_1pm <= args.minimum_straight_threshold_1pm:
        raise ValueError("Curve threshold must exceed straight threshold.")
    require_empty_output(args.output_dir)
    all_positions = load_pose_positions(args.poses)
    if args.end_frame >= len(all_positions):
        raise ValueError("Requested frame exceeds pose file length.")
    frame_ids = np.arange(args.start_frame, args.end_frame + 1, dtype=np.int64)
    positions = all_positions[frame_ids]
    estimate = estimate_curvature(
        positions, args.resample_step_m, args.smoothing_window_m
    )
    signed_curvature = np.asarray(estimate["frame_signed_curvature_1pm"])
    absolute_curvature = np.abs(signed_curvature)
    ranges = parse_ranges(
        args.straight_seed_ranges, args.start_frame, args.end_frame
    )
    thresholds = calibrate_thresholds(
        absolute_curvature,
        seed_mask(frame_ids, ranges),
        args.minimum_straight_threshold_1pm,
        args.minimum_curve_threshold_1pm,
        args.straight_noise_sigma,
        args.curve_noise_sigma,
    )
    straight_threshold = float(thresholds["straight_threshold_abs_curvature_1pm"])
    curve_threshold = float(thresholds["curve_threshold_abs_curvature_1pm"])
    states = hysteresis_states(
        absolute_curvature,
        straight_threshold,
        curve_threshold,
        args.minimum_state_run_frames,
    )
    frame_rows = []
    s_m = np.asarray(estimate["frame_s_m"])
    heading = np.asarray(estimate["frame_heading_rad"])
    for index, frame_id in enumerate(frame_ids):
        frame_rows.append(
            {
                "frame_id": int(frame_id),
                "arc_length_from_start_m": float(s_m[index]),
                "x_m": float(positions[index, 0]),
                "z_m": float(positions[index, 1]),
                "heading_deg": float(np.degrees(heading[index])),
                "signed_curvature_1pm": float(signed_curvature[index]),
                "absolute_curvature_1pm": float(absolute_curvature[index]),
                "state": states[index],
            }
        )
    window_rows = []
    for window_id, (start, end) in enumerate(
        windows(args.start_frame, args.end_frame, args.window_length, args.window_stride)
    ):
        mask = (frame_ids >= start) & (frame_ids <= end)
        values = absolute_curvature[mask]
        labels = np.asarray(states)[mask]
        counts = {state: int(np.sum(labels == state)) for state in STATE_COLOURS}
        if counts["curve"] >= max(counts["straight"], 1):
            classification = "curve"
        elif counts["straight"] > counts["curve"] and not (
            counts["enter_transition"] or counts["exit_transition"]
        ):
            classification = "straight"
        else:
            classification = "transition"
        window_rows.append(
            {
                "window_id": window_id,
                "start_frame": start,
                "end_frame": end,
                "classification": classification,
                "median_absolute_curvature_1pm": float(np.median(values)),
                "p90_absolute_curvature_1pm": float(np.percentile(values, 90)),
                **{f"{state}_frame_count": count for state, count in counts.items()},
            }
        )
    segments = contiguous_segments(frame_ids, states)
    write_csv(args.output_dir / "pose_curvature_frames.csv", frame_rows)
    write_csv(args.output_dir / "pose_curvature_windows.csv", window_rows)
    write_csv(args.output_dir / "pose_curvature_segments.csv", segments)
    plot_result(
        args.output_dir / "pose_curvature_overview.png",
        frame_ids,
        positions,
        signed_curvature,
        states,
        straight_threshold,
        curve_threshold,
        args.sequence_id,
    )
    result = {
        "status": "complete",
        "sequence_id": args.sequence_id,
        "requested_frames": [args.start_frame, args.end_frame],
        "coordinate_system": "KITTI pose metric Cartesian X/Z",
        "curvature_definition": "signed heading change per metric arc length, 1/m",
        "thresholds": thresholds,
        "straight_seed_ranges": [list(item) for item in ranges],
        "smoothing": {
            "method": "Savitzky-Golay on uniformly resampled pose positions",
            "requested_window_m": args.smoothing_window_m,
            "effective_resample_step_m": estimate["effective_resample_step_m"],
            "window_samples": estimate["savgol_window_samples"],
            "polynomial_degree": estimate["savgol_degree"],
        },
        "hysteresis": {
            "minimum_state_run_frames": args.minimum_state_run_frames,
            "states": list(STATE_COLOURS),
        },
        "counts": {
            "frames": len(frame_ids),
            "windows": len(window_rows),
            "segments": len(segments),
            "state_frames": {
                state: int(states.count(state)) for state in STATE_COLOURS
            },
        },
        "warning": (
            "These are trajectory-shape labels derived from poses, not official "
            "KITTI road-geometry annotations. Numerical thresholds are registered "
            "project thresholds raised by the measured straight-seed noise floor."
        ),
    }
    (args.output_dir / "CURVATURE_RESULT.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
