"""Conservative leave-one-frame-out denoising for pose-aligned lane points."""

from __future__ import annotations

import numpy as np


def interpolate_lane(lane: np.ndarray, z: float) -> float | None:
    lane = np.asarray(lane, dtype=np.float64).reshape(-1, 2)
    order = np.argsort(lane[:, 1])
    sorted_lane = lane[order]
    unique_z, unique_index = np.unique(sorted_lane[:, 1], return_index=True)
    unique_x = sorted_lane[unique_index, 0]
    if len(unique_z) < 2 or z < unique_z[0] or z > unique_z[-1]:
        return None
    return float(np.interp(z, unique_z, unique_x))


def leave_one_out_statistics(
    points: np.ndarray,
    ranges: list[dict[str, object]],
) -> dict[str, np.ndarray]:
    """Compute residual to other frames' same-side median X at each point's Z."""

    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    residual = np.full(len(points), np.nan, dtype=np.float64)
    robust_sigma = np.full(len(points), np.nan, dtype=np.float64)
    support = np.zeros(len(points), dtype=np.int32)
    records = []
    for item in ranges:
        start, end = int(item["start"]), int(item["end"])
        records.append({**item, "lane": points[start:end]})

    for current in records:
        start, end = int(current["start"]), int(current["end"])
        peers = [
            item["lane"]
            for item in records
            if item["side"] == current["side"]
            and item["frame_id"] != current["frame_id"]
        ]
        for offset, point in enumerate(points[start:end]):
            predictions = [
                prediction
                for lane in peers
                if (prediction := interpolate_lane(lane, float(point[1])))
                is not None
            ]
            point_index = start + offset
            support[point_index] = len(predictions)
            if not predictions:
                continue
            values = np.asarray(predictions, dtype=np.float64)
            center = float(np.median(values))
            mad = float(np.median(np.abs(values - center)))
            robust_sigma[point_index] = 1.4826 * mad
            residual[point_index] = abs(float(point[0]) - center)
    return {
        "residual": residual,
        "robust_sigma": robust_sigma,
        "support": support,
    }


def consensus_mask_from_statistics(
    statistics: dict[str, np.ndarray],
    base_threshold: float = 0.30,
    mad_multiplier: float = 3.0,
    threshold_cap: float = 1.00,
    minimum_other_frames: int = 2,
) -> tuple[np.ndarray, dict[str, object]]:
    residual = statistics["residual"]
    robust_sigma = statistics["robust_sigma"]
    support = statistics["support"]
    evaluated = support >= minimum_other_frames
    thresholds = np.minimum(
        threshold_cap,
        np.maximum(base_threshold, mad_multiplier * robust_sigma),
    )
    keep = np.ones(len(residual), dtype=bool)
    keep[evaluated] = residual[evaluated] <= thresholds[evaluated]
    return keep, {
        "base_threshold_m": base_threshold,
        "mad_multiplier": mad_multiplier,
        "threshold_cap_m": threshold_cap,
        "minimum_other_frames": minimum_other_frames,
        "evaluated_points": int(np.count_nonzero(evaluated)),
        "unsupported_points_kept": int(np.count_nonzero(~evaluated)),
        "median_residual_m": (
            float(np.median(residual[evaluated])) if np.any(evaluated) else None
        ),
        "median_threshold_m": (
            float(np.median(thresholds[evaluated])) if np.any(evaluated) else None
        ),
        "minimum_support_seen": int(np.min(support)) if len(support) else 0,
    }


def leave_one_out_consensus_mask(
    points: np.ndarray,
    ranges: list[dict[str, object]],
    base_threshold: float = 0.30,
    mad_multiplier: float = 3.0,
    threshold_cap: float = 1.00,
    minimum_other_frames: int = 2,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return a conservative keep mask; unsupported points remain untouched."""

    statistics = leave_one_out_statistics(points, ranges)
    return consensus_mask_from_statistics(
        statistics,
        base_threshold,
        mad_multiplier,
        threshold_cap,
        minimum_other_frames,
    )
