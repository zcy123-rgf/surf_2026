"""Auditable serialization of metric lane-point denoising decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


CORE_COLUMNS = (
    "point_id",
    "frame_index",
    "frame_id",
    "side",
    "lane_index",
    "point_index_in_lane",
    "x_m",
    "z_m",
)


def _validate(
    points: np.ndarray,
    ranges: Sequence[Mapping[str, object]],
    masks: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    point_array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    cursor = 0
    for record in ranges:
        start, end = int(record["start"]), int(record["end"])
        if start != cursor or end < start or end > len(point_array):
            raise ValueError(
                "Point ranges must cover the flattened point array once, in order."
            )
        cursor = end
    if cursor != len(point_array):
        raise ValueError("Point ranges do not cover every flattened point.")

    normalized: dict[str, np.ndarray] = {}
    for name, mask in masks.items():
        value = np.asarray(mask, dtype=bool).reshape(-1)
        if len(value) != len(point_array):
            raise ValueError(
                f"Mask {name!r} has {len(value)} entries for "
                f"{len(point_array)} points."
            )
        normalized[str(name)] = value
    if not normalized:
        raise ValueError("At least one denoising method mask is required.")
    return point_array, normalized


def build_point_decision_rows(
    points: np.ndarray,
    ranges: Sequence[Mapping[str, object]],
    masks: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    """Return one tabular row per source point with every method decision."""

    point_array, normalized = _validate(points, ranges, masks)
    rows: list[dict[str, object]] = []
    for record in ranges:
        start, end = int(record["start"]), int(record["end"])
        for offset, point_id in enumerate(range(start, end)):
            row: dict[str, object] = {
                "point_id": point_id,
                "frame_index": int(record.get("frame_index", record["frame_id"])),
                "frame_id": int(record["frame_id"]),
                "side": str(record["side"]),
                "lane_index": (
                    int(record["lane_index"])
                    if record.get("lane_index") is not None
                    else ""
                ),
                "point_index_in_lane": offset,
                "x_m": float(point_array[point_id, 0]),
                "z_m": float(point_array[point_id, 1]),
            }
            row.update(
                {
                    method: int(mask[point_id])
                    for method, mask in normalized.items()
                }
            )
            rows.append(row)
    return rows


def rows_for_method(
    decision_rows: Sequence[Mapping[str, object]],
    method: str,
    kept: bool = True,
) -> list[dict[str, object]]:
    """Select kept or rejected coordinates and retain only one decision column."""

    wanted = int(kept)
    output = []
    for source in decision_rows:
        if method not in source:
            raise KeyError(f"Unknown method column: {method}")
        if int(source[method]) != wanted:
            continue
        row = {name: source[name] for name in CORE_COLUMNS}
        row["kept"] = wanted
        output.append(row)
    return output


def build_point_sets_document(
    points: np.ndarray,
    ranges: Sequence[Mapping[str, object]],
    masks: Mapping[str, np.ndarray],
    *,
    coordinate_system: str,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a JSON-ready, per-method/per-frame coordinate document."""

    point_array, normalized = _validate(points, ranges, masks)
    methods: dict[str, object] = {}
    for method, mask in normalized.items():
        frames: dict[tuple[int, int], dict[str, object]] = {}
        for record in ranges:
            start, end = int(record["start"]), int(record["end"])
            frame_index = int(record.get("frame_index", record["frame_id"]))
            frame_id = int(record["frame_id"])
            frame = frames.setdefault(
                (frame_index, frame_id),
                {
                    "frame_index": frame_index,
                    "frame_id": frame_id,
                    "lanes": [],
                },
            )
            ids = np.arange(start, end, dtype=np.int64)
            lane_mask = mask[start:end]
            kept_ids = ids[lane_mask]
            rejected_ids = ids[~lane_mask]
            frame["lanes"].append(
                {
                    "side": str(record["side"]),
                    "lane_index": (
                        int(record["lane_index"])
                        if record.get("lane_index") is not None
                        else None
                    ),
                    "source_start": start,
                    "source_end": end,
                    "kept_source_point_ids": kept_ids.tolist(),
                    "rejected_source_point_ids": rejected_ids.tolist(),
                    "kept_points_xz_m": point_array[kept_ids].tolist(),
                    "rejected_points_xz_m": point_array[rejected_ids].tolist(),
                }
            )
        methods[method] = {
            "kept_points": int(np.count_nonzero(mask)),
            "rejected_points": int(np.count_nonzero(~mask)),
            "frames": [frames[key] for key in sorted(frames)],
        }

    return {
        "schema_version": 1,
        "coordinate_system": coordinate_system,
        "point_order_note": (
            "point_id indexes the common flattened pre-denoising point array; "
            "all methods are decisions on exactly that array."
        ),
        "metadata": dict(metadata or {}),
        "total_source_points": int(len(point_array)),
        "methods": methods,
    }
