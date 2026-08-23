"""Post-hoc fixed metrics for completed SURF full-Sequence results.

The evaluator never treats CLRNet/IPM observations as lane ground truth. It
compares the selected per-window fitted curve with the observations already
used by the pipeline, and separately summarizes the saved held-out-frame
residuals. Historical result files are read-only; every report is written to
a new output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


THRESHOLDS_M = (0.3, 0.5, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resample-spacing-m", type=float, default=0.5)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require_new_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def finite_float(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def points_from_rows(
    rows: Iterable[dict[str, str]], order_column: str
) -> np.ndarray:
    ordered = sorted(rows, key=lambda row: finite_float(row.get(order_column)))
    points = np.array(
        [
            [finite_float(row.get("x_common_m")), finite_float(row.get("z_common_m"))]
            for row in ordered
        ],
        dtype=float,
    )
    if not len(points):
        return np.empty((0, 2), dtype=float)
    return points[np.isfinite(points).all(axis=1)]


def resample_polyline(points: np.ndarray, spacing_m: float) -> np.ndarray:
    if len(points) < 2:
        return points.copy()
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9]
    points = points[keep]
    if len(points) < 2:
        return points.copy()
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    total = float(cumulative[-1])
    if total <= 1e-9:
        return points[:1].copy()
    sample_count = max(2, int(math.ceil(total / spacing_m)) + 1)
    query = np.linspace(0.0, total, sample_count)
    return np.column_stack(
        [
            np.interp(query, cumulative, points[:, 0]),
            np.interp(query, cumulative, points[:, 1]),
        ]
    )


def directed_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if not len(source) or not len(target):
        return np.empty(0, dtype=float)
    distances, _ = cKDTree(target).query(source, k=1)
    return np.asarray(distances, dtype=float)


def distance_stats(values: np.ndarray, prefix: str) -> dict[str, float | int]:
    if not len(values):
        return {f"{prefix}_count": 0}
    return {
        f"{prefix}_count": int(len(values)),
        f"{prefix}_mean_m": float(np.mean(values)),
        f"{prefix}_rmse_m": float(np.sqrt(np.mean(values**2))),
        f"{prefix}_median_m": float(np.median(values)),
        f"{prefix}_p90_m": float(np.quantile(values, 0.90)),
        f"{prefix}_p95_m": float(np.quantile(values, 0.95)),
        f"{prefix}_max_m": float(np.max(values)),
    }


def threshold_key(value: float) -> str:
    return str(value).replace(".", "p")


def discrete_frechet(first: np.ndarray, second: np.ndarray) -> float:
    if not len(first) or not len(second):
        return float("nan")
    previous = np.full(len(second), np.inf, dtype=float)
    for i, point_a in enumerate(first):
        current = np.full(len(second), np.inf, dtype=float)
        for j, point_b in enumerate(second):
            distance = float(np.linalg.norm(point_a - point_b))
            if i == 0 and j == 0:
                current[j] = distance
            elif i == 0:
                current[j] = max(current[j - 1], distance)
            elif j == 0:
                current[j] = max(previous[j], distance)
            else:
                current[j] = max(
                    min(previous[j], previous[j - 1], current[j - 1]), distance
                )
        previous = current
    return float(previous[-1])


def evaluate_pair(
    observed: np.ndarray, predicted: np.ndarray, spacing_m: float
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    observed_sampled = resample_polyline(observed, spacing_m)
    predicted_sampled = resample_polyline(predicted, spacing_m)
    pred_to_obs = directed_distances(predicted_sampled, observed_sampled)
    obs_to_pred = directed_distances(observed_sampled, predicted_sampled)
    symmetric = np.r_[pred_to_obs, obs_to_pred]
    result: dict[str, object] = {
        "observed_raw_count": int(len(observed)),
        "predicted_raw_count": int(len(predicted)),
        "observed_resampled_count": int(len(observed_sampled)),
        "predicted_resampled_count": int(len(predicted_sampled)),
        "resample_spacing_m": spacing_m,
        **distance_stats(pred_to_obs, "prediction_to_observation"),
        **distance_stats(obs_to_pred, "observation_to_prediction"),
        **distance_stats(symmetric, "symmetric"),
        "symmetric_chamfer_m": (
            float(0.5 * (np.mean(pred_to_obs) + np.mean(obs_to_pred)))
            if len(pred_to_obs) and len(obs_to_pred)
            else float("nan")
        ),
        "discrete_frechet_m": discrete_frechet(observed_sampled, predicted_sampled),
    }
    for threshold in THRESHOLDS_M:
        key = threshold_key(threshold)
        result[f"prediction_precision_at_{key}m"] = (
            float(np.mean(pred_to_obs <= threshold)) if len(pred_to_obs) else float("nan")
        )
        result[f"observation_coverage_at_{key}m"] = (
            float(np.mean(obs_to_pred <= threshold)) if len(obs_to_pred) else float("nan")
        )
    return result, pred_to_obs, obs_to_pred


def find_adaptive_dir(result_directory: Path) -> Path:
    candidate = result_directory / "03_curve_models_and_fusion"
    if candidate.is_dir():
        return candidate
    if (result_directory / "model_comparison.csv").is_file():
        return result_directory
    raise FileNotFoundError(f"Adaptive result directory was not found: {result_directory}")


def group_rows(
    rows: list[dict[str, str]], keys: tuple[str, ...]
) -> dict[tuple[str, ...], list[dict[str, str]]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(key, "")) for key in keys)].append(row)
    return grouped


def summarize_heldout(rows: list[dict[str, str]]) -> dict[str, object]:
    selected = [row for row in rows if row.get("selected_for_output") == "True"]
    counts = np.array([finite_float(row.get("heldout_count")) for row in selected])
    means = np.array([finite_float(row.get("heldout_mean_m")) for row in selected])
    rmses = np.array([finite_float(row.get("heldout_rmse_m")) for row in selected])
    p90s = np.array([finite_float(row.get("heldout_p90_m")) for row in selected])
    valid = np.isfinite(counts) & np.isfinite(means) & np.isfinite(rmses) & (counts > 0)
    p90_valid = p90s[np.isfinite(p90s)]
    if not np.any(valid):
        return {"selected_window_side_count": len(selected), "heldout_count": 0}
    return {
        "selected_window_side_count": len(selected),
        "heldout_count": int(np.sum(counts[valid])),
        "heldout_count_weighted_mean_m": float(
            np.average(means[valid], weights=counts[valid])
        ),
        "heldout_count_weighted_rmse_m": float(
            np.sqrt(np.average(rmses[valid] ** 2, weights=counts[valid]))
        ),
        "heldout_window_p90_median_m": (
            float(np.median(p90_valid)) if len(p90_valid) else float("nan")
        ),
        "heldout_window_p90_p90_m": (
            float(np.quantile(p90_valid, 0.90)) if len(p90_valid) else float("nan")
        ),
        "p90_interpretation": (
            "distribution across saved window-side P90 values; not a global raw-residual P90"
        ),
    }


def evaluate_sequence(
    sequence_id: str, result_directory: Path, spacing_m: float
) -> tuple[dict[str, object], list[dict[str, object]]]:
    adaptive = find_adaptive_dir(result_directory)
    aggregate_path = adaptive / "window_aggregate_points.csv"
    curves_path = adaptive / "window_curve_samples.csv"
    models_path = adaptive / "model_comparison.csv"
    final_metrics_path = result_directory / "FINAL_METRICS.json"
    for required in (aggregate_path, curves_path, models_path, final_metrics_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required completed-result file is missing: {required}")

    observations = read_csv(aggregate_path)
    curves = read_csv(curves_path)
    model_rows = read_csv(models_path)
    final_metrics = json.loads(final_metrics_path.read_text(encoding="utf-8-sig"))
    observation_groups = group_rows(observations, ("window_id", "side"))
    curve_groups = group_rows(curves, ("window_id", "side"))
    pair_keys = sorted(set(observation_groups) & set(curve_groups))
    window_rows: list[dict[str, object]] = []
    all_pred_to_obs: list[np.ndarray] = []
    all_obs_to_pred: list[np.ndarray] = []
    frechet_values: list[float] = []

    for window_id, side in pair_keys:
        observation_rows = observation_groups[(window_id, side)]
        curve_rows = curve_groups[(window_id, side)]
        observed = points_from_rows(observation_rows, "point_id")
        predicted = points_from_rows(curve_rows, "sample_id")
        if len(observed) < 2 or len(predicted) < 2:
            continue
        pair_metrics, pred_to_obs, obs_to_pred = evaluate_pair(
            observed, predicted, spacing_m
        )
        first_curve = curve_rows[0]
        row: dict[str, object] = {
            "sequence_id": sequence_id,
            "window_id": int(float(window_id)),
            "side": side,
            "classification": first_curve.get("classification", ""),
            "model": first_curve.get("model", ""),
            **pair_metrics,
        }
        window_rows.append(row)
        all_pred_to_obs.append(pred_to_obs)
        all_obs_to_pred.append(obs_to_pred)
        frechet = finite_float(pair_metrics.get("discrete_frechet_m"))
        if math.isfinite(frechet):
            frechet_values.append(frechet)

    if not window_rows:
        raise ValueError(f"Sequence {sequence_id} has no evaluable window-side pairs.")
    pred_to_obs_all = np.concatenate(all_pred_to_obs)
    obs_to_pred_all = np.concatenate(all_obs_to_pred)
    symmetric_all = np.r_[pred_to_obs_all, obs_to_pred_all]
    summary: dict[str, object] = {
        "sequence_id": sequence_id,
        "status": "complete",
        "result_directory": str(result_directory),
        "window_side_pair_count": len(window_rows),
        "scope": (
            "post-hoc agreement with CLRNet/IPM observations; not lane ground-truth accuracy"
        ),
        "sampling_note": (
            "overlapping windows are evaluated separately after equal arc-length resampling"
        ),
        **distance_stats(pred_to_obs_all, "prediction_to_observation"),
        **distance_stats(obs_to_pred_all, "observation_to_prediction"),
        **distance_stats(symmetric_all, "symmetric"),
        "symmetric_chamfer_m": float(
            0.5 * (np.mean(pred_to_obs_all) + np.mean(obs_to_pred_all))
        ),
        "window_discrete_frechet_median_m": float(np.median(frechet_values)),
        "window_discrete_frechet_p90_m": float(np.quantile(frechet_values, 0.90)),
        "heldout_consistency": summarize_heldout(model_rows),
        "observation_coverage": final_metrics.get("observation_coverage", {}),
        "fitting_coverage": final_metrics.get("fitting_coverage", {}),
        "continuity": final_metrics.get("continuity", {}),
    }
    for threshold in THRESHOLDS_M:
        key = threshold_key(threshold)
        summary[f"prediction_precision_at_{key}m"] = float(
            np.mean(pred_to_obs_all <= threshold)
        )
        summary[f"observation_coverage_at_{key}m"] = float(
            np.mean(obs_to_pred_all <= threshold)
        )
    return summary, window_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames and not isinstance(row[key], (dict, list)):
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: object) -> object:
    """Replace non-finite floats recursively so strict JSON remains valid."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def summary_csv_row(summary: dict[str, object]) -> dict[str, object]:
    heldout = summary.get("heldout_consistency", {})
    observation = summary.get("observation_coverage", {})
    fitting = summary.get("fitting_coverage", {})
    continuity = summary.get("continuity", {})
    assert isinstance(heldout, dict)
    assert isinstance(observation, dict)
    assert isinstance(fitting, dict)
    assert isinstance(continuity, dict)
    return {
        "sequence_id": summary["sequence_id"],
        "status": summary["status"],
        "window_side_pair_count": summary["window_side_pair_count"],
        "both_observation_fraction": observation.get("both_fraction"),
        "completed_fit_fraction": fitting.get("completed_fraction"),
        "continuity_pass_fraction": continuity.get("passed_fraction"),
        "symmetric_mean_m": summary.get("symmetric_mean_m"),
        "symmetric_rmse_m": summary.get("symmetric_rmse_m"),
        "symmetric_median_m": summary.get("symmetric_median_m"),
        "symmetric_p90_m": summary.get("symmetric_p90_m"),
        "symmetric_max_m": summary.get("symmetric_max_m"),
        "symmetric_chamfer_m": summary.get("symmetric_chamfer_m"),
        "window_discrete_frechet_median_m": summary.get(
            "window_discrete_frechet_median_m"
        ),
        "window_discrete_frechet_p90_m": summary.get(
            "window_discrete_frechet_p90_m"
        ),
        "prediction_precision_at_0p5m": summary.get("prediction_precision_at_0p5m"),
        "observation_coverage_at_0p5m": summary.get("observation_coverage_at_0p5m"),
        "heldout_count_weighted_rmse_m": heldout.get(
            "heldout_count_weighted_rmse_m"
        ),
        "heldout_window_p90_median_m": heldout.get(
            "heldout_window_p90_median_m"
        ),
        "result_directory": summary.get("result_directory"),
    }


def create_figure(rows: list[dict[str, object]], output: Path) -> None:
    labels = [str(row["sequence_id"]) for row in rows]
    x = np.arange(len(rows), dtype=float)
    p90 = np.array([finite_float(row.get("symmetric_p90_m")) for row in rows])
    precision = np.array(
        [finite_float(row.get("prediction_precision_at_0p5m")) for row in rows]
    )
    coverage = np.array(
        [finite_float(row.get("observation_coverage_at_0p5m")) for row in rows]
    )
    both = np.array(
        [finite_float(row.get("both_observation_fraction")) for row in rows]
    )
    continuity = np.array(
        [finite_float(row.get("continuity_pass_fraction")) for row in rows]
    )

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)
    axes[0].bar(x, p90, color="#4c78a8")
    axes[0].set_ylabel("metres")
    axes[0].set_title("Symmetric P90: fitted curve vs CLRNet/IPM observations")
    width = 0.36
    axes[1].bar(x - width / 2, precision, width, label="prediction precision @ 0.5 m")
    axes[1].bar(x + width / 2, coverage, width, label="observation coverage @ 0.5 m")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("fraction")
    axes[1].set_title("Bidirectional 0.5 m support")
    axes[1].legend()
    axes[2].bar(x - width / 2, both, width, label="both-side frame availability")
    axes[2].bar(x + width / 2, continuity, width, label="continuity pass fraction")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_ylabel("fraction")
    axes[2].set_title("Operational availability and continuity")
    axes[2].legend()
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.set_xlabel("KITTI Odometry Sequence")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("SURF fixed internal-consistency scorecard (not ground-truth accuracy)")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run(batch_summary: Path, output_dir: Path, spacing_m: float) -> dict[str, object]:
    if spacing_m <= 0:
        raise ValueError("resample-spacing-m must be positive.")
    require_new_output(output_dir)
    batch_rows = read_csv(batch_summary)
    completed = [row for row in batch_rows if row.get("status", "").startswith("complete")]
    if not completed:
        raise ValueError("Batch summary contains no completed Sequence.")

    summaries: list[dict[str, object]] = []
    all_window_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for batch_row in completed:
        sequence_id = str(batch_row.get("sequence_id", ""))
        try:
            result_directory = Path(str(batch_row["result_directory"]))
            summary, window_rows = evaluate_sequence(
                sequence_id, result_directory, spacing_m
            )
            summaries.append(summary)
            all_window_rows.extend(window_rows)
        except Exception as exc:
            failures.append({"sequence_id": sequence_id, "error": str(exc)})

    if not summaries:
        raise RuntimeError("No Sequence could be evaluated from the completed batch.")
    summaries.sort(key=lambda row: str(row["sequence_id"]))
    summary_rows = [summary_csv_row(summary) for summary in summaries]
    write_csv(output_dir / "FIXED_METRICS_BY_SEQUENCE.csv", summary_rows)
    write_csv(output_dir / "FIXED_METRICS_BY_WINDOW_SIDE.csv", all_window_rows)
    if failures:
        write_csv(output_dir / "FAILED_SEQUENCES.csv", failures)
    (output_dir / "FIXED_METRICS.json").write_text(
        json.dumps(
            json_safe({
                "status": "complete" if not failures else "complete_with_failures",
                "metric_scope": (
                    "internal consistency and operational feasibility; no official KITTI lane ground truth"
                ),
                "detector_accuracy_status": (
                    "not measured locally because no CULane or equivalent annotated lane benchmark was found"
                ),
                "resample_spacing_m": spacing_m,
                "thresholds_m": list(THRESHOLDS_M),
                "sequence_summaries": summaries,
                "failures": failures,
                "claims_not_allowed": [
                    "Do not call these values CLRNet detection accuracy.",
                    "Do not call CLRNet/IPM observations lane ground truth.",
                    "Do not compare methods without also reporting bidirectional coverage.",
                ],
            }),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    create_figure(summary_rows, output_dir / "FIXED_METRICS_SCORECARD.png")
    status = {
        "status": "complete" if not failures else "complete_with_failures",
        "evaluated_sequence_count": len(summaries),
        "failed_sequence_count": len(failures),
        "previous_outputs_modified": False,
        "output_dir": str(output_dir),
        "primary_internal_metrics": [
            "symmetric_p90_m",
            "observation_coverage_at_0p5m",
            "prediction_precision_at_0p5m",
        ],
        "accuracy_claim": "not official lane-detection accuracy",
    }
    (output_dir / "STATUS.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return status


def main() -> None:
    args = parse_args()
    result = run(args.batch_summary, args.output_dir, args.resample_spacing_m)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
