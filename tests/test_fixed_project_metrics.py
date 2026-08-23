from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_fixed_project_metrics as metrics  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_bidirectional_metric_detects_missing_length() -> None:
    observed = np.column_stack([np.zeros(11), np.linspace(0.0, 10.0, 11)])
    predicted = np.column_stack([np.zeros(6), np.linspace(0.0, 5.0, 6)])
    result, pred_to_obs, obs_to_pred = metrics.evaluate_pair(observed, predicted, 0.5)
    assert np.max(pred_to_obs) < 1e-9
    assert np.max(obs_to_pred) >= 5.0
    assert result["prediction_precision_at_0p5m"] == 1.0
    assert result["observation_coverage_at_0p5m"] < 1.0


def test_batch_posthoc_run(tmp_path: Path) -> None:
    result_root = tmp_path / "sequence_00"
    adaptive = result_root / "03_curve_models_and_fusion"
    adaptive.mkdir(parents=True)
    observations = []
    curves = []
    for index, z_value in enumerate(np.linspace(0.0, 5.0, 11)):
        observations.append(
            {
                "window_id": 0,
                "side": "left",
                "segment_id": 0,
                "point_id": index,
                "supporting_frame_count": 1,
                "x_common_m": 0.0,
                "z_common_m": z_value,
            }
        )
        curves.append(
            {
                "window_id": 0,
                "start_frame": 0,
                "end_frame": 14,
                "side": "left",
                "segment_id": 0,
                "classification": "straight",
                "model": "parametric_polynomial",
                "sample_id": index,
                "ordering_key": z_value,
                "x_common_m": 0.1,
                "z_common_m": z_value,
            }
        )
    write_csv(adaptive / "window_aggregate_points.csv", observations)
    write_csv(adaptive / "window_curve_samples.csv", curves)
    write_csv(
        adaptive / "model_comparison.csv",
        [
            {
                "selected_for_output": "True",
                "heldout_count": 10,
                "heldout_mean_m": 0.1,
                "heldout_rmse_m": 0.1,
                "heldout_p90_m": 0.1,
            }
        ],
    )
    (result_root / "FINAL_METRICS.json").write_text(
        json.dumps(
            {
                "observation_coverage": {"both_fraction": 0.8},
                "fitting_coverage": {"completed_fraction": 1.0},
                "continuity": {"passed_fraction": 1.0},
            }
        ),
        encoding="utf-8",
    )
    batch = tmp_path / "BATCH_SUMMARY.csv"
    write_csv(
        batch,
        [
            {
                "sequence_id": "00",
                "status": "complete_with_skips",
                "result_directory": str(result_root),
            }
        ],
    )
    output = tmp_path / "fixed"
    status = metrics.run(batch, output, 0.5)
    assert status["evaluated_sequence_count"] == 1
    with (output / "FIXED_METRICS_BY_SEQUENCE.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        summary = list(csv.DictReader(handle))[0]
    assert abs(float(summary["symmetric_p90_m"]) - 0.1) < 1e-9
    assert float(summary["observation_coverage_at_0p5m"]) == 1.0
    assert (output / "FIXED_METRICS_SCORECARD.png").is_file()
