import json
from argparse import Namespace
from pathlib import Path

from scripts import summarize_curved_improvement as summary


def write(path: Path, value: dict[str, object]) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def mode(name: str, valid: int, mean: float, maximum: float) -> dict[str, object]:
    return {
        "method": name,
        "valid_two_lane_frames": valid,
        "valid_two_lane_rate": valid / 100.0,
        "longest_consecutive_valid_run": valid,
        "continuity_mean_m": mean,
        "continuity_p90_m": mean * 1.2,
        "continuity_max_m": maximum,
        "continuity_gate_failure_count": 0,
    }


def test_summary_records_coverage_and_coupled_rmse_changes(tmp_path: Path) -> None:
    old_identity = write(
        tmp_path / "old.json",
        {"mode_summaries": [mode("ego_adjacent", 43, 0.16, 1.8), mode("temporal_ego", 39, 0.11, 0.20)]},
    )
    joint_identity = write(
        tmp_path / "joint.json",
        {"mode_summaries": [mode("ego_adjacent", 43, 0.16, 1.8), mode("temporal_joint", 48, 0.12, 0.24)]},
    )
    polynomial = write(
        tmp_path / "poly.json",
        {
            "selected_degree": 3,
            "models": [
                {
                    "model": "robust Frenet polynomial d(s), degree 3",
                    "mean_fold_rmse_m": 1.0,
                    "fold_rmse_standard_error_m": 0.1,
                }
            ],
        },
    )
    independent = write(
        tmp_path / "independent.json",
        {
            "cross_validation": [
                {
                    "selected_by_one_standard_error_rule": True,
                    "mean_fold_rmse_m": 0.30,
                    "fold_rmse_standard_error_m": 0.03,
                }
            ]
        },
    )
    coupled = write(
        tmp_path / "coupled.json",
        {
            "cross_validation": [
                {
                    "selected_by_one_standard_error_rule": True,
                    "mean_fold_rmse_m": 0.27,
                    "fold_rmse_standard_error_m": 0.02,
                }
            ]
        },
    )
    output = tmp_path / "output"
    result = summary.run(
        Namespace(
            old_identity_result=old_identity,
            joint_identity_result=joint_identity,
            polynomial_result=polynomial,
            bspline_result=independent,
            coupled_result=coupled,
            output_dir=output,
        )
    )

    assert result["registered_decisions"]["joint_valid_frame_change"] == 9
    assert result["registered_decisions"]["coupled_minus_independent_mean_fold_rmse_m"] < 0
    assert (output / "identity_comparison.csv").is_file()
    assert (output / "curve_model_comparison.csv").is_file()
    assert (output / "improvement_summary.png").is_file()
    assert (output / "RESULT.json").is_file()
