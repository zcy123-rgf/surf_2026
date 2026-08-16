import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from scripts import diagnose_curved_scan_failures as diagnosis
from scripts import fit_piecewise_bspline as piecewise


def groups(count: int = 10) -> list[dict[str, object]]:
    result = []
    for frame_id in range(count):
        s = np.linspace(frame_id, frame_id + 30.0, 60)
        centre = 0.4 * np.sin(s / 10.0)
        width = 3.6 + 0.1 * np.cos(s / 18.0)
        result.append(
            {
                "group_id": frame_id,
                "lanes_sd": [
                    np.column_stack([s, centre - width / 2.0]),
                    np.column_stack([s, centre + width / 2.0]),
                ],
            }
        )
    return result


def settings() -> Namespace:
    return Namespace(
        bin_size_m=0.5,
        huber_delta_m=0.2,
        irls_iterations=4,
        minimum_local_bins=8,
        maximum_cv_folds=5,
    )


def test_piecewise_pair_has_overlap_support_and_no_crossing() -> None:
    fitted = piecewise.fit_piecewise_pair(groups(), 20.0, 10.0, 0.01, settings())
    check = piecewise.pair_width_check(fitted)

    assert fitted.completed_windows >= 2
    assert check["supported_fraction"] > 0.95
    assert check["curves_cross"] is False
    assert 3.0 < check["median_right_minus_left_m"] < 4.2


def test_piecewise_candidate_reports_heldout_support() -> None:
    row = piecewise.evaluate_candidate(groups(), 20.0, 10.0, 0.01, settings())

    assert row["fold_count"] > 0
    assert row["heldout_supported_point_fraction"] > 0.8
    assert row["mean_fold_rmse_m"] >= 0.0
    assert row["curves_cross"] is False


def write_scan(path: Path, reasons: list[tuple[int, str, bool, bool]]) -> None:
    rows = []
    for frame_id, reason, pair_gate, metric_gate in reasons:
        rows.append(
            {
                "frame_id": frame_id,
                "candidate_count": 1 if reason == "below_minimum_candidate_count" else 2,
                "pair_selection_reason": reason,
                "pair_selection_gate": pair_gate,
                "metric_bev_point_gate": metric_gate,
                "selected_projected_point_counts": "4;4" if metric_gate else "2;4",
            }
        )
    path.write_text(json.dumps({"counts": rows}), encoding="utf-8")


def test_scan_diagnosis_separates_detection_pairing_and_ipm_failures(
    tmp_path: Path,
) -> None:
    old = tmp_path / "old.json"
    joint = tmp_path / "joint.json"
    rows = [
        (0, "below_minimum_candidate_count", False, False),
        (1, "missing_metric_candidate_on_one_side", False, False),
        (2, "matched", True, False),
        (3, "matched", True, True),
    ]
    write_scan(old, rows)
    write_scan(joint, rows)
    output = tmp_path / "output"
    result = diagnosis.run(
        Namespace(
            scan=[f"temporal_ego,{old}", f"temporal_joint,{joint}"],
            output_dir=output,
        )
    )
    summary = result["summary"][0]
    assert summary["fewer_than_two_clrnet_candidates"] == 1
    assert summary["image_side_pairing_failed"] == 1
    assert summary["selected_pair_has_too_few_ipm_points"] == 1
    assert summary["valid_two_lane_metric_frames"] == 1
    assert result["comparisons"]["temporal_joint"]["valid_frame_change"] == 0
