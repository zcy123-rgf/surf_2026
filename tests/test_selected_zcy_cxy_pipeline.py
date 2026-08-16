from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import analyze_lane_identity_modes as identity
from scripts import evaluate_semantickitti_curve_reference as semantic


def lane(x: float) -> np.ndarray:
    z = np.linspace(3.0, 20.0, 12)
    return np.column_stack([np.full_like(z, x), z])


def record(frame_id: int, left: float, right: float) -> dict[str, object]:
    return {
        "frame_id": frame_id,
        "candidate_count": 3,
        "eligible_for_two_curve_fit": True,
        "selected_lanes_local_ground_xz_m": [lane(left).tolist(), lane(right).tolist()],
    }


def test_identity_continuity_detects_discontinuous_frame() -> None:
    poses = [np.eye(4) for _ in range(3)]
    stable = {index: record(index, -2.0, 2.0) for index in range(3)}
    switched = dict(stable)
    switched[2] = record(2, -5.0, 5.0)
    stable_rows = identity.continuity_rows("temporal_ego", stable, poses, 1.65, 0.0, 1.5)
    switched_rows = identity.continuity_rows("ego_adjacent", switched, poses, 1.65, 0.0, 1.5)
    assert stable_rows[-1]["mean_continuity_m"] < 1e-9
    assert switched_rows[-1]["continuity_gate_failed"] is True


def test_semantic_reference_assignment_is_one_side_only() -> None:
    anchors = {"left": lane(-2.0), "right": lane(2.0)}
    reference = np.vstack([lane(-2.05)[::3], lane(2.05)[::3], lane(8.0)[::3]])
    assigned, rows, summary = semantic.associate_reference(reference, anchors, 0.5, 0.1)
    assert len(assigned["left"]) == 4
    assert len(assigned["right"]) == 4
    assert summary["unassigned"] == 4
    assert all(row["assigned_side"] in {"left", "right", "unassigned"} for row in rows)


def test_semantic_metric_is_small_for_matching_curve() -> None:
    references = {"left": lane(-2.0), "right": lane(2.0)}
    rows, distances = semantic.evaluate_method(
        "matching", {"left": lane(-2.01), "right": lane(2.01)}, references, 3.0
    )
    assert len(rows) == 2
    assert max(row["reference_to_prediction_rmse_m"] for row in rows) < 0.02
    assert max(np.max(value) for value in distances.values()) < 0.02


def test_semantic_loader_uses_raw_lower_16_bit_id_and_exact_frame(tmp_path: Path) -> None:
    velodyne = tmp_path / "velodyne"
    labels = tmp_path / "labels"
    velodyne.mkdir()
    labels.mkdir()
    scan = np.asarray(
        [[-2.0, 1.65, 10.0, 0.5], [2.0, 1.65, 12.0, 0.5], [0.0, 0.0, 8.0, 0.5]],
        dtype=np.float32,
    )
    scan.tofile(velodyne / "000000.bin")
    np.asarray([60, (7 << 16) | 60, 40], dtype=np.uint32).tofile(
        labels / "000000.label"
    )
    calibration = {
        "image_from_velodyne": np.eye(4),
    }
    points, audit = semantic.semantic_reference_points(
        velodyne, labels, [np.eye(4)], calibration, 0, 0, 0, 60,
        (-20.0, 20.0), (3.0, 50.0),
    )
    assert points.shape == (2, 2)
    assert audit[0]["raw_class_60_points"] == 2
    assert np.allclose(points, [[-2.0, 10.0], [2.0, 12.0]])


def test_fixed_selection_script_uses_exact_reviewed_range(tmp_path: Path) -> None:
    scan_dir = tmp_path / "scan"
    scan_dir.mkdir()
    (scan_dir / "selected_lane_points.json").write_text("{}", encoding="utf-8")
    counts = [
        {"frame_id": frame_id, "eligible_for_two_curve_fit": frame_id != 3}
        for frame_id in range(10)
    ]
    (scan_dir / "scan.json").write_text(
        json.dumps({"counts": counts}), encoding="utf-8"
    )
    output = tmp_path / "selection.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "create_fixed_window_selection.py"),
            "--scan-json", str(scan_dir / "scan.json"),
            "--start-frame", "0",
            "--end-frame", "9",
            "--dataset-name", "synthetic",
            "--minimum-valid-frames", "9",
            "--output", str(output),
        ],
        check=True,
    )
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["status"] == "selected"
    assert document["selected"]["start_frame"] == 0
    assert document["selected"]["end_frame"] == 9
    assert document["selected"]["valid_frame_count"] == 9


def test_windows_runner_contains_both_independent_tasks() -> None:
    text = (ROOT / "scripts" / "run_selected_zcy_cxy_windows.ps1").read_text(
        encoding="utf-8-sig"
    )
    assert "ego_adjacent" in text
    assert "temporal_ego" in text
    assert "evaluate_semantickitti_curve_reference.py" in text
    assert "previous_outputs_modified" in text
    assert "| Out-Host" in text
    assert (ROOT / "scripts" / "finalize_selected_zcy_cxy_run.ps1").is_file()
