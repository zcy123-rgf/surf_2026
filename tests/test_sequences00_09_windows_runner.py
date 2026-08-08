from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_batch_runner_lists_sequences_00_through_09_and_keeps_outputs_separate():
    text = (
        ROOT / "scripts" / "run_sequences00_09_curve_models_windows.ps1"
    ).read_text(encoding="utf-8")
    for sequence_id in range(10):
        assert f'"{sequence_id:02d}"' in text
    assert '"sequence_$SequenceId"' in text
    assert "BATCH_SUMMARY.csv" in text
    assert "DATASET_INVENTORY.csv" in text
    assert "total_dataset_frames" in text


def test_single_sequence_runner_uses_sequence_specific_inputs_and_metadata():
    text = (
        ROOT / "scripts" / "run_single_odometry_sequence_curve_windows.ps1"
    ).read_text(encoding="utf-8")
    assert "Resolve-KittiOdometryWorkstationInput" in text
    assert '"--dataset-name", $DatasetName' in text
    assert '"--poses", $Kitti.Poses' in text
    assert '"--calib", $Kitti.Calib' in text
    assert '"--ranking-mode", $RankingMode' in text
    assert '"sustained_curve"' in text
    assert "previous_outputs_modified = $false" in text


def test_release_copy_contains_multi_sequence_entrypoints():
    text = (ROOT / "scripts" / "create_clean_workstation_copy.ps1").read_text(
        encoding="utf-8"
    )
    assert '"kitti_odometry_workstation_input.ps1"' in text
    assert '"run_single_odometry_sequence_curve_windows.ps1"' in text
    assert '"run_sequences00_09_curve_models_windows.ps1"' in text
    assert '"README_SEQUENCES00_09_WINDOWS_ZH.md"' in text
