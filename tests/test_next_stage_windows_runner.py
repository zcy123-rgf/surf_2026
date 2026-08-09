from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_next_stage_lane_experiments_windows.ps1"


def test_next_stage_runner_preserves_old_outputs_and_uses_registered_methods() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "previous_outputs_modified = $false" in text
    assert "run_weekly_lane_hierarchy.py" in text
    assert '"--feature-count-grid", "4,6,8,12"' in text
    assert "Sequence 09" in text
    assert "annotation_template.json" in text
    assert "not official KITTI ground truth" in text


def test_next_stage_runner_uses_new_output_and_independent_annotations() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "next_stage_lane_$Stamp" in text
    assert "OutputRoot must be new or empty" in text
    assert "Do not copy CLRNet points into the annotation" in text
    assert "NEXT_STAGE_RESULT_BUNDLE.zip" in text
    assert "sequence09_manual_annotation_package.zip" in text
