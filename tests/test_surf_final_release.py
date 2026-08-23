import csv
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_sequence01_release_entrypoint_is_registered() -> None:
    runner = (ROOT / "scripts" / "run_surf_final_sequence01_windows.ps1").read_text(
        encoding="utf-8"
    )
    assert '-SequenceId "01"' in runner
    assert "-StartFrame 851" in runner
    assert "-EndFrame 1005" in runner
    assert '-StraightSeedRanges "851-875,991-1005"' in runner
    assert "Sequence 03" in runner and "Sequence 07" in runner
    assert "previous_outputs_modified = $false" in runner

    creator = (ROOT / "scripts" / "create_surf_final_workspace.ps1").read_text(
        encoding="utf-8"
    )
    assert '"scripts\\run_surf_final_sequence01_windows.ps1"' in creator
    assert '"scripts\\run_surf_final_sequence01_full_windows.ps1"' in creator
    assert '"scripts\\run_surf_final_all_sequences_windows.ps1"' in creator
    assert '"scripts\\summarize_surf_final_all_sequences.py"' in creator
    assert '"scripts\\evaluate_fixed_project_metrics.py"' in creator
    assert '"scripts\\run_fixed_project_metrics_windows.ps1"' in creator
    assert '"scripts\\audit_surf_final_workspace.ps1"' in creator
    assert '"SURF_FINAL_DECISIONS_ZH.md"' in creator
    assert '"FIXED_EVALUATION_PROTOCOL_ZH.md"' in creator
    assert '"FINAL_WORKSPACE_INVENTORY_ZH.md"' in creator
    assert '"README_SURF_FINAL_ZH.md"' in creator
    assert '"surf_bev\\detectors.py"' in creator
    assert '"surf_bev\\lane_fusion_weighted.py"' not in creator


def test_workspace_audit_is_non_destructive_and_checks_final_artifacts() -> None:
    audit = (ROOT / "scripts" / "audit_surf_final_workspace.ps1").read_text(
        encoding="utf-8"
    )
    for token in (
        "audit_is_read_only_except_new_report_and_optional_backup",
        "deletion_performed = $false",
        "CLRNet\\weights\\culane_r18.pth",
        "surf_final_all_sequences_*",
        "2.0-domain-aligned",
        "fixed_project_metrics_*",
        "ready_for_final_retention",
    ):
        assert token in audit
    for destructive in ("Remove-Item", "Move-Item", "Clear-Content"):
        assert destructive not in audit


def test_final_window_exports_auditable_metrics() -> None:
    runner = (ROOT / "scripts" / "run_surf_final_window_windows.ps1").read_text(
        encoding="utf-8"
    )
    for token in (
        "FINAL_METRICS.json",
        "observation_coverage",
        "fitting_coverage",
        "selected_model_counts",
        "continuity",
        "bridge_audit",
        "blended_lane_nodes.csv",
        "final_lane_nodes_with_hypotheses.csv",
        "not real-world lane accuracy",
    ):
        assert token in runner


def test_full_sequence01_scale_runner_keeps_registered_policy() -> None:
    runner = (
        ROOT / "scripts" / "run_surf_final_sequence01_full_windows.ps1"
    ).read_text(encoding="utf-8")
    assert '-SequenceId "01"' in runner
    assert "-StartFrame 0" in runner
    assert "-EndFrame 1100" in runner
    assert '-StraightSeedRanges "851-875,991-1005"' in runner
    assert "-WindowLength 15" in runner
    assert "-WindowStride 10" in runner
    assert "official lane-accuracy" in runner
    assert "previous_outputs_modified = $false" in runner


def test_all_sequences_runner_is_resumable_and_auditable() -> None:
    runner = (
        ROOT / "scripts" / "run_surf_final_all_sequences_windows.ps1"
    ).read_text(encoding="utf-8")
    for token in (
        '"00,01,02,03,04,05,06,07,08,09,10"',
        "reused_exact_completed_result",
        "BATCH_SUMMARY.csv",
        "BATCH_STATUS.json",
        "ForceRerun",
        "StopOnFailure",
        "threshold_baseline",
        "not official lane-position accuracy",
        "surf_final_all_sequences_review_",
        "summarize_surf_final_all_sequences.py",
    ):
        assert token in runner


def test_all_sequences_summary_is_honest_about_metric_scope() -> None:
    summarizer = (
        ROOT / "scripts" / "summarize_surf_final_all_sequences.py"
    ).read_text(encoding="utf-8")
    for token in (
        "ALL_SEQUENCES_SUMMARY.png",
        "ALL_SEQUENCES_AGGREGATE.json",
        "frame_weighted_observation_fraction",
        "selected_model_totals",
        "not official lane-position accuracy",
    ):
        assert token in summarizer


def test_all_sequences_summary_writes_plot_and_aggregate(tmp_path: Path) -> None:
    module_path = ROOT / "scripts" / "summarize_surf_final_all_sequences.py"
    spec = importlib.util.spec_from_file_location("surf_batch_summary", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fields = (
        "sequence_id",
        "status",
        "frame_count",
        "left_observation_fraction",
        "right_observation_fraction",
        "both_observation_fraction",
        "completed_fit_fraction",
        "continuity_pass_fraction",
        "straight_polynomial_fits",
        "transition_bspline_fits",
        "curve_polynomial_fits",
        "curve_bspline_fits",
        "accepted_low_confidence_bridges",
    )
    summary = tmp_path / "BATCH_SUMMARY.csv"
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "sequence_id": "01",
                "status": "complete_with_skips",
                "frame_count": "1101",
                "left_observation_fraction": "0.47",
                "right_observation_fraction": "0.91",
                "both_observation_fraction": "0.41",
                "completed_fit_fraction": "0.67",
                "continuity_pass_fraction": "0.86",
                "straight_polynomial_fits": "97",
                "transition_bspline_fits": "15",
                "curve_polynomial_fits": "2",
                "curve_bspline_fits": "33",
                "accepted_low_confidence_bridges": "6",
            }
        )
    output = tmp_path / "review"
    aggregate = module.run(summary, output)
    assert aggregate["completed_sequence_frames"] == 1101
    assert (output / "ALL_SEQUENCES_SUMMARY.png").is_file()
    assert (output / "ALL_SEQUENCES_AGGREGATE.json").is_file()
