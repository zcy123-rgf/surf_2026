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
    assert '"SURF_FINAL_DECISIONS_ZH.md"' in creator


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
        "not real-world lane accuracy",
    ):
        assert token in runner
