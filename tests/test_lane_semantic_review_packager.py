from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_semantic_review_packager_is_read_only_and_samples_expected_ranges() -> None:
    text = (
        ROOT / "scripts" / "package_lane_semantic_review_windows.ps1"
    ).read_text(encoding="utf-8")
    assert "lane_semantic_review_$Stamp" in text
    assert "previous_outputs_modified = $false" in text
    assert "negative_control_000000_000104" in text
    assert "candidate_000143_000164" in text
    assert "candidate_001549_001578" in text
    assert "candidate_001588_001608" in text
    assert "lane_semantic_review_package.zip" in text
    assert "Copy-Item" in text
    assert "Remove-Item" not in text
