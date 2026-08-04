from __future__ import annotations

import json
from pathlib import Path

from pycharm_entrypoints import common


ROOT = Path(__file__).resolve().parents[1]


def make_pipeline_output(root: Path, name: str) -> Path:
    output = root / name
    metadata = output / "00_metadata"
    metadata.mkdir(parents=True)
    (metadata / "detected_lane_points.json").write_text("{}", encoding="utf-8")
    (metadata / "aligned_lane_points.json").write_text("{}", encoding="utf-8")
    return output


def test_direct_entrypoints_are_small_in_process_launchers() -> None:
    expected = {
        "00_check_environment.py": "check_windows_env.main",
        "01_generate_pose_aligned_points.py": "run_full_point_pipeline.main",
        "02_run_optimized_ransac.py": "run_selected_ransac_reference.main",
        "03_compare_curve_models.py": "compare_polynomial_bspline.main",
    }
    for name, target in expected.items():
        text = (ROOT / "pycharm_entrypoints" / name).read_text(encoding="utf-8")
        assert target in text
        assert "subprocess" not in text


def test_latest_pipeline_manifest_and_fallback(tmp_path, monkeypatch) -> None:
    output_root = tmp_path / "pycharm_outputs"
    manifest = output_root / "LATEST_PIPELINE.json"
    monkeypatch.setattr(common, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(common, "LATEST_PIPELINE", manifest)

    first = make_pipeline_output(output_root, "01_pipeline_first")
    files = common.record_latest_pipeline(first)
    assert files["output"] == first
    document = json.loads(manifest.read_text(encoding="utf-8"))
    assert Path(document["output"]) == first.resolve()
    assert common.latest_pipeline_files()["aligned"].is_file()

    manifest.unlink()
    assert common.latest_pipeline_files()["detected"].is_file()


def test_new_output_is_timestamped_and_not_precreated(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(common, "OUTPUT_ROOT", tmp_path)
    output = common.new_output("02_ransac")
    assert output.parent == tmp_path
    assert output.name.startswith("02_ransac_")
    assert not output.exists()
