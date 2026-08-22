from __future__ import annotations

import argparse
import json

import numpy as np

from scripts import compare_polynomial_bspline as comparison


def test_focused_model_comparison_writes_no_hierarchy_outputs(tmp_path) -> None:
    rng = np.random.default_rng(20260803)
    frames = []
    for frame_id in range(5):
        z = np.linspace(3.0, 45.0, 58)
        center = 0.002 * (z - 22.0) ** 2
        frames.append(
            {
                "frame_id": frame_id,
                "lanes_reference_xz_m": [
                    np.column_stack(
                        [center - 1.8 + rng.normal(0, 0.03, len(z)), z]
                    ).tolist(),
                    np.column_stack(
                        [center + 1.8 + rng.normal(0, 0.03, len(z)), z]
                    ).tolist(),
                ],
            }
        )
    source = tmp_path / "aligned.json"
    source.write_text(json.dumps({"frames": frames}), encoding="utf-8")
    output = tmp_path / "output"
    result = comparison.run(
        argparse.Namespace(
            aligned_json=source,
            output_dir=output,
            bin_size_m=0.5,
            smoothing_grid="0.0025,0.01,0.04",
            huber_delta_m=0.2,
            irls_iterations=4,
            curve_samples=220,
        )
    )
    assert result["status"] == "complete"
    assert len(result["models"]) == 4
    assert (output / "model_comparison.png").is_file()
    assert (output / "bspline_left_right_curves.png").is_file()
    assert (output / "model_lofo.csv").is_file()
    assert not (output / "03_hierarchical_refusion").exists()
