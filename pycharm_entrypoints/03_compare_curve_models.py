"""Compare robust polynomials and cubic B-splines on the latest aligned points."""

from __future__ import annotations

from common import latest_pipeline_files, new_output, run_main
from scripts import compare_polynomial_bspline


if __name__ == "__main__":
    files = latest_pipeline_files()
    output = new_output("03_curve_models")
    run_main(
        compare_polynomial_bspline.main,
        [
            "--aligned-json",
            str(files["aligned"]),
            "--output-dir",
            str(output),
        ],
    )
    print("Curve-model output:", output)
