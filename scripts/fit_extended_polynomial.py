"""Fit only robust Frenet polynomials on one ranked extended-range option."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import extended_curve_model_common as common  # noqa: E402
from scripts import run_weekly_lane_hierarchy as weekly  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_arguments(parser)
    parser.add_argument(
        "--degrees",
        default="1,2,3",
        help="Comma-separated polynomial degrees compared by held-out frames.",
    )
    return parser.parse_args()


def parse_degrees(value: str) -> list[int]:
    degrees = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not degrees or len(set(degrees)) != len(degrees):
        raise ValueError("--degrees must contain unique integer degrees.")
    if any(degree < 1 or degree > 5 for degree in degrees):
        raise ValueError("Polynomial degrees must be between one and five.")
    return degrees


def plot_curves(
    path: Path,
    data: common.ExtendedCurveInput,
    curves: dict[str, np.ndarray],
) -> None:
    colors = {"left": "#1f77b4", "right": "#d62728"}
    figure, axis = plt.subplots(figsize=(7.2, 9.0))
    for side_index, side in enumerate(weekly.SIDES):
        raw = np.vstack(
            [np.asarray(frame["lanes_xz"][side_index]) for frame in data.frames]
        )
        axis.scatter(
            raw[:, 0], raw[:, 1], s=3, alpha=0.12, color=colors[side]
        )
        axis.plot(
            curves[side][:, 0],
            curves[side][:, 1],
            linewidth=2.2,
            color=colors[side],
            label=f"{side} robust polynomial",
        )
    axis.set_xlabel("X right in selected reference frame [m]")
    axis.set_ylabel("Z forward in selected reference frame [m]")
    axis.set_title("Separate robust polynomial left/right curves")
    axis.grid(True, linewidth=0.5, alpha=0.35)
    axis.legend()
    axis.set_aspect("equal", adjustable="datalim")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    degrees = parse_degrees(args.degrees)
    data = common.load_input(args)

    model_rows = [
        weekly.polynomial_cross_validation(data.groups, degree, args)
        for degree in degrees
    ]
    best = min(model_rows, key=lambda row: float(row["mean_fold_rmse_m"]))
    best_degree = int(str(best["model"]).rsplit(" ", 1)[-1])

    curves_xz: dict[str, np.ndarray] = {}
    coefficient_document: dict[str, object] = {}
    for side in weekly.SIDES:
        aggregate = weekly.aggregate_equal_group_bins(
            weekly.grouped_side_points(data.groups, side), args.bin_size_m
        )
        coefficients = weekly.robust_polyfit(aggregate, best_degree, args)
        progress = np.linspace(
            float(aggregate[0, 0]),
            float(aggregate[-1, 0]),
            args.curve_samples,
        )
        curve_sd = np.column_stack(
            [progress, np.polyval(coefficients, progress)]
        )
        curve_xz = weekly.frenet_to_xz(curve_sd, data.reference_path)
        curves_xz[side] = curve_xz
        weekly.write_csv(
            args.output_dir / "02_curves" / f"{side}_polynomial_curve.csv",
            [
                {
                    "sample_index": index,
                    "s_path_m": sd[0],
                    "d_right_m": sd[1],
                    "x_right_reference_m": xz[0],
                    "z_forward_reference_m": xz[1],
                }
                for index, (sd, xz) in enumerate(zip(curve_sd, curve_xz))
            ],
        )
        coefficient_document[side] = {
            "degree": best_degree,
            "coefficients_high_to_low": coefficients.tolist(),
            "s_min_m": float(progress[0]),
            "s_max_m": float(progress[-1]),
            "aggregate_bin_count": len(aggregate),
        }

    weekly.write_csv(
        args.output_dir / "01_evaluation" / "polynomial_cross_validation.csv",
        model_rows,
    )
    weekly.write_csv(
        args.output_dir / "00_audit" / "frame_validity.csv",
        data.frame_audit.values(),
    )
    plot_curves(
        args.output_dir / "03_figures" / "polynomial_left_right.png",
        data,
        curves_xz,
    )

    result = {
        "status": "complete",
        "method_scope": "robust polynomial only; no B-spline",
        **common.audit_document(data, args),
        "compared_degrees": degrees,
        "selected_degree": best_degree,
        "selection_rule": "lowest observed held-out-frame mean RMSE",
        "models": model_rows,
        "coefficients": coefficient_document,
        "result_image": str(
            (args.output_dir / "03_figures" / "polynomial_left_right.png").resolve()
        ),
    }
    common.write_json(args.output_dir / "RESULT.json", result)
    common.write_json(
        args.output_dir / "STATUS.json",
        {
            "status": "complete",
            "method": "polynomial",
            "frames": [data.start_frame, data.end_frame],
            "valid_frames": len(data.frames),
            "selected_degree": best_degree,
            "previous_outputs_modified": False,
        },
    )
    print(
        f"POLYNOMIAL FINISHED: frames {data.start_frame}-{data.end_frame}, "
        f"degree {best_degree}, output {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()

