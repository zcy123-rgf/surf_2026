"""Optimize side-aware polynomial RANSAC without changing the denoising family.

The five-frame KITTI sample has no lane-line ground truth.  This experiment
therefore separates:

1. safety checks on untouched real and manual pseudo-label point sets;
2. parameter tuning on controlled, labelled outliers injected into the real
   pose-aligned geometry;
3. held-out validation repeats with different random seeds;
4. a final stability audit with another disjoint seed range.

The production pipeline is not modified by this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.evaluate_denoise_methods as base  # noqa: E402
import scripts.optimize_temporal_denoise as injection  # noqa: E402


DEFAULT_ANNOTATION = (
    ROOT / "annotations" / "kitti00_first5_manual_annotations.json"
)

PHASE1_SEED_OFFSET = 100_000
PHASE2_SEED_OFFSET = 200_000
FINAL_SEED_OFFSET = 300_000
SAFETY_SEED_OFFSET = 400_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune side-aware polynomial RANSAC parameters and constraints."
    )
    parser.add_argument(
        "--clrnet-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--manual-json",
        type=Path,
        default=DEFAULT_ANNOTATION,
    )
    parser.add_argument(
        "--calib", type=Path, required=True
    )
    parser.add_argument(
        "--poses",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "workstation_outputs"
        / "kitti00_first5_ransac_optimized_20260727_full",
    )
    parser.add_argument("--phase1-repeats", type=int, default=8)
    parser.add_argument("--phase2-repeats", type=int, default=20)
    parser.add_argument("--final-repeats", type=int, default=50)
    parser.add_argument("--phase1-top", type=int, default=8)
    return parser.parse_args()


def load_points(args: argparse.Namespace):
    required = [args.clrnet_json, args.manual_json, args.calib, args.poses]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))
    calibration = base.parse_projection(args.calib)
    poses_cam0 = base.load_kitti_poses(args.poses)
    poses = [pose @ calibration["cam0_from_image"] for pose in poses_cam0]
    clrnet_lanes = base.load_image_lanes(args.clrnet_json)
    manual_lanes = base.load_manual_lanes(args.manual_json)
    _, points, ranges = base.align_image_lanes(clrnet_lanes, calibration["K"], poses)
    _, manual_points, manual_ranges = base.align_image_lanes(
        manual_lanes, calibration["K"], poses
    )
    return points, ranges, manual_points, manual_ranges


def config_id(config: dict[str, object]) -> str:
    return (
        f"d{int(config['degree'])}_"
        f"b{float(config['base_threshold']):.3f}_"
        f"s{float(config['distance_slope']):.3f}_"
        f"c{float(config['threshold_cap']):.2f}_"
        f"i{int(config['iterations'])}_"
        f"z{float(config['minimum_sample_z_span']):.1f}_"
        f"q{str(config['score_mode'])}_"
        f"r{int(config['local_refinements'])}"
    )


def threshold_profiles() -> list[dict[str, float]]:
    fixed = [
        {
            "base_threshold": threshold,
            "distance_slope": 0.0,
            "threshold_cap": threshold,
        }
        for threshold in (0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
    ]
    adaptive = [
        {
            "base_threshold": base_threshold,
            "distance_slope": distance_slope,
            "threshold_cap": threshold_cap,
        }
        for base_threshold, distance_slope, threshold_cap in itertools.product(
            (0.15, 0.20, 0.25, 0.30),
            (0.005, 0.010, 0.015),
            (0.50, 0.80, 1.00),
        )
        if threshold_cap > base_threshold
    ]
    return fixed + adaptive


def phase1_grid() -> list[dict[str, object]]:
    output = []
    for degree, profile in itertools.product((1, 2, 3), threshold_profiles()):
        output.append(
            {
                "degree": degree,
                **profile,
                "iterations": 500,
                "minimum_sample_z_span": 3.0,
                "score_mode": "count",
                "local_refinements": 1,
            }
        )
    return output


def phase2_grid(
    top_phase1: list[dict[str, object]],
) -> list[dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for row in top_phase1:
        core = {
            "degree": int(row["degree"]),
            "base_threshold": float(row["base_threshold"]),
            "distance_slope": float(row["distance_slope"]),
            "threshold_cap": float(row["threshold_cap"]),
        }
        for iterations, span, score_mode, refinements in itertools.product(
            (200, 500, 1000),
            (1.0, 5.0, 10.0),
            ("count", "balanced"),
            (1, 2),
        ):
            config = {
                **core,
                "iterations": iterations,
                "minimum_sample_z_span": span,
                "score_mode": score_mode,
                "local_refinements": refinements,
            }
            output[config_id(config)] = config
    return list(output.values())


def polynomial_design_normalized(
    z: np.ndarray, degree: int, center: float, scale: float
) -> np.ndarray:
    normalized = (np.asarray(z, dtype=np.float64) - center) / scale
    return np.vander(normalized, N=degree + 1, increasing=True)


class RansacEvaluator:
    """Caches random hypotheses for one point set and one RANSAC seed."""

    def __init__(
        self,
        points: np.ndarray,
        ranges: list[dict[str, object]],
        seed: int,
    ):
        self.points = np.asarray(points, dtype=np.float64)
        self.ranges = ranges
        self.seed = int(seed)
        self.side_indexes = {
            side: np.concatenate(
                [
                    np.arange(int(item["start"]), int(item["end"]))
                    for item in ranges
                    if item["side"] == side
                ]
            )
            for side in ("left", "right")
        }
        self._bank: dict[tuple[str, int, int, float], dict[str, object]] = {}

    def hypothesis_bank(
        self,
        side: str,
        degree: int,
        iterations: int,
        minimum_z_span: float,
    ) -> dict[str, object]:
        key = (side, degree, iterations, minimum_z_span)
        if key in self._bank:
            return self._bank[key]
        indexes = self.side_indexes[side]
        subset = self.points[indexes]
        z = subset[:, 1]
        center = float(np.median(z))
        scale = max(float(np.ptp(z)) / 2.0, 1.0)
        design = polynomial_design_normalized(z, degree, center, scale)
        sample_size = degree + 1
        side_offset = 0 if side == "left" else 10_000_019
        rng = np.random.default_rng(
            self.seed
            + side_offset
            + degree * 1_000_003
            + iterations * 101
            + int(minimum_z_span * 1000)
        )
        coefficients = []
        attempts = 0
        maximum_attempts = iterations * 30
        while len(coefficients) < iterations and attempts < maximum_attempts:
            attempts += 1
            selected = rng.choice(len(subset), sample_size, replace=False)
            if float(np.ptp(z[selected])) < minimum_z_span:
                continue
            matrix = design[selected]
            if np.linalg.matrix_rank(matrix) < sample_size:
                continue
            candidate = np.linalg.lstsq(matrix, subset[selected, 0], rcond=None)[0]
            if np.isfinite(candidate).all():
                coefficients.append(candidate)
        if not coefficients:
            coefficients.append(np.linalg.lstsq(design, subset[:, 0], rcond=None)[0])
        coefficients_array = np.asarray(coefficients, dtype=np.float64)
        residuals = np.abs(
            subset[:, 0, None] - design @ coefficients_array.T
        )
        bank = {
            "indexes": indexes,
            "subset": subset,
            "z": z,
            "center": center,
            "scale": scale,
            "design": design,
            "coefficients": coefficients_array,
            "residuals": residuals,
            "accepted_hypotheses": len(coefficients_array),
            "attempts": attempts,
        }
        self._bank[key] = bank
        return bank

    @staticmethod
    def threshold_vector(
        z: np.ndarray, config: dict[str, object]
    ) -> np.ndarray:
        base_threshold = float(config["base_threshold"])
        slope = float(config["distance_slope"])
        cap = float(config["threshold_cap"])
        threshold = base_threshold + slope * np.maximum(z - 3.0, 0.0)
        return np.minimum(threshold, cap)

    @staticmethod
    def score_hypotheses(
        residuals: np.ndarray,
        inliers: np.ndarray,
        z: np.ndarray,
        mode: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        counts = np.count_nonzero(inliers, axis=0)
        count_ratio = counts / max(len(z), 1)
        if mode == "balanced":
            ratios = []
            for low, high in zip((3, 10, 20, 30, 40), (10, 20, 30, 40, 50.1)):
                selected = (z >= low) & (z < high)
                if np.count_nonzero(selected) >= 3:
                    ratios.append(np.mean(inliers[selected], axis=0))
            balance = np.mean(ratios, axis=0) if ratios else count_ratio
            scores = 0.70 * count_ratio + 0.30 * balance
        else:
            scores = count_ratio
        error = np.sum(np.where(inliers, residuals, 0.0), axis=0) / np.maximum(
            counts, 1
        )
        return scores, error

    def apply_side(
        self, side: str, config: dict[str, object]
    ) -> tuple[np.ndarray, dict[str, object]]:
        degree = int(config["degree"])
        bank = self.hypothesis_bank(
            side,
            degree,
            int(config["iterations"]),
            float(config["minimum_sample_z_span"]),
        )
        residuals = np.asarray(bank["residuals"])
        z = np.asarray(bank["z"])
        threshold = self.threshold_vector(z, config)
        inliers = residuals <= threshold[:, None]
        scores, errors = self.score_hypotheses(
            residuals, inliers, z, str(config["score_mode"])
        )
        order = np.lexsort((errors, -scores))
        best_index = int(order[0])
        best_coefficients = np.asarray(bank["coefficients"])[best_index]
        best_mask = inliers[:, best_index].copy()
        best_score = float(scores[best_index])
        design = np.asarray(bank["design"])
        for _ in range(int(config["local_refinements"])):
            if np.count_nonzero(best_mask) < degree + 3:
                break
            coefficients = np.linalg.lstsq(
                design[best_mask],
                np.asarray(bank["subset"])[best_mask, 0],
                rcond=None,
            )[0]
            refit_residual = np.abs(
                np.asarray(bank["subset"])[:, 0] - design @ coefficients
            )
            refit_mask = refit_residual <= threshold
            refit_score, refit_error = self.score_hypotheses(
                refit_residual[:, None],
                refit_mask[:, None],
                z,
                str(config["score_mode"]),
            )
            current_residual = np.abs(
                np.asarray(bank["subset"])[:, 0] - design @ best_coefficients
            )
            current_error = float(
                np.mean(current_residual[best_mask])
                if np.any(best_mask)
                else math.inf
            )
            if float(refit_score[0]) > best_score or (
                np.isclose(float(refit_score[0]), best_score)
                and float(refit_error[0]) <= current_error
            ):
                best_coefficients = coefficients
                best_mask = refit_mask
                best_score = float(refit_score[0])
            else:
                break
        final_residual = np.abs(
            np.asarray(bank["subset"])[:, 0] - design @ best_coefficients
        )
        return best_mask, {
            "side": side,
            "total": int(len(best_mask)),
            "kept": int(np.count_nonzero(best_mask)),
            "accepted_hypotheses": int(bank["accepted_hypotheses"]),
            "hypothesis_attempts": int(bank["attempts"]),
            "normalized_z_center": float(bank["center"]),
            "normalized_z_scale": float(bank["scale"]),
            "coefficients_normalized_low_to_high": best_coefficients.tolist(),
            "median_abs_residual_m": float(np.median(final_residual)),
            "median_threshold_m": float(np.median(threshold)),
            "maximum_threshold_m": float(np.max(threshold)),
        }

    def apply(
        self, config: dict[str, object]
    ) -> tuple[np.ndarray, dict[str, object]]:
        final = np.zeros(len(self.points), dtype=bool)
        details = {
            "description": (
                "side-aware polynomial RANSAC X=f(Z), normalized Z, optional "
                "distance-adaptive threshold and distance-balanced scoring"
            ),
            "config_id": config_id(config),
            "parameters": config,
        }
        for side in ("left", "right"):
            selected, side_details = self.apply_side(side, config)
            final[self.side_indexes[side]] = selected
            details[side] = side_details
        return final, details


def safety_metrics(
    mask: np.ndarray, points: np.ndarray
) -> dict[str, float | int]:
    far = points[:, 1] >= 30.0
    return {
        "points": int(np.count_nonzero(mask)),
        "point_retention": float(np.mean(mask)),
        "far_points": int(np.count_nonzero(mask & far)),
        "far_retention": float(
            np.count_nonzero(mask & far) / max(np.count_nonzero(far), 1)
        ),
    }


def evaluate_safety(
    configs: list[dict[str, object]],
    points: np.ndarray,
    ranges: list[dict[str, object]],
    manual_points: np.ndarray,
    manual_ranges: list[dict[str, object]],
    seeds: int,
    seed_offset: int,
) -> tuple[list[dict[str, object]], dict[str, list[np.ndarray]]]:
    raw: dict[str, list[dict[str, object]]] = {
        config_id(config): [] for config in configs
    }
    masks: dict[str, list[np.ndarray]] = {config_id(config): [] for config in configs}
    for repeat in range(seeds):
        real_evaluator = RansacEvaluator(
            points, ranges, base.SEED + seed_offset + repeat
        )
        manual_evaluator = RansacEvaluator(
            manual_points,
            manual_ranges,
            base.SEED + seed_offset + 50_000 + repeat,
        )
        for config in configs:
            identifier = config_id(config)
            real_mask, _ = real_evaluator.apply(config)
            manual_mask, _ = manual_evaluator.apply(config)
            real = safety_metrics(real_mask, points)
            manual = safety_metrics(manual_mask, manual_points)
            raw[identifier].append(
                {
                    "seed_repeat": repeat,
                    **{f"real_{key}": value for key, value in real.items()},
                    **{f"manual_{key}": value for key, value in manual.items()},
                }
            )
            masks[identifier].append(real_mask)
    summary = []
    by_id = {config_id(config): config for config in configs}
    for identifier, rows in raw.items():
        config = by_id[identifier]
        summary.append(
            {
                "config_id": identifier,
                **config,
                "real_point_retention_mean": float(
                    np.mean([row["real_point_retention"] for row in rows])
                ),
                "real_point_retention_min": float(
                    np.min([row["real_point_retention"] for row in rows])
                ),
                "real_far_retention_mean": float(
                    np.mean([row["real_far_retention"] for row in rows])
                ),
                "real_far_retention_min": float(
                    np.min([row["real_far_retention"] for row in rows])
                ),
                "manual_point_retention_mean": float(
                    np.mean([row["manual_point_retention"] for row in rows])
                ),
                "manual_point_retention_min": float(
                    np.min([row["manual_point_retention"] for row in rows])
                ),
                "manual_far_retention_mean": float(
                    np.mean([row["manual_far_retention"] for row in rows])
                ),
                "manual_far_retention_min": float(
                    np.min([row["manual_far_retention"] for row in rows])
                ),
                "real_kept_points_min": int(
                    np.min([row["real_points"] for row in rows])
                ),
                "real_kept_points_max": int(
                    np.max([row["real_points"] for row in rows])
                ),
            }
        )
    return summary, masks


def evaluate_injections(
    configs: list[dict[str, object]],
    points: np.ndarray,
    ranges: list[dict[str, object]],
    repeats: int,
    seed_offset: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for repeat in range(repeats):
        for scenario_index, (scenario, injector) in enumerate(
            injection.INJECTORS.items()
        ):
            data_seed = base.SEED + seed_offset + repeat * 101 + scenario_index
            rng = np.random.default_rng(data_seed)
            contaminated, known_outlier = injector(points, ranges, rng)
            evaluator = RansacEvaluator(
                contaminated,
                ranges,
                base.SEED + seed_offset + 500_000 + repeat * 101 + scenario_index,
            )
            for config in configs:
                keep, _ = evaluator.apply(config)
                metrics = injection.classifier_metrics(
                    keep, known_outlier, points
                )
                rows.append(
                    {
                        "config_id": config_id(config),
                        **config,
                        "repeat": repeat,
                        "scenario": scenario,
                        "known_outliers": int(np.count_nonzero(known_outlier)),
                        **metrics,
                    }
                )
    return rows


def aggregate_trials(
    configs: list[dict[str, object]],
    trials: list[dict[str, object]],
    safety: list[dict[str, object]],
) -> list[dict[str, object]]:
    safety_by_id = {str(row["config_id"]): row for row in safety}
    output = []
    for config in configs:
        identifier = config_id(config)
        selected = [row for row in trials if row["config_id"] == identifier]
        scenario_stats = {}
        for scenario in injection.INJECTORS:
            rows = [row for row in selected if row["scenario"] == scenario]
            scenario_stats[scenario] = {
                "f1_mean": float(np.mean([row["f1"] for row in rows])),
                "f1_std": float(np.std([row["f1"] for row in rows])),
                "precision_mean": float(
                    np.mean([row["precision"] for row in rows])
                ),
                "recall_mean": float(np.mean([row["recall"] for row in rows])),
            }
        safety_row = safety_by_id[identifier]
        output.append(
            {
                "config_id": identifier,
                **config,
                "mean_f1": float(np.mean([row["f1"] for row in selected])),
                "f1_std_all_trials": float(
                    np.std([row["f1"] for row in selected])
                ),
                "worst_scenario_f1": min(
                    item["f1_mean"] for item in scenario_stats.values()
                ),
                "mean_precision": float(
                    np.mean([row["precision"] for row in selected])
                ),
                "mean_recall": float(
                    np.mean([row["recall"] for row in selected])
                ),
                "mean_clean_retention_in_injection": float(
                    np.mean([row["clean_retention"] for row in selected])
                ),
                "mean_clean_far_retention_in_injection": float(
                    np.mean([row["clean_far_retention"] for row in selected])
                ),
                **{
                    f"f1_{scenario}": stats["f1_mean"]
                    for scenario, stats in scenario_stats.items()
                },
                **{
                    f"f1_std_{scenario}": stats["f1_std"]
                    for scenario, stats in scenario_stats.items()
                },
                **{
                    key: value
                    for key, value in safety_row.items()
                    if key
                    not in {
                        "config_id",
                        "degree",
                        "base_threshold",
                        "distance_slope",
                        "threshold_cap",
                        "iterations",
                        "minimum_sample_z_span",
                        "score_mode",
                        "local_refinements",
                    }
                },
            }
        )
    return output


def eligible(row: dict[str, object]) -> bool:
    return (
        float(row["real_point_retention_min"]) >= 0.95
        and float(row["real_far_retention_min"]) >= 0.90
        and float(row["manual_point_retention_min"]) >= 0.95
        and float(row["manual_far_retention_min"]) >= 0.90
        and float(row["mean_clean_retention_in_injection"]) >= 0.95
        and float(row["mean_clean_far_retention_in_injection"]) >= 0.90
    )


def rank_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    candidates = [row for row in rows if eligible(row)]
    if not candidates:
        raise RuntimeError("No RANSAC configuration satisfied the safety constraints.")
    return sorted(
        candidates,
        key=lambda row: (
            -float(row["worst_scenario_f1"]),
            -float(row["mean_f1"]),
            float(row["f1_std_all_trials"]),
            -float(row["mean_clean_retention_in_injection"]),
            -float(row["real_point_retention_mean"]),
        ),
    )


def fit_geometry_diagnostics(
    points: np.ndarray,
    ranges: list[dict[str, object]],
    mask: np.ndarray,
    degree: int,
) -> dict[str, object]:
    coefficients = {}
    z_limits = {}
    for side in ("left", "right"):
        indexes = np.concatenate(
            [
                np.arange(int(item["start"]), int(item["end"]))
                for item in ranges
                if item["side"] == side
            ]
        )
        subset = points[indexes][mask[indexes]]
        coefficients[side] = np.polyfit(subset[:, 1], subset[:, 0], degree)
        z_limits[side] = (float(np.min(subset[:, 1])), float(np.max(subset[:, 1])))
    low = max(3.0, z_limits["left"][0], z_limits["right"][0])
    high = min(50.0, z_limits["left"][1], z_limits["right"][1])
    z = np.arange(low, high + 1e-9, 0.25)
    left_x = np.polyval(coefficients["left"], z)
    right_x = np.polyval(coefficients["right"], z)
    width = right_x - left_x

    def curvature(coef: np.ndarray, values: np.ndarray) -> np.ndarray:
        first = np.polyval(np.polyder(coef, 1), values)
        second = np.polyval(np.polyder(coef, 2), values)
        return np.abs(second) / np.power(1.0 + first**2, 1.5)

    left_curvature = curvature(coefficients["left"], z)
    right_curvature = curvature(coefficients["right"], z)
    return {
        "diagnostic_only": True,
        "warning": (
            "The two selected outer lines are not proven to be the boundaries of "
            "one traffic lane; absolute 3.0-3.75 m width is not used as a deletion rule."
        ),
        "common_z_range_m": [float(low), float(high)],
        "fitted_width_m": {
            "minimum": float(np.min(width)),
            "median": float(np.median(width)),
            "maximum": float(np.max(width)),
            "fraction_in_3p0_to_3p75": float(
                np.mean((width >= 3.0) & (width <= 3.75))
            ),
        },
        "curvature_1_per_m": {
            "left_median": float(np.median(left_curvature)),
            "right_median": float(np.median(right_curvature)),
            "median_absolute_difference": float(
                np.median(np.abs(left_curvature - right_curvature))
            ),
        },
    }


def selected_real_result(
    config: dict[str, object],
    points: np.ndarray,
    ranges: list[dict[str, object]],
    manual_points: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, object], dict[str, object], dict[str, object]]:
    evaluator = RansacEvaluator(points, ranges, seed)
    mask, details = evaluator.apply(config)
    frames = base.frames_from_mask(points, ranges, mask)
    _, fusion = base.lane_mask(frames, *base.FUSION_RANGE)
    metrics = base.method_metrics(
        "selected_ransac", points, ranges, mask, manual_points, fusion
    )
    return mask, details, metrics, fusion


def draw_point_comparison(
    path: Path,
    points: np.ndarray,
    ranges: list[dict[str, object]],
    methods: list[tuple[str, np.ndarray, str]],
) -> None:
    figure, axes = plt.subplots(1, len(methods), figsize=(4.0 * len(methods), 7.0))
    axes = np.atleast_1d(axes)
    for axis, (name, mask, subtitle) in zip(axes, methods):
        for item in ranges:
            start, end = int(item["start"]), int(item["end"])
            selected = mask[start:end]
            lane = points[start:end][selected]
            frame_id = int(item["frame_id"])
            axis.scatter(
                lane[:, 0],
                lane[:, 1],
                s=8,
                color=base.FRAME_COLORS_RGB[frame_id],
            )
        axis.set(
            xlim=base.FUSION_RANGE[0],
            ylim=base.FUSION_RANGE[1],
            xlabel="X right [m]",
            ylabel="Z forward [m]",
            title=f"{name}\n{subtitle}",
        )
        axis.set_aspect("equal")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def draw_parameter_summary(
    path: Path,
    rows: list[dict[str, object]],
    selected_id: str,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = [
        "#d62728" if row["config_id"] == selected_id else "#4c78a8"
        for row in rows
    ]
    axes[0].scatter(
        [row["real_far_retention_mean"] for row in rows],
        [row["mean_f1"] for row in rows],
        c=colors,
        s=20,
        alpha=0.8,
    )
    axes[0].axvline(0.90, color="black", linestyle="--", linewidth=1)
    axes[0].set(
        xlabel="Real Z>=30 m retention",
        ylabel="Mean injected-outlier F1",
        title="Validation configurations",
    )
    axes[0].grid(alpha=0.25)
    top = rank_rows(rows)[:10]
    axes[1].barh(
        range(len(top)),
        [row["worst_scenario_f1"] for row in reversed(top)],
        color=[
            "#d62728" if row["config_id"] == selected_id else "#777777"
            for row in reversed(top)
        ],
    )
    axes[1].set_yticks(
        range(len(top)),
        [str(row["config_id"])[:31] for row in reversed(top)],
        fontsize=7,
    )
    axes[1].set(
        xlabel="Worst-scenario mean F1",
        title="Top eligible configurations",
    )
    axes[1].grid(axis="x", alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def hash_inputs(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: {"path": str(path.resolve()), "sha256": base.sha256(path)}
        for key, path in {
            "clrnet_json": args.clrnet_json,
            "manual_json": args.manual_json,
            "calib": args.calib,
            "poses": args.poses,
        }.items()
    }


def main() -> None:
    args = parse_args()
    for name in (
        "phase1_repeats",
        "phase2_repeats",
        "final_repeats",
        "phase1_top",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    base.require_empty_output(args.output_dir)
    points, ranges, manual_points, manual_ranges = load_points(args)

    phase1_configs = phase1_grid()
    phase1_safety, _ = evaluate_safety(
        phase1_configs,
        points,
        ranges,
        manual_points,
        manual_ranges,
        seeds=3,
        seed_offset=SAFETY_SEED_OFFSET,
    )
    phase1_trials = evaluate_injections(
        phase1_configs,
        points,
        ranges,
        repeats=args.phase1_repeats,
        seed_offset=PHASE1_SEED_OFFSET,
    )
    phase1_summary = aggregate_trials(
        phase1_configs, phase1_trials, phase1_safety
    )
    phase1_ranked = rank_rows(phase1_summary)
    phase1_top = phase1_ranked[: args.phase1_top]

    phase2_configs = phase2_grid(phase1_top)
    phase2_safety, phase2_masks = evaluate_safety(
        phase2_configs,
        points,
        ranges,
        manual_points,
        manual_ranges,
        seeds=10,
        seed_offset=SAFETY_SEED_OFFSET + 10_000,
    )
    phase2_trials = evaluate_injections(
        phase2_configs,
        points,
        ranges,
        repeats=args.phase2_repeats,
        seed_offset=PHASE2_SEED_OFFSET,
    )
    phase2_summary = aggregate_trials(
        phase2_configs, phase2_trials, phase2_safety
    )
    selected_row = rank_rows(phase2_summary)[0]
    selected_id = str(selected_row["config_id"])
    selected_config = next(
        config for config in phase2_configs if config_id(config) == selected_id
    )

    final_safety, final_masks = evaluate_safety(
        [selected_config],
        points,
        ranges,
        manual_points,
        manual_ranges,
        seeds=args.final_repeats,
        seed_offset=SAFETY_SEED_OFFSET + 20_000,
    )
    final_trials = evaluate_injections(
        [selected_config],
        points,
        ranges,
        repeats=args.final_repeats,
        seed_offset=FINAL_SEED_OFFSET,
    )
    final_summary = aggregate_trials(
        [selected_config], final_trials, final_safety
    )[0]

    final_mask, final_details, final_metrics, final_fusion = selected_real_result(
        selected_config,
        points,
        ranges,
        manual_points,
        base.SEED + 999_999,
    )
    final_frames = base.frames_from_mask(points, ranges, final_mask)
    legacy_mask, _ = base.legacy_mask(points)
    quadratic_mask, _ = base.polynomial_ransac_mask(
        points, ranges, degree=2, threshold=0.3
    )
    raw_mask = np.ones(len(points), dtype=bool)

    output_selected = args.output_dir / "selected_method"
    base.imwrite(
        output_selected / "points_by_frame.png", base.draw_points(final_frames)
    )
    base.imwrite(
        output_selected / "weighted_heatmap.png", final_fusion["heatmap"]
    )
    base.imwrite(
        output_selected / "weighted_binary.png", final_fusion["binary"]
    )
    base.plot_points(
        output_selected / "metric_plot.png",
        final_frames,
        f"selected RANSAC: {selected_id}",
    )

    comparison = args.output_dir / "comparison"
    draw_point_comparison(
        comparison / "best_vs_baselines.png",
        points,
        ranges,
        [
            ("No denoising", raw_mask, "590 points"),
            (
                "Legacy",
                legacy_mask,
                f"{np.count_nonzero(legacy_mask)} points",
            ),
            (
                "Previous quadratic 0.3 m",
                quadratic_mask,
                f"{np.count_nonzero(quadratic_mask)} points",
            ),
            (
                "Optimized RANSAC",
                final_mask,
                f"{np.count_nonzero(final_mask)} points",
            ),
        ],
    )
    draw_parameter_summary(
        comparison / "parameter_search_summary.png",
        phase2_summary,
        selected_id,
    )
    example_rng = np.random.default_rng(base.SEED + FINAL_SEED_OFFSET + 999)
    example_points, example_outlier = injection.inject_mixed(
        points, ranges, example_rng
    )
    example_evaluator = RansacEvaluator(
        example_points, ranges, base.SEED + FINAL_SEED_OFFSET + 1_999
    )
    example_keep, _ = example_evaluator.apply(selected_config)
    injection.draw_injection_example(
        comparison / "known_outlier_example.png",
        example_points,
        example_outlier,
        example_keep,
    )

    diagnostics = fit_geometry_diagnostics(
        points,
        ranges,
        final_mask,
        degree=int(selected_config["degree"]),
    )
    audit_dir = args.output_dir / "00_audit"
    base.write_csv(audit_dir / "phase1_all_trials.csv", phase1_trials)
    base.write_csv(audit_dir / "phase1_summary.csv", phase1_summary)
    base.write_csv(audit_dir / "phase2_validation_trials.csv", phase2_trials)
    base.write_csv(audit_dir / "phase2_summary.csv", phase2_summary)
    base.write_csv(audit_dir / "final_stability_trials.csv", final_trials)
    base.write_csv(audit_dir / "final_safety_summary.csv", final_safety)

    phase1_trial_count = len(phase1_trials)
    phase2_trial_count = len(phase2_trials)
    final_trial_count = len(final_trials)
    all_config_ids = {
        config_id(config) for config in phase1_configs + phase2_configs
    }
    audit = {
        "status": "complete",
        "scope": (
            "RANSAC-only parameter and constraint optimization. The production "
            "pipeline and original result directories were not modified."
        ),
        "inputs": hash_inputs(args),
        "data": {
            "real_points": int(len(points)),
            "manual_pseudo_label_points": int(len(manual_points)),
            "reference_frame": 4,
            "warning": (
                "There is no lane-line ground truth for these five frames. Manual "
                "annotations are pseudo-labels and are used only as an anti-over-"
                "deletion safety check."
            ),
        },
        "search": {
            "phase1_configurations": len(phase1_configs),
            "phase1_repeats_per_scenario": args.phase1_repeats,
            "phase1_controlled_trials": phase1_trial_count,
            "phase2_configurations": len(phase2_configs),
            "phase2_repeats_per_scenario": args.phase2_repeats,
            "phase2_controlled_trials": phase2_trial_count,
            "final_repeats_per_scenario": args.final_repeats,
            "final_controlled_trials": final_trial_count,
            "unique_configurations_tested": len(all_config_ids),
            "total_controlled_trials": (
                phase1_trial_count + phase2_trial_count + final_trial_count
            ),
            "seed_split": {
                "phase1_tuning": PHASE1_SEED_OFFSET,
                "phase2_validation": PHASE2_SEED_OFFSET,
                "final_stability": FINAL_SEED_OFFSET,
            },
        },
        "selection_constraints": {
            "minimum_real_point_retention_each_seed": 0.95,
            "minimum_real_far_retention_each_seed": 0.90,
            "minimum_manual_safety_point_retention_each_seed": 0.95,
            "minimum_manual_safety_far_retention_each_seed": 0.90,
            "minimum_injection_clean_retention_mean": 0.95,
            "minimum_injection_clean_far_retention_mean": 0.90,
            "ranking": (
                "worst-scenario mean F1, mean F1, lower F1 standard deviation, "
                "clean retention, real retention"
            ),
        },
        "selected_config_id": selected_id,
        "selected_parameters": selected_config,
        "selected_real_run": {
            "metrics": final_metrics,
            "details": final_details,
        },
        "selected_final_stability": final_summary,
        "geometry_diagnostics": diagnostics,
        "limitations": [
            "Injected anomalies are controlled tests, not KITTI lane accuracy.",
            "The clean base assumption can hide unknown errors already present in the real points.",
            "Only five consecutive frames were used.",
            "Absolute lane-width rejection was not enabled because the two outer detected lines are not proven to bound one traffic lane.",
        ],
    }
    (audit_dir / "ransac_optimization.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    scenarios = {
        scenario: {
            "f1": float(
                np.mean(
                    [
                        row["f1"]
                        for row in final_trials
                        if row["scenario"] == scenario
                    ]
                )
            ),
            "precision": float(
                np.mean(
                    [
                        row["precision"]
                        for row in final_trials
                        if row["scenario"] == scenario
                    ]
                )
            ),
            "recall": float(
                np.mean(
                    [
                        row["recall"]
                        for row in final_trials
                        if row["scenario"] == scenario
                    ]
                )
            ),
            "clean_retention": float(
                np.mean(
                    [
                        row["clean_retention"]
                        for row in final_trials
                        if row["scenario"] == scenario
                    ]
                )
            ),
            "clean_far_retention": float(
                np.mean(
                    [
                        row["clean_far_retention"]
                        for row in final_trials
                        if row["scenario"] == scenario
                    ]
                )
            ),
        }
        for scenario in injection.INJECTORS
    }
    width = diagnostics["fitted_width_m"]
    report = [
        "# RANSAC参数与限制优化核验报告",
        "",
        "## 结论",
        "",
        f"- 共测试 {len(all_config_ids)} 个不同RANSAC配置，完成 "
        f"{phase1_trial_count + phase2_trial_count + final_trial_count} 次受控异常试验。",
        f"- 最终配置：`{selected_id}`。",
        f"- 真实五帧确定性运行保留 {final_metrics['points']}/{len(points)} 点，"
        f"30 m后保留率 {final_metrics['far_point_retention_z_ge_30m']:.1%}。",
        "- 最终方案仍属于分侧多项式RANSAC；没有替换为其他去噪模型。",
        "",
        "## 最终参数",
        "",
        f"- 多项式次数：{selected_config['degree']}",
        f"- 基础残差阈值：{selected_config['base_threshold']:.3f} m",
        f"- 距离阈值增量：{selected_config['distance_slope']:.3f} m/m",
        f"- 阈值上限：{selected_config['threshold_cap']:.2f} m",
        f"- RANSAC迭代：{selected_config['iterations']}",
        f"- 随机样本最小Z跨度：{selected_config['minimum_sample_z_span']:.1f} m",
        f"- 假设评分：{selected_config['score_mode']}",
        f"- 局部重拟合次数：{selected_config['local_refinements']}",
        "",
        "## 最终独立重复验证（50次平均）",
        "",
        "| 场景 | Precision | Recall | F1 | 干净点保留 | 远处干净点保留 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scenario, values in scenarios.items():
        report.append(
            f"| {scenario} | {values['precision']:.1%} | {values['recall']:.1%} | "
            f"{values['f1']:.1%} | {values['clean_retention']:.1%} | "
            f"{values['clean_far_retention']:.1%} |"
        )
    report.extend(
        [
            "",
            "## 宽度与曲率限制的真实核验",
            "",
            f"- 最终拟合两条外侧线的间距为 {width['minimum']:.2f}–"
            f"{width['maximum']:.2f} m，中位数 {width['median']:.2f} m。",
            f"- 落在3.0–3.75 m范围的比例为 {width['fraction_in_3p0_to_3p75']:.1%}。",
            "- 因此不能把3.0–3.75 m作为当前两条线的硬删除范围；这两条外侧线并未被证明是同一条交通车道的左右边界。",
            "- 曲率一致性已输出为诊断量，但在没有真值和更长序列前不作为硬删除规则。",
            "",
            "## 使用边界",
            "",
            "- 注入异常指标不是KITTI车道线准确率。",
            "- 人工标注只用于检查是否过删，不作为真值。",
            "- 最终方案仍需在更长序列和人工逐点核验上复查。",
        ]
    )
    (args.output_dir / "RANSAC_OPTIMIZATION_REPORT_ZH.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(json.dumps(audit["search"], ensure_ascii=False))
    print(selected_id)
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
