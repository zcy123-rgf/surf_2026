"""Fixed-family side-aware polynomial RANSAC used by the audited reference.

The implementation is copied from the verified optimizer's evaluator so the
reference run does not depend on parameter-search or injected-outlier scripts.
It fits X=f(Z) independently for left and right point ranges, normalizes Z for
conditioning, supports a distance-adaptive residual threshold, balances the
hypothesis score across distance bands, and optionally performs local refits.
"""

from __future__ import annotations

import math

import numpy as np


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


def polynomial_design_normalized(
    z: np.ndarray, degree: int, center: float, scale: float
) -> np.ndarray:
    normalized = (np.asarray(z, dtype=np.float64) - center) / scale
    return np.vander(normalized, N=degree + 1, increasing=True)


class SideAwarePolynomialRansac:
    """Apply one deterministic RANSAC seed to left/right point ranges."""

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
        residuals = np.abs(subset[:, 0, None] - design @ coefficients_array.T)
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
        details: dict[str, object] = {
            "description": (
                "side-aware polynomial RANSAC X=f(Z), normalized Z, "
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
