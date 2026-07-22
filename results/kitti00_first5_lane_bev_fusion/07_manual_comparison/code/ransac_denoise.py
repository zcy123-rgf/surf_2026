"""Standalone deterministic RANSAC denoiser used by the fusion comparison.

Source: zcy123-rgf/surf_2026, branch ``fangyu``, commit 6d1ae811,
``surf_bev/lane_fusion_weighted.py``.  The mathematical clustering and
RANSAC rules are preserved.  An explicit RNG seed is the only behavioral
control added so comparisons are reproducible.
"""

from __future__ import annotations

import numpy as np


def simple_cluster_by_x(points: np.ndarray):
    """Cluster lane points by X coordinate before fitting left/right lines."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 10:
        return points, points

    x_values = points[:, 0]
    left_center = (
        np.percentile(x_values[x_values < 0], 75) if np.any(x_values < 0) else -1.5
    )
    right_center = (
        np.percentile(x_values[x_values > 0], 25) if np.any(x_values > 0) else 1.5
    )
    left_mask = np.abs(points[:, 0] - left_center) < 0.8
    right_mask = np.abs(points[:, 0] - right_center) < 0.8
    if np.sum(left_mask) < 20:
        left_mask = np.abs(points[:, 0] - left_center) < 1.2
    if np.sum(right_mask) < 20:
        right_mask = np.abs(points[:, 0] - right_center) < 1.2
    return points[left_mask], points[right_mask]


def ransac_line(
    points: np.ndarray,
    *,
    max_iter: int = 100,
    threshold: float = 0.3,
    rng: np.random.Generator | None = None,
):
    """Preserved two-point line RANSAC with deterministic RNG support."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 5:
        return None, points
    if rng is None:
        rng = np.random.default_rng()

    best_inliers = np.empty((0, 2), dtype=np.float64)
    for _ in range(max_iter):
        idx = rng.choice(len(points), 2, replace=False)
        p1, p2 = points[idx]
        if abs(p2[0] - p1[0]) < 1e-6:
            continue
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        intercept = p1[1] - slope * p1[0]
        distances = np.abs(points[:, 1] - (slope * points[:, 0] + intercept))
        distances /= np.sqrt(slope**2 + 1)
        inliers = points[distances < threshold]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    if len(best_inliers) < 5:
        return None, points
    A = np.vstack([best_inliers[:, 0], np.ones(len(best_inliers))]).T
    slope, intercept = np.linalg.lstsq(A, best_inliers[:, 1], rcond=None)[0]
    return (float(slope), float(intercept)), best_inliers


def denoise_left_right(
    points: np.ndarray,
    *,
    max_iter: int = 100,
    threshold: float = 0.3,
    seed: int = 20260717,
):
    """Run the exact cluster-then-RANSAC structure as a reusable function."""
    left_raw, right_raw = simple_cluster_by_x(points)
    rng = np.random.default_rng(seed)
    left_model, left_inliers = ransac_line(
        left_raw, max_iter=max_iter, threshold=threshold, rng=rng
    )
    right_model, right_inliers = ransac_line(
        right_raw, max_iter=max_iter, threshold=threshold, rng=rng
    )
    return {
        "left_raw": left_raw,
        "right_raw": right_raw,
        "left_model": left_model,
        "right_model": right_model,
        "left_inliers": left_inliers,
        "right_inliers": right_inliers,
    }
