from argparse import Namespace

import numpy as np

from scripts import fit_coupled_bspline as coupled


def synthetic_groups(count: int = 10) -> list[dict[str, object]]:
    groups = []
    for frame_id in range(count):
        progress = np.linspace(frame_id * 0.8, frame_id * 0.8 + 20.0, 40)
        centre = 0.35 * np.sin(progress / 12.0)
        width = 3.6 + 0.08 * np.cos(progress / 15.0)
        left = np.column_stack([progress, centre - 0.5 * width])
        right = np.column_stack([progress, centre + 0.5 * width])
        if frame_id == 4:
            right[15:19, 1] += 1.2
        groups.append(
            {
                "group_id": frame_id,
                "lanes_sd": [left, right],
            }
        )
    return groups


def settings() -> Namespace:
    return Namespace(
        bin_size_m=0.5,
        huber_delta_m=0.2,
        irls_iterations=4,
        width_smoothing_multiplier=4.0,
        log_width_huber_delta=0.12,
        maximum_cv_folds=6,
    )


def test_coupled_spline_guarantees_positive_non_crossing_width():
    fit = coupled.fit_coupled(synthetic_groups(), 0.04, settings())
    progress = np.linspace(fit.s_min, fit.s_max, 300)
    left = fit.evaluate("left", progress)
    right = fit.evaluate("right", progress)

    assert np.all(np.isfinite(left))
    assert np.all(np.isfinite(right))
    assert np.all(right > left)
    assert np.allclose(right - left, fit.width(progress))
    assert 3.0 < float(np.median(fit.width(progress))) < 4.2


def test_coupled_cross_validation_selects_registered_grid_candidate():
    selected, rows = coupled.cross_validate(
        synthetic_groups(), [0.01, 0.04], settings()
    )

    assert selected in (0.01, 0.04)
    assert len(rows) == 2
    assert sum(bool(row["selected_by_one_standard_error_rule"]) for row in rows) == 1
    assert all(float(row["rmse_m"]) >= 0.0 for row in rows)
    assert all(int(row["fold_count"]) > 0 for row in rows)
