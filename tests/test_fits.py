"""Power-law fit port. Recovery case adapted from upstream (MIT)."""

import pytest

from spectral_telemetry.core.fits import fit_power_law, fit_power_law_with_ci


def test_power_law_fit_recovers_beta():
    x = [1, 2, 4, 8]
    y = [3 * v**2 for v in x]
    out = fit_power_law(x, y)
    assert abs(out["beta"] - 2.0) < 1e-8


def test_ci_brackets_truth_on_noisy_data():
    import numpy as np

    rng = np.random.default_rng(0)
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    y = 2.0 * x**0.7 * np.exp(rng.normal(0, 0.02, size=x.size))
    fit = fit_power_law_with_ci(x, y)
    assert fit["valid"] and fit["beta_lower"] <= 0.7 <= fit["beta_upper"]
    assert fit["ci_method"] == "ols_loglog_t_interval"


def test_invalid_inputs():
    assert fit_power_law_with_ci([1, 2], [1, 2])["valid"] is False  # < min_points
    assert fit_power_law_with_ci([1, -2, 3], [1, 2, float("nan")])["valid"] is False
    with pytest.raises(ValueError):
        fit_power_law([1], [1])
