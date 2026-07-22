"""Pass-3: estimation diagnostics, eigenvalue validation, Renyi stability, fits."""

import numpy as np
import pytest

from spectral_telemetry.core.fits import _t_critical_975, fit_power_law_with_ci
from spectral_telemetry.core.ranks import (
    eigs_from_covariance,
    estimation_diagnostics,
    rank_metrics_from_activations,
    renyi_rank,
    soft_rank,
    validate_covariance,
)


def test_estimation_status_thresholds():
    assert estimation_diagnostics(120, 10)["estimation_status"] == "well_sampled"
    assert estimation_diagnostics(50, 10)["estimation_status"] == "moderately_undersampled"
    assert estimation_diagnostics(15, 10)["estimation_status"] == "severely_undersampled"
    assert estimation_diagnostics(1, 10)["estimation_status"] == "insufficient_samples"
    custom = estimation_diagnostics(50, 10, thresholds={"well_sampled": 4.0})
    assert custom["estimation_status"] == "well_sampled"
    d = estimation_diagnostics(15, 10)
    assert d["sample_to_dim_ratio"] == pytest.approx(1.5)
    assert d["max_empirical_rank"] == 10  # min(D, N-1) = min(10, 14)
    assert estimation_diagnostics(5, 16)["max_empirical_rank"] == 4


def test_undersampled_record_flags_itself():
    rng = np.random.default_rng(0)
    rec = rank_metrics_from_activations(rng.normal(size=(5, 16)))  # N << D
    assert rec["estimation_status"] == "severely_undersampled"
    assert rec["max_empirical_rank"] == 4
    assert rec["soft_rank_fraction_of_empirical_max"] <= 1.05
    assert 0 < rec["soft_rank_fraction_of_dim"] < 1
    assert rec["min_raw_eigenvalue"] <= rec["negative_eigenvalue_mass"] + 1e-6
    well = rank_metrics_from_activations(rng.normal(size=(400, 16)))
    assert well["estimation_status"] == "well_sampled"


def test_renyi_stable_near_one_and_monotone_to_inf():
    rng = np.random.default_rng(1)
    lam = rng.uniform(0.1, 2.0, size=12)
    r1 = soft_rank(lam)
    for alpha in (0.999, 0.999999, 1.000001, 1.001):
        assert renyi_rank(lam, alpha) == pytest.approx(r1, rel=2e-3)
        assert abs(renyi_rank(lam, alpha) - r1) <= abs(alpha - 1.0) * 50 * r1
    chain = [renyi_rank(lam, a) for a in (0.5, 0.999, 1.0, 1.001, 2.0, 8.0, np.inf)]
    assert all(a >= b - 1e-9 for a, b in zip(chain, chain[1:]))
    p = np.clip(lam, 1e-12, None) / np.clip(lam, 1e-12, None).sum()
    assert renyi_rank(lam, np.inf) == pytest.approx(1.0 / p.max(), rel=1e-12)
    for bad in (0.0, -1.0, -np.inf):
        with pytest.raises(ValueError):
            renyi_rank(lam, bad)


def test_non_finite_eigenvalues_rejected():
    with pytest.raises(ValueError, match="NaN or Inf"):
        soft_rank([1.0, np.nan, 2.0])
    with pytest.raises(ValueError, match="NaN or Inf"):
        renyi_rank([1.0, np.inf], 2.0)


def test_covariance_validation():
    with pytest.raises(ValueError, match="square"):
        validate_covariance(np.ones((3, 4)))
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_covariance(np.array([[1.0, np.nan], [np.nan, 1.0]]))
    bad = np.array([[1.0, 0.5], [0.1, 1.0]])
    with pytest.raises(ValueError, match="symmetric"):
        validate_covariance(bad)


def test_materially_negative_eigenvalue_raises_tiny_is_clipped():
    with pytest.raises(ValueError, match="materially negative"):
        eigs_from_covariance(np.diag([1.0, -0.5]))
    eigs, diag = eigs_from_covariance(np.diag([1.0, -1e-14]), return_diagnostics=True)
    assert eigs.min() >= 1e-12
    assert diag["min_raw_eigenvalue"] == pytest.approx(-1e-14)
    assert 0 <= diag["negative_eigenvalue_mass"] < 1e-10


def test_t_critical_conservative_fallback():
    assert _t_critical_975(1) == 12.706
    assert _t_critical_975(22) == 2.074  # now tabulated exactly
    assert _t_critical_975(35) == 2.042  # gap -> next SMALLER dof (30)
    assert _t_critical_975(500) == 1.980  # capped conservative tail
    vals = [_t_critical_975(d) for d in range(1, 200)]
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:]))


def test_fit_invalid_reasons_and_min_points():
    out = fit_power_law_with_ci([1, 2], [1, 2])
    assert out["valid"] is False and "need >=" in out["reason"]
    out = fit_power_law_with_ci([2, 2, 2], [1, 2, 3])
    assert out["valid"] is False and "unique" in out["reason"]
    with pytest.raises(ValueError, match="min_points"):
        fit_power_law_with_ci([1, 2, 3], [1, 2, 3], min_points=1)
