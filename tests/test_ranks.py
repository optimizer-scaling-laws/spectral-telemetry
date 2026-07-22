"""Core rank metrics. Includes cases adapted from upstream
spectral-scaling-laws tests (MIT)."""

import numpy as np
import pytest

from spectral_telemetry.core.ranks import (
    hard_rank,
    rank_metrics_from_activations,
    renyi_rank,
    scaling_asymmetry,
    soft_rank,
    spectral_asymmetry,
)


def test_uniform_spectrum_has_full_effective_rank():
    lam = np.ones(8)
    assert soft_rank(lam) == pytest.approx(8.0, abs=1e-4)
    assert hard_rank(lam) == pytest.approx(8.0, abs=1e-4)


def test_rank_one_spectrum_has_low_effective_rank():
    assert hard_rank([1.0, 1e-12, 1e-12, 1e-12]) < 1.01
    assert soft_rank([10.0, 0.0, 0.0, 0.0]) == pytest.approx(1.0, abs=1e-4)


def test_soft_dominates_hard_on_random_spectra():
    rng = np.random.default_rng(0)
    for _ in range(20):
        lam = rng.uniform(0.0, 1.0, size=16)
        assert soft_rank(lam) >= hard_rank(lam) - 1e-9


def test_renyi_special_cases_and_monotonicity():
    rng = np.random.default_rng(1)
    lam = rng.uniform(0.1, 2.0, size=12)
    assert renyi_rank(lam, 1.0) == pytest.approx(soft_rank(lam), rel=1e-9)
    assert renyi_rank(lam, 2.0) == pytest.approx(hard_rank(lam), rel=1e-9)
    values = [renyi_rank(lam, a) for a in (0.5, 1.0, 2.0, 4.0, 8.0)]
    assert all(a >= b - 1e-9 for a, b in zip(values, values[1:]))


def test_metrics_from_activations_shapes_and_keys():
    rng = np.random.default_rng(2)
    m = rank_metrics_from_activations(rng.normal(size=(64, 16)))
    assert m["n_samples"] == 64 and m["hidden_dim"] == 16
    assert m["soft_rank"] > 0 and m["hard_rank"] > 0
    assert m["soft_rank"] >= m["hard_rank"] - 1e-6


def test_asymmetries():
    assert spectral_asymmetry(5.0, 3.0) == 2.0
    assert scaling_asymmetry(0.8, 0.29) == pytest.approx(0.51)
