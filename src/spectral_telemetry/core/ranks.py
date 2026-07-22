"""Effective-rank metrics on covariance eigenspectra (numpy-only).

Numerical conventions (eps clamps, entropy form, participation ratio on raw
eigenvalues) are preserved from the upstream paper repository so values are
directly comparable with published telemetry.

Adds the order-alpha Renyi effective rank, of which the paper's two metrics
are special cases: alpha=1 -> soft rank (Shannon), alpha=2 -> hard rank
(participation ratio). Renyi rank is non-increasing in alpha, which is why
hard_rank <= soft_rank always ("forbidden region" of the phase map).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12
__all__ = [
    "EPS",
    "DEFAULT_ESTIMATION_THRESHOLDS",
    "estimation_diagnostics",
    "validate_covariance",
    "normalize_eigs",
    "spectral_entropy",
    "soft_rank",
    "hard_rank",
    "renyi_rank",
    "rank_metrics",
    "eigs_from_covariance",
    "covariance_from_activations",
    "rank_metrics_from_activations",
    "attach_rank_fractions",
    "spectral_asymmetry",
    "scaling_asymmetry",
]


def _as_eigs(eigenvalues) -> np.ndarray:
    lam = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(lam)):
        raise ValueError("eigenvalues contain NaN or Inf")
    # Negative user-supplied eigenvalues are permitted and clipped to EPS
    # (documented convention); covariance-derived spectra are validated more
    # strictly in eigs_from_covariance.
    return np.clip(lam, EPS, None)


def normalize_eigs(eigenvalues) -> np.ndarray:
    """Normalize eigenvalues into a probability distribution over eigenmodes."""
    lam = _as_eigs(eigenvalues)
    return lam / (lam.sum() + EPS)


def spectral_entropy(eigenvalues) -> float:
    """Shannon entropy of the normalized eigenspectrum."""
    p = normalize_eigs(eigenvalues)
    return float(-np.sum(p * np.log(p + EPS)))


def soft_rank(eigenvalues) -> float:
    """Shannon effective rank: exp(spectral entropy). Upstream 'diffuse capacity'."""
    return float(np.exp(spectral_entropy(eigenvalues)))


def hard_rank(eigenvalues) -> float:
    """Participation ratio (sum^2 / sum of squares). Upstream 'dominant-mode capacity'."""
    lam = _as_eigs(eigenvalues)
    return float((lam.sum() ** 2) / (np.sum(lam**2) + EPS))


def renyi_rank(eigenvalues, alpha: float) -> float:
    """Order-alpha Renyi effective rank; alpha=1 -> soft, alpha=2 -> hard.

    Computed in the log domain for stability near ``alpha = 1``:
    ``log R = logsumexp(alpha * log p) / (1 - alpha)``. ``alpha = np.inf``
    returns the min-entropy rank ``1 / max(p)``. ``alpha -> 0`` (Hartley/
    support rank) is not supported because eps-clipping makes every mode
    nominally nonzero; use a thresholded count on raw eigenvalues instead.
    """
    if np.isinf(alpha):
        if alpha > 0:
            return float(1.0 / normalize_eigs(eigenvalues).max())
        raise ValueError("alpha must be positive (or +inf)")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be positive (or +inf)")
    if abs(alpha - 1.0) < 1e-12:
        return soft_rank(eigenvalues)
    logp = np.log(normalize_eigs(eigenvalues))
    x = alpha * logp
    m = float(x.max())
    lse = m + float(np.log(np.sum(np.exp(x - m))))
    return float(np.exp(lse / (1.0 - alpha)))


def rank_metrics(eigenvalues) -> dict[str, float]:
    """Paper-facing metric dict from covariance eigenvalues."""
    return {
        "spectral_entropy": spectral_entropy(eigenvalues),
        "soft_rank": soft_rank(eigenvalues),
        "hard_rank": hard_rank(eigenvalues),
    }


def validate_covariance(cov, *, symmetry_rtol: float = 1e-8) -> np.ndarray:
    """Check a covariance matrix is finite, square, and symmetric within tolerance."""
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"covariance must be a square matrix, got shape {cov.shape}")
    if not np.all(np.isfinite(cov)):
        raise ValueError("covariance contains NaN or Inf")
    scale = max(float(np.abs(cov).max()), EPS)
    asym = float(np.abs(cov - cov.T).max())
    if asym > symmetry_rtol * scale:
        raise ValueError(
            f"covariance is not symmetric within tolerance "
            f"(max asymmetry {asym:.3e} vs scale {scale:.3e})"
        )
    return cov


def eigs_from_covariance(
    cov,
    *,
    materially_negative_rtol: float = 1e-6,
    return_diagnostics: bool = False,
):
    """Descending, eps-clamped eigenvalues of a validated covariance matrix.

    Tiny negative eigenvalues consistent with floating-point error are clipped
    to ``EPS``; a most-negative eigenvalue below
    ``-materially_negative_rtol * max|lambda|`` raises, because a materially
    indefinite "covariance" indicates corrupted input rather than noise. With
    ``return_diagnostics=True``, also returns ``min_raw_eigenvalue`` and
    ``negative_eigenvalue_mass`` (sum of negative magnitude / total magnitude).
    """
    cov = validate_covariance(cov)
    raw = np.linalg.eigvalsh(cov)
    max_abs = max(float(np.abs(raw).max()), EPS)
    min_raw = float(raw[0])
    neg_mass = float(np.sum(np.clip(-raw, 0.0, None)) / (np.sum(np.abs(raw)) + EPS))
    if min_raw < -materially_negative_rtol * max_abs:
        raise ValueError(
            f"covariance has materially negative eigenvalue {min_raw:.3e} "
            f"(scale {max_abs:.3e}); input is not a valid covariance"
        )
    clipped = np.clip(raw, EPS, None)[::-1].copy()
    if return_diagnostics:
        return clipped, {
            "min_raw_eigenvalue": min_raw,
            "negative_eigenvalue_mass": neg_mass,
        }
    return clipped


def covariance_from_activations(x2d) -> np.ndarray:
    """Sample covariance of [N, D] activations (upstream (n-1)+eps denominator)."""
    x = np.asarray(x2d, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected 2D activations, got shape {x.shape}")
    if x.shape[0] < 2:
        raise ValueError("at least two activation samples are required")
    centered = x - x.mean(axis=0, keepdims=True)
    return (centered.T @ centered) / (x.shape[0] - 1 + EPS)


DEFAULT_ESTIMATION_THRESHOLDS = {
    "well_sampled": 10.0,
    "moderately_undersampled": 3.0,
}


def estimation_diagnostics(n_samples: int, hidden_dim: int, *, thresholds=None) -> dict:
    """Sampling-regime diagnostics for an empirical covariance spectrum.

    The empirical rank of a sample covariance is at most ``min(D, N - 1)``, so
    a low rank at small ``N/D`` may reflect the sample count rather than the
    representation. Status thresholds on ``N/D`` are configurable heuristics
    (defaults: >= 10 well_sampled, >= 3 moderately_undersampled, else
    severely_undersampled; ``N < 2`` insufficient_samples) -- documented
    conventions, not statistical guarantees.
    """
    t = {**DEFAULT_ESTIMATION_THRESHOLDS, **(thresholds or {})}
    n, d = int(n_samples), int(hidden_dim)
    ratio = n / d if d > 0 else float("nan")
    if n < 2:
        status = "insufficient_samples"
    elif ratio >= t["well_sampled"]:
        status = "well_sampled"
    elif ratio >= t["moderately_undersampled"]:
        status = "moderately_undersampled"
    else:
        status = "severely_undersampled"
    return {
        "sample_to_dim_ratio": float(ratio),
        "max_empirical_rank": int(max(0, min(d, n - 1))),
        "estimation_status": status,
    }


def attach_rank_fractions(record: dict) -> dict:
    """Add normalized rank fractions (of D and of min(D, N-1)) to a record."""
    d = record["hidden_dim"]
    me = record["max_empirical_rank"]
    for kind in ("soft", "hard"):
        rank = record[f"{kind}_rank"]
        record[f"{kind}_rank_fraction_of_dim"] = float(rank / d) if d > 0 else float("nan")
        record[f"{kind}_rank_fraction_of_empirical_max"] = (
            float(rank / me) if me > 0 else float("nan")
        )
    return record


def rank_metrics_from_activations(x2d, *, estimation_thresholds=None) -> dict[str, float]:
    eigs, eig_diag = eigs_from_covariance(covariance_from_activations(x2d), return_diagnostics=True)
    out = rank_metrics(eigs)
    x = np.asarray(x2d)
    out["n_samples"] = int(x.shape[0])
    out["hidden_dim"] = int(x.shape[1])
    out.update(eig_diag)
    out.update(estimation_diagnostics(x.shape[0], x.shape[1], thresholds=estimation_thresholds))
    return attach_rank_fractions(out)


def spectral_asymmetry(soft: float, hard: float) -> float:
    """Soft-minus-hard rank gap at a single measurement point."""
    return float(soft) - float(hard)


def scaling_asymmetry(beta_soft: float, beta_hard: float) -> float:
    """Soft-minus-hard scaling-exponent gap (delta-beta)."""
    return float(beta_soft) - float(beta_hard)
