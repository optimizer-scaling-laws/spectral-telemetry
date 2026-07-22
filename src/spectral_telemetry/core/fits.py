"""Log-log power-law fits with OLS t-interval CIs.

Port of upstream ``optimizer_ssl.analysis.scaling_fits`` (MIT), kept
scipy-free: the two-sided 97.5% t-critical values are tabulated for small
degrees of freedom and fall back to the normal quantile above dof 30.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

__all__ = ["fit_power_law_with_ci", "fit_power_law"]

_T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
    40: 2.021,
    60: 2.000,
    120: 1.980,
}


def _to_float(value: Any) -> float:
    if value in (None, ""):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _t_critical_975(dof: int) -> float:
    """Two-sided 97.5% Student-t critical value, conservative between rows.

    For untabulated degrees of freedom, the value at the next *smaller*
    tabulated dof is used -- a larger critical value, hence a slightly wider
    (conservative) interval. Above dof 120 this fixes t at 1.980 rather than
    the asymptotic 1.960, again on the conservative side.
    """
    if dof <= 0:
        return float("inf")
    if dof in _T_CRITICAL_975:
        return _T_CRITICAL_975[dof]
    return _T_CRITICAL_975[max(k for k in _T_CRITICAL_975 if k <= dof)]


def fit_power_law_with_ci(
    x_values: Iterable[Any],
    y_values: Iterable[Any],
    min_points: int = 3,
) -> dict[str, Any]:
    """Fit ``y = A * x^beta`` in log-log space; returns upstream-schema dict."""
    if min_points < 2:
        raise ValueError("min_points must be >= 2")
    x = np.asarray([_to_float(v) for v in x_values], dtype=float)
    y = np.asarray([_to_float(v) for v in y_values], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    n = int(mask.sum())
    if n < min_points:
        return {
            "valid": False,
            "reason": f"only {n} finite positive points (need >= {min_points})",
        }
    if np.unique(x[mask]).size < 2:
        return {"valid": False, "reason": "fewer than two unique positive x values"}

    lx, ly = np.log(x[mask]), np.log(y[mask])
    slope, intercept = np.polyfit(lx, ly, deg=1)
    pred = slope * lx + intercept
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - float(np.mean(ly))) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    dof = n - 2
    if dof > 0:
        s_err = float(np.sqrt(ss_res / dof))
        sxx = float(np.sum((lx - float(np.mean(lx))) ** 2))
        if sxx <= 0:  # pragma: no cover - guarded by the unique-x check above
            return {"valid": False, "reason": "zero variance in log-x"}
        std_err = s_err / np.sqrt(sxx)
        ci_half = _t_critical_975(dof) * std_err
    else:
        ci_half = float("inf")

    return {
        "valid": True,
        "beta": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "beta_lower": float(slope - ci_half),
        "beta_upper": float(slope + ci_half),
        "n_points": n,
        "ci_method": "ols_loglog_t_interval",
    }


def fit_power_law(x: Iterable[Any], y: Iterable[Any]) -> dict[str, float]:
    """Compact-key compatibility wrapper (beta, intercept, r2)."""
    fit = fit_power_law_with_ci(x, y, min_points=2)
    if not fit.get("valid"):
        raise ValueError("Need at least two positive points for a power-law fit")
    return {"beta": fit["beta"], "intercept": fit["intercept"], "r2": fit["r_squared"]}
