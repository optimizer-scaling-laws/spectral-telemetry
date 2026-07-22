"""spectral-telemetry public facade.

Everything importable from one place::

    from spectral_telemetry.telemetry import spectral_rank, attach_probes

Numpy-only names import eagerly; torch-backed names load lazily so the facade
works in environments without torch (you just cannot touch the torch names).
"""

from __future__ import annotations

# --- numpy-only core (always available) -------------------------------------
from spectral_telemetry.core.fits import fit_power_law, fit_power_law_with_ci
from spectral_telemetry.core.frequency import BUCKET_NAMES, FrequencyTable, load_frequency_vector
from spectral_telemetry.core.ranks import (
    DEFAULT_ESTIMATION_THRESHOLDS,
    EPS,
    attach_rank_fractions,
    estimation_diagnostics,
    validate_covariance,
    eigs_from_covariance,
    hard_rank,
    normalize_eigs,
    rank_metrics,
    rank_metrics_from_activations,
    renyi_rank,
    scaling_asymmetry,
    soft_rank,
    spectral_asymmetry,
    spectral_entropy,
)
from spectral_telemetry.core.schema import (
    CANONICAL_METRICS,
    LEGACY_KEY_MAP,
    parse_layer_metric_line,
    parse_metric_pairs,
)

_TORCH_NAMES = {
    "StreamingCovariance": "spectral_telemetry.torch_backend.streaming",
    "spectral_rank": "spectral_telemetry.torch_backend.probe",
    "attach_probes": "spectral_telemetry.torch_backend.probe",
    "ProbeSet": "spectral_telemetry.torch_backend.probe",
    "ModuleProbe": "spectral_telemetry.torch_backend.probe",
    "ensure_full_dim_activation": "spectral_telemetry.torch_backend.guards",
    "is_dtensor": "spectral_telemetry.torch_backend.guards",
    "merge_states": "spectral_telemetry.torch_backend.distributed",
    "all_gather_merge": "spectral_telemetry.torch_backend.distributed",
    "all_reduce_merge": "spectral_telemetry.torch_backend.distributed",
}

__all__ = [
    "EPS",
    "DEFAULT_ESTIMATION_THRESHOLDS",
    "estimation_diagnostics",
    "attach_rank_fractions",
    "validate_covariance",
    "normalize_eigs",
    "spectral_entropy",
    "soft_rank",
    "hard_rank",
    "renyi_rank",
    "rank_metrics",
    "eigs_from_covariance",
    "rank_metrics_from_activations",
    "spectral_asymmetry",
    "scaling_asymmetry",
    "fit_power_law",
    "fit_power_law_with_ci",
    "FrequencyTable",
    "BUCKET_NAMES",
    "load_frequency_vector",
    "CANONICAL_METRICS",
    "LEGACY_KEY_MAP",
    "parse_metric_pairs",
    "parse_layer_metric_line",
    *sorted(_TORCH_NAMES),
]


def __getattr__(name: str):
    if name in _TORCH_NAMES:
        import importlib

        module = importlib.import_module(_TORCH_NAMES[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'spectral_telemetry.telemetry' has no attribute {name!r}")
