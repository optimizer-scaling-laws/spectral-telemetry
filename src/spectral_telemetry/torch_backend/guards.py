"""Parallelism-safety guards for activation capture.

Spectral covariance telemetry is only meaningful when the captured activation
holds the FULL hidden dimension on this rank (data-parallel sharding). Under
tensor parallelism, FFN projections are column-sharded, so each rank sees a
slice of the hidden dimension; treating per-rank slices as full-dimension
covariances yields silently wrong spectra. Until the DTensor-aware covariance
path lands (roadmap), sharded activations are a hard error.
"""

from __future__ import annotations

__all__ = ["ensure_full_dim_activation", "is_dtensor"]

try:  # torch >= 2.4
    from torch.distributed.tensor import DTensor as _DTensor
except Exception:  # pragma: no cover
    try:
        from torch.distributed._tensor import DTensor as _DTensor
    except Exception:
        _DTensor = None


def is_dtensor(tensor) -> bool:
    return _DTensor is not None and isinstance(tensor, _DTensor)


def ensure_full_dim_activation(tensor, context: str = "capture") -> None:
    """Raise ``NotImplementedError`` on DTensor (tensor-parallel) activations."""
    if is_dtensor(tensor):
        raise NotImplementedError(
            f"{context}: got a DTensor activation. spectral-telemetry requires each "
            "rank to observe the full hidden dimension (data-parallel sharding only). "
            "A TP-aware covariance path is on the roadmap; for now run probes with "
            "tp_size=1 or gather activations to full dimension before capture."
        )
