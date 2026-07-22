"""Streaming covariance accumulation with exact, order-stable merging.

This is the piece the upstream paper repo does not have: covariance from an
unbounded activation stream in O(D^2) memory, via the parallel/pairwise update
of Chan, Golub & LeVeque (1983). Upstream buffers every captured activation
tensor and concatenates before computing covariance (O(N*D) memory); this
accumulator is numerically equivalent (see tests) without the buffering.

The same ``merge`` operation serves three roles:

1. chunked capture across micro-batches (replaces activation buffering),
2. HEAD/MID/TAIL bucket accumulation (one accumulator per bucket, merged or
   reported separately),
3. cross-rank DDP reduction (gather ``state()`` from all ranks and merge) --
   this reproduces exactly the "between-means correction" reduction in the
   upstream tracker, generalized to arbitrarily many partial streams.

State is three tensors: ``n`` (samples), ``mean`` [D], and ``m2`` [D, D], the
centered scatter matrix, so ``cov = m2 / (n - ddof)``.
"""

from __future__ import annotations

import numpy as np
import torch

from spectral_telemetry.core import ranks as _ranks

__all__ = ["StreamingCovariance"]


def _negative_rtol_for(dtype: torch.dtype) -> float:
    """Materially-negative eigenvalue tolerance, scaled to accumulation precision.

    float64 accumulation keeps covariances PSD to ~1e-12 relative, so 1e-6 is a
    loud-corruption threshold. Lower-precision accumulation (float32/bf16) has
    an expected rounding floor around 1e-5..1e-4 relative on long streams;
    1e-3 still catches real corruption while admitting that floor. The raw
    minimum eigenvalue is reported in every record either way.
    """
    return 1e-6 if dtype == torch.float64 else 1e-3


class StreamingCovariance:
    """Numerically stable streaming covariance over the last tensor dimension.

    Not thread-safe: guard concurrent ``update``/``merge`` calls externally.
    """

    def __init__(
        self,
        dim: int,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
        check_finite: bool = True,
    ) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.dim = int(dim)
        self.n = 0
        self.check_finite = check_finite
        self.mean = torch.zeros(dim, dtype=dtype, device=device)
        self.m2 = torch.zeros(dim, dim, dtype=dtype, device=device)

    # ------------------------------------------------------------------ update
    def update(self, activations: torch.Tensor) -> "StreamingCovariance":
        """Fold a batch of activations ``[..., dim]`` into the running stats."""
        x = activations.detach()
        if x.ndim < 1 or x.shape[-1] != self.dim:
            raise ValueError(f"expected trailing dim {self.dim}, got shape {tuple(x.shape)}")
        x = x.reshape(-1, self.dim)
        m = int(x.shape[0])
        if m == 0:
            return self
        if self.check_finite and not torch.isfinite(x).all():
            raise ValueError(
                f"non-finite values in activation batch (shape {tuple(x.shape)}); "
                "fail-near-source policy -- pass check_finite=False to disable"
            )
        x = x.to(device=self.mean.device, dtype=self.mean.dtype)

        batch_mean = x.mean(dim=0)
        centered = x - batch_mean
        batch_m2 = centered.t() @ centered

        if self.n == 0:
            self.mean.copy_(batch_mean)
            self.m2.copy_(batch_m2)
            self.n = m
            return self

        total = self.n + m
        delta = batch_mean - self.mean
        self.m2 += batch_m2 + torch.outer(delta, delta) * (self.n * m / total)
        self.mean += delta * (m / total)
        self.n = total
        return self

    # ------------------------------------------------------------------- merge
    def merge(self, other: "StreamingCovariance") -> "StreamingCovariance":
        """Fold another accumulator into this one (exact, in place)."""
        if other.dim != self.dim:
            raise ValueError(f"dim mismatch: {self.dim} vs {other.dim}")
        if other.n == 0:
            return self
        o_mean = other.mean.to(device=self.mean.device, dtype=self.mean.dtype)
        o_m2 = other.m2.to(device=self.m2.device, dtype=self.m2.dtype)
        if self.n == 0:
            self.mean.copy_(o_mean)
            self.m2.copy_(o_m2)
            self.n = other.n
            return self

        total = self.n + other.n
        delta = o_mean - self.mean
        self.m2 += o_m2 + torch.outer(delta, delta) * (self.n * other.n / total)
        self.mean += delta * (other.n / total)
        self.n = total
        return self

    @classmethod
    def merged(cls, accumulators: list["StreamingCovariance"]) -> "StreamingCovariance":
        if not accumulators:
            raise ValueError("cannot merge an empty list of accumulators")
        out = accumulators[0].clone()
        for acc in accumulators[1:]:
            out.merge(acc)
        return out

    # ------------------------------------------------------------------ export
    def covariance(self, ddof: int = 1) -> torch.Tensor:
        if self.n < ddof + 1:
            raise ValueError(f"need at least {ddof + 1} samples for covariance (have {self.n})")
        return self.m2 / (self.n - ddof)

    def eigenvalues(self) -> np.ndarray:
        """Descending, eps-clamped covariance eigenvalues (float64 numpy)."""
        cov = self.covariance().to("cpu", torch.float64).numpy()
        return _ranks.eigs_from_covariance(cov)

    def metrics(self, *, estimation_thresholds=None) -> dict:
        """Paper-facing metrics dict with estimation and numeric diagnostics.

        Beyond upstream-compatible ``soft_rank``/``hard_rank``/
        ``spectral_entropy``/``n_samples``/``hidden_dim``, records include
        ``sample_to_dim_ratio``, ``max_empirical_rank = min(D, N-1)``,
        ``estimation_status`` (configurable heuristic thresholds), rank
        fractions of ``D`` and of the empirical maximum, and eigenvalue
        diagnostics (``min_raw_eigenvalue``, ``negative_eigenvalue_mass``).
        """
        cov = self.covariance().to("cpu", torch.float64).numpy()
        eigs, eig_diag = _ranks.eigs_from_covariance(
            cov,
            materially_negative_rtol=_negative_rtol_for(self.mean.dtype),
            return_diagnostics=True,
        )
        out = _ranks.rank_metrics(eigs)
        out["n_samples"] = int(self.n)
        out["hidden_dim"] = int(self.dim)
        out.update(eig_diag)
        out.update(
            _ranks.estimation_diagnostics(self.n, self.dim, thresholds=estimation_thresholds)
        )
        return _ranks.attach_rank_fractions(out)

    # ------------------------------------------------------------- state / etc
    def state(self) -> tuple[int, torch.Tensor, torch.Tensor]:
        return self.n, self.mean.clone(), self.m2.clone()

    @classmethod
    def from_state(
        cls, n: int, mean: torch.Tensor, m2: torch.Tensor, *, check_finite: bool = True
    ) -> "StreamingCovariance":
        if int(n) < 0:
            raise ValueError(f"sample count must be non-negative, got {n}")
        if mean.dim() != 1:
            raise ValueError(f"mean must be 1-D, got shape {tuple(mean.shape)}")
        if m2.shape != (mean.numel(), mean.numel()):
            raise ValueError(
                f"m2 must be square {mean.numel()}x{mean.numel()}, got {tuple(m2.shape)}"
            )
        if check_finite and not (torch.isfinite(mean).all() and torch.isfinite(m2).all()):
            raise ValueError("state contains non-finite values")
        acc = cls(mean.numel(), dtype=mean.dtype, device=mean.device, check_finite=check_finite)
        acc.n = int(n)
        acc.mean.copy_(mean)
        acc.m2.copy_(m2)
        return acc

    def clone(self) -> "StreamingCovariance":
        return StreamingCovariance.from_state(*self.state(), check_finite=self.check_finite)

    def reset(self) -> None:
        self.n = 0
        self.mean.zero_()
        self.m2.zero_()

    def __repr__(self) -> str:  # pragma: no cover
        return f"StreamingCovariance(dim={self.dim}, n={self.n})"
