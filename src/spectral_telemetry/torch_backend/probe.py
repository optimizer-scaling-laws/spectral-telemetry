"""Model-agnostic spectral probes.

Two entry points:

* :func:`spectral_rank` -- one-shot metrics for an activation tensor, with
  optional sample masking and HEAD/MID/TAIL bucketing. Output keys are
  compatible with the upstream paper repo's ``optimizer_ssl.probe``.
* :func:`attach_probes` -- hook selected modules of any ``nn.Module`` and
  accumulate *streaming* covariance (O(D^2) memory per accumulator) across
  forward passes, optionally per frequency bucket and under a sample mask.

Batch-context lifecycle
-----------------------
Token IDs and sample masks are supplied per forward pass through a
single-use batch context::

    with probes.batch_context(token_ids=input_ids, sample_mask=attention_mask):
        model(input_ids)

or equivalently ``probes.set_batch_context(...)`` immediately before each
forward. The context is armed for exactly the next forward of the model
object passed to :func:`attach_probes`: a pre-forward hook on that root
module activates it and a post-forward hook clears it, so a context can
never be silently reused for a later batch. Bucketed probes raise if a
forward arrives without a fresh context. Call the *root* model -- invoking a
probed submodule directly bypasses the context hooks (use
:func:`spectral_rank` for ad-hoc single-module measurements).
"""

from __future__ import annotations

import contextlib
import fnmatch
from typing import Callable, Iterable, Literal, Optional, Sequence, Union

import numpy as np
import torch

from spectral_telemetry.core.frequency import BUCKET_NAMES, FrequencyTable
from spectral_telemetry.core.ranks import estimation_diagnostics
from spectral_telemetry.torch_backend.guards import ensure_full_dim_activation
from spectral_telemetry.torch_backend.streaming import StreamingCovariance

CaptureKind = Literal["input", "output"]
Selector = Callable[[str, torch.nn.Module], bool]
ArrayLike = Union[torch.Tensor, np.ndarray, Sequence[int]]

__all__ = ["spectral_rank", "attach_probes", "ProbeSet", "ModuleProbe"]


# ------------------------------------------------------------------ conversion
def _flat_ids(token_ids: ArrayLike) -> np.ndarray:
    """Token IDs (torch / numpy / sequence) -> flat int64 numpy array."""
    if isinstance(token_ids, torch.Tensor):
        arr = token_ids.detach().cpu().numpy()
    else:
        arr = np.asarray(token_ids)
    if arr.size and not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"token IDs must be integers, got dtype {arr.dtype}")
    return arr.reshape(-1).astype(np.int64)


def _flat_bool_mask(mask: ArrayLike) -> np.ndarray:
    """Sample mask (torch / numpy / sequence) -> flat boolean numpy array."""
    if isinstance(mask, torch.Tensor):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)
    if arr.dtype != np.bool_:
        if np.issubdtype(arr.dtype, np.number):
            arr = arr != 0  # accept 0/1 attention masks
        else:
            raise ValueError(f"sample mask must be boolean or 0/1, got dtype {arr.dtype}")
    return arr.reshape(-1)


def _flatten(activations: torch.Tensor) -> torch.Tensor:
    if activations.dim() < 2:
        raise ValueError(f"expected at least 2D activations, got shape {tuple(activations.shape)}")
    return activations.detach().reshape(-1, activations.shape[-1])


# --------------------------------------------------------------------- one-shot
def spectral_rank(
    activations: torch.Tensor,
    *,
    token_ids: Optional[ArrayLike] = None,
    sample_mask: Optional[ArrayLike] = None,
    freq_table: FrequencyTable | str | None = None,
    min_samples_per_bucket: int = 2,
) -> dict:
    """Soft/hard spectral-rank diagnostics for arbitrary activations.

    ``sample_mask`` (e.g. an attention mask) selects which flattened rows
    contribute; it is applied identically to activations and ``token_ids``
    before any bucketing. With ``token_ids`` + ``freq_table``, per-bucket
    metrics are returned under ``"frequency_buckets"``.
    """
    ensure_full_dim_activation(activations, context="spectral_rank")
    x2d = _flatten(activations)
    n_rows = x2d.shape[0]

    tokens: Optional[np.ndarray] = None
    if token_ids is not None:
        tokens = _flat_ids(token_ids)
        if tokens.size != n_rows:
            raise ValueError(
                f"token_ids has {tokens.size} entries but activations have {n_rows} rows"
            )

    if sample_mask is not None:
        mask_np = _flat_bool_mask(sample_mask)
        if mask_np.size != n_rows:
            raise ValueError(
                f"sample_mask has {mask_np.size} entries but activations have {n_rows} rows"
            )
        keep = torch.as_tensor(mask_np, device=x2d.device)
        x2d = x2d[keep]
        if tokens is not None:
            tokens = tokens[mask_np]

    if x2d.shape[0] < 2:
        raise ValueError("at least two unmasked activation samples are required")

    out = StreamingCovariance(x2d.shape[-1]).update(x2d).metrics()

    if tokens is None and freq_table is None:
        return out
    if tokens is None or freq_table is None:
        raise ValueError("bucketed metrics require both token_ids and freq_table")
    if isinstance(freq_table, str):
        freq_table = FrequencyTable.from_file(freq_table)

    buckets = freq_table.get_bucket(tokens)
    bucket_metrics: dict[str, dict] = {}
    for bucket_idx, bucket_name in enumerate(BUCKET_NAMES):
        sel_np = buckets == bucket_idx
        n = int(sel_np.sum())
        if n < max(min_samples_per_bucket, 2):
            entry = {
                "status": "insufficient_samples",
                "n_samples": n,
                "hidden_dim": int(x2d.shape[-1]),
            }
            entry.update(estimation_diagnostics(n, int(x2d.shape[-1])))
            bucket_metrics[bucket_name] = entry
        else:
            sel = torch.as_tensor(sel_np, device=x2d.device)
            bucket_metrics[bucket_name] = (
                StreamingCovariance(x2d.shape[-1]).update(x2d[sel]).metrics()
            )
    out["frequency_buckets"] = bucket_metrics
    return out


# -------------------------------------------------------------------- streaming
class _BatchContext:
    __slots__ = ("tokens", "mask")

    def __init__(self, tokens: Optional[np.ndarray], mask: Optional[np.ndarray]) -> None:
        self.tokens = tokens
        self.mask = mask


class ModuleProbe:
    """Streaming spectral accumulation for a single module's input or output."""

    def __init__(
        self,
        name: str,
        module: torch.nn.Module,
        *,
        capture: CaptureKind = "output",
        dtype: torch.dtype = torch.float64,
        accumulation_device: Union[str, torch.device] = "cpu",
        freq_table: FrequencyTable | None = None,
        expected_hidden_dim: Optional[int] = None,
        tensor_extractor: Optional[Callable] = None,
        owner: "ProbeSet | None" = None,
    ) -> None:
        if capture not in ("input", "output"):
            raise ValueError("capture must be 'input' or 'output'")
        self.name = name
        self.module = module
        self.capture = capture
        self.dtype = dtype
        self.accumulation_device = torch.device(accumulation_device)
        self.freq_table = freq_table
        self.expected_hidden_dim = expected_hidden_dim
        self.tensor_extractor = tensor_extractor
        self._owner = owner
        self.pooled: StreamingCovariance | None = None
        self.buckets: dict[str, StreamingCovariance] = {}
        if capture == "output":
            self._handle = module.register_forward_hook(self._on_output)
        else:
            self._handle = module.register_forward_pre_hook(self._on_input)

    # hooks ------------------------------------------------------------------
    def _extract(self, module, inputs, output) -> torch.Tensor:
        if self.tensor_extractor is not None:
            tensor = self.tensor_extractor(module, inputs, output)
        elif output is not None:
            tensor = output[0] if isinstance(output, tuple) else output
        else:
            tensor = inputs[0]
        if not torch.is_tensor(tensor):
            raise TypeError(
                f"probe[{self.name}]: extracted object is {type(tensor).__name__}, "
                "not a tensor; provide tensor_extractor=(module, inputs, output) -> Tensor"
            )
        if not tensor.is_floating_point():
            raise TypeError(
                f"probe[{self.name}]: extracted tensor has non-floating dtype {tensor.dtype}"
            )
        if self.expected_hidden_dim is not None and tensor.shape[-1] != self.expected_hidden_dim:
            raise ValueError(
                f"probe[{self.name}]: captured trailing dimension {tensor.shape[-1]} "
                f"!= expected_hidden_dim {self.expected_hidden_dim} -- possible sharded "
                "or wrong-module capture"
            )
        return tensor

    def _on_output(self, module, inputs, output):
        if self._owner is not None and not self._owner.enabled:
            return
        self._ingest(self._extract(module, inputs, output))

    def _on_input(self, module, inputs):
        if self._owner is not None and not self._owner.enabled:
            return
        self._ingest(self._extract(module, inputs, None))

    def _ingest(self, tensor: torch.Tensor) -> None:
        ensure_full_dim_activation(tensor, context=f"probe[{self.name}]")
        flat = _flatten(tensor)
        n_rows = flat.shape[0]

        ctx = self._owner._active_context if self._owner is not None else None
        tokens: Optional[np.ndarray] = None

        if self.freq_table is not None:
            if ctx is None or ctx.tokens is None:
                raise RuntimeError(
                    f"probe[{self.name}]: frequency bucketing is enabled but this forward "
                    "has no fresh batch context with token IDs. Arm one per forward with "
                    "ProbeSet.set_batch_context(token_ids=...) or "
                    "`with probes.batch_context(token_ids=...):` -- contexts are "
                    "single-use and cleared after each forward of the probed model."
                )
            tokens = ctx.tokens
            if tokens.size != n_rows:
                raise ValueError(
                    f"probe[{self.name}]: {tokens.size} token IDs vs {n_rows} activation "
                    "rows; shapes must align (e.g. [B, T] tokens with [B, T, D] activations)."
                )

        mask_np = ctx.mask if ctx is not None else None
        if mask_np is not None:
            if mask_np.size != n_rows:
                raise ValueError(
                    f"probe[{self.name}]: sample_mask has {mask_np.size} entries vs "
                    f"{n_rows} activation rows"
                )
            keep = torch.as_tensor(mask_np, device=flat.device)
            flat = flat[keep]
            if tokens is not None:
                tokens = tokens[mask_np]

        if flat.shape[0] == 0:
            return  # fully masked forward contributes nothing

        cap = self._owner.max_tokens_per_forward if self._owner is not None else None
        if cap is not None and flat.shape[0] > cap:
            idx = self._owner._sampling_rng.choice(flat.shape[0], size=cap, replace=False)
            idx.sort()
            flat = flat[torch.as_tensor(idx, device=flat.device)]
            if tokens is not None:
                tokens = tokens[idx]

        x2d = flat.to(self.accumulation_device, self.dtype)
        if self.pooled is None:
            self.pooled = StreamingCovariance(
                x2d.shape[-1], dtype=self.dtype, device=self.accumulation_device
            )
        self._update_checked(self.pooled, x2d)

        if self.freq_table is None:
            return
        bucket_ids = self.freq_table.get_bucket(tokens)
        for bucket_idx, bucket_name in enumerate(BUCKET_NAMES):
            sel_np = bucket_ids == bucket_idx
            n = int(sel_np.sum())
            if n == 0:
                continue
            if bucket_name not in self.buckets:
                self.buckets[bucket_name] = StreamingCovariance(
                    x2d.shape[-1], dtype=self.dtype, device=self.accumulation_device
                )
            self._update_checked(self.buckets[bucket_name], x2d[torch.as_tensor(sel_np)])

    def _update_checked(self, acc: StreamingCovariance, x2d: torch.Tensor) -> None:
        try:
            acc.update(x2d)
        except ValueError as e:
            raise ValueError(f"probe[{self.name}]: {e}") from None

    # export -----------------------------------------------------------------
    def _record(
        self,
        pooled: StreamingCovariance | None,
        buckets: dict[str, StreamingCovariance],
        *,
        min_samples_per_bucket: int = 2,
    ) -> dict:
        if pooled is None or pooled.n < 2:
            raise RuntimeError(
                f"probe[{self.name}]: no activations recorded; run a forward pass first"
            )
        out = pooled.metrics()
        if self.freq_table is not None:
            bucket_metrics: dict[str, dict] = {}
            for bucket_name in BUCKET_NAMES:
                acc = buckets.get(bucket_name)
                n = 0 if acc is None else acc.n
                if acc is None or n < max(min_samples_per_bucket, 2):
                    entry = {
                        "status": "insufficient_samples",
                        "n_samples": int(n),
                        "hidden_dim": int(pooled.dim),
                    }
                    entry.update(estimation_diagnostics(n, pooled.dim))
                    bucket_metrics[bucket_name] = entry
                else:
                    bucket_metrics[bucket_name] = acc.metrics()
            out["frequency_buckets"] = bucket_metrics
        out["accumulation_device"] = str(self.accumulation_device)
        out["accumulation_dtype"] = str(self.dtype)
        if self._owner is not None:
            out["n_forwards"] = self._owner.n_forwards
            if self._owner.max_tokens_per_forward is not None:
                out["max_tokens_per_forward"] = self._owner.max_tokens_per_forward
        return out

    def compute(self, *, min_samples_per_bucket: int = 2) -> dict:
        return self._record(
            self.pooled, self.buckets, min_samples_per_bucket=min_samples_per_bucket
        )

    def reset(self) -> None:
        self.pooled = None
        self.buckets = {}

    def close(self) -> None:
        self._handle.remove()


class ProbeSet:
    """Probes sharing a per-forward batch context (token IDs + sample mask)."""

    def __init__(
        self,
        probes: list[ModuleProbe],
        model: torch.nn.Module,
        *,
        max_tokens_per_forward: Optional[int] = None,
        sampling_seed: int = 0,
    ) -> None:
        if not probes:
            raise ValueError("selector matched no modules")
        self.probes = probes
        self._model = model
        self.enabled = True
        self.is_closed = False
        self.n_forwards = 0
        self.max_tokens_per_forward = max_tokens_per_forward
        self._sampling_rng = np.random.default_rng(sampling_seed)
        self._sampling_seed = sampling_seed
        self._pending_context: _BatchContext | None = None
        self._active_context: _BatchContext | None = None
        for p in self.probes:
            p._owner = self
        # Root hooks manage the context lifecycle. prepend=True guarantees the
        # pre-hook runs before any probe pre-hook on the same (root) module;
        # the post-hook is registered after all probe hooks so it clears last.
        self._root_pre = model.register_forward_pre_hook(self._on_root_pre, prepend=True)
        self._root_post = model.register_forward_hook(self._on_root_post)

    # context lifecycle ------------------------------------------------------
    def _on_root_pre(self, module, args):
        if not self.enabled:
            self._active_context = None  # keep pending armed for when re-enabled
            return
        self._active_context = self._pending_context
        self._pending_context = None

    def _on_root_post(self, module, args, output):
        if self.enabled:
            self.n_forwards += 1
        self._active_context = None

    def enable(self) -> None:
        """Resume accumulation (hooks stay registered while disabled)."""
        self.enabled = True

    def disable(self) -> None:
        """Pause accumulation without removing hooks."""
        self.enabled = False

    def set_batch_context(
        self,
        *,
        token_ids: Optional[ArrayLike] = None,
        sample_mask: Optional[ArrayLike] = None,
    ) -> None:
        """Arm token IDs / sample mask for exactly the next forward pass."""
        tokens = _flat_ids(token_ids) if token_ids is not None else None
        mask = _flat_bool_mask(sample_mask) if sample_mask is not None else None
        if tokens is not None and mask is not None and tokens.size != mask.size:
            raise ValueError(
                f"token_ids ({tokens.size}) and sample_mask ({mask.size}) must describe "
                "the same positions"
            )
        self._pending_context = _BatchContext(tokens, mask)

    @contextlib.contextmanager
    def batch_context(
        self,
        *,
        token_ids: Optional[ArrayLike] = None,
        sample_mask: Optional[ArrayLike] = None,
    ):
        """Context-manager form of :meth:`set_batch_context` (one forward)."""
        self.set_batch_context(token_ids=token_ids, sample_mask=sample_mask)
        try:
            yield self
        finally:
            self._pending_context = None
            self._active_context = None

    # aggregation ------------------------------------------------------------
    def compute(
        self,
        *,
        min_samples_per_bucket: int = 2,
        distributed: bool = False,
        group=None,
        reset: bool = False,
    ) -> dict[str, dict]:
        """Metrics for every probe; optionally reduced exactly across ranks.

        With ``distributed=True``, probe names/dimensions/bucketing are
        validated to be identical on every rank, then pooled and per-bucket
        states are reduced with the two-phase all-reduce (collectives are
        issued in identical order on every rank; ranks missing a bucket
        participate with an empty state). ``reset=True`` clears accumulators,
        contexts, and the forwards counter after computing.
        """
        if distributed:
            out = self._compute_distributed(
                min_samples_per_bucket=min_samples_per_bucket, group=group
            )
        else:
            out = {
                p.name: p.compute(min_samples_per_bucket=min_samples_per_bucket)
                for p in self.probes
            }
        if reset:
            self.reset()
        return out

    def _compute_distributed(self, *, min_samples_per_bucket: int, group) -> dict[str, dict]:
        from spectral_telemetry.torch_backend.distributed import (
            all_reduce_merge,
            gather_object_list,
        )

        local_meta = [
            (p.name, -1 if p.pooled is None else p.pooled.dim, p.freq_table is not None)
            for p in self.probes
        ]
        all_meta = gather_object_list(local_meta, group=group)
        names = [[m[0] for m in meta] for meta in all_meta]
        if any(nm != names[0] for nm in names[1:]):
            raise RuntimeError(
                f"distributed compute requires identical probe names on every rank; got {names}"
            )
        buckety = [[m[2] for m in meta] for meta in all_meta]
        if any(b != buckety[0] for b in buckety[1:]):
            raise RuntimeError("probe bucketing configuration differs across ranks")
        dims: list[int] = []
        for i in range(len(self.probes)):
            seen = {meta[i][1] for meta in all_meta} - {-1}
            if not seen:
                raise RuntimeError(
                    f"probe[{self.probes[i].name}]: no activations recorded on any rank"
                )
            if len(seen) > 1:
                raise RuntimeError(
                    f"probe[{self.probes[i].name}]: hidden dimension differs across "
                    f"ranks: {sorted(seen)}"
                )
            dims.append(seen.pop())

        try:
            import torch.distributed as dist

            world = dist.get_world_size(group=group) if dist.is_initialized() else 1
        except Exception:  # pragma: no cover
            world = 1

        out: dict[str, dict] = {}
        for probe, dim in zip(self.probes, dims):

            def _empty() -> StreamingCovariance:
                return StreamingCovariance(dim, dtype=probe.dtype, device=probe.accumulation_device)

            pooled = all_reduce_merge(probe.pooled or _empty(), group=group)
            buckets: dict[str, StreamingCovariance] = {}
            if probe.freq_table is not None:
                for bucket_name in BUCKET_NAMES:  # identical order on all ranks
                    buckets[bucket_name] = all_reduce_merge(
                        probe.buckets.get(bucket_name) or _empty(), group=group
                    )
            record = probe._record(pooled, buckets, min_samples_per_bucket=min_samples_per_bucket)
            record["world_size"] = world
            out[probe.name] = record
        return out

    def reset(self) -> None:
        """Clear all accumulators and any pending/active batch context."""
        for p in self.probes:
            p.reset()
        self.n_forwards = 0
        self._pending_context = None
        self._active_context = None

    def close(self) -> None:
        """Remove all hooks; safe to call more than once."""
        if self.is_closed:
            return
        for p in self.probes:
            p.close()
        self._root_pre.remove()
        self._root_post.remove()
        self.is_closed = True

    def __enter__(self) -> "ProbeSet":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self.probes)


def _normalize_selector(select) -> Selector:
    if callable(select):
        return select
    if isinstance(select, str):
        pattern = select
        return lambda name, module: fnmatch.fnmatch(name, pattern) or pattern in name
    if isinstance(select, torch.nn.Module):
        target = select
        return lambda name, module: module is target
    if isinstance(select, Iterable):
        targets = list(select)
        return lambda name, module: module in targets
    raise TypeError(f"unsupported selector type: {type(select)!r}")


def attach_probes(
    model: torch.nn.Module,
    select,
    *,
    capture: CaptureKind = "output",
    dtype: torch.dtype = torch.float64,
    accumulation_device: Union[str, torch.device] = "cpu",
    freq_table: FrequencyTable | str | None = None,
    expected_hidden_dim: Optional[int] = None,
    tensor_extractor: Optional[Callable] = None,
    max_tokens_per_forward: Optional[int] = None,
    sampling_seed: int = 0,
) -> ProbeSet:
    """Attach streaming spectral probes to every module matching ``select``.

    ``select`` may be a callable ``(qualified_name, module) -> bool``, an
    fnmatch-style name pattern (substring also matches), a module instance, or
    an iterable of module instances. Token IDs and sample masks are supplied
    per forward via :meth:`ProbeSet.batch_context` /
    :meth:`ProbeSet.set_batch_context`.
    """
    if isinstance(freq_table, str):
        freq_table = FrequencyTable.from_file(freq_table)
    selector = _normalize_selector(select)
    probes = [
        ModuleProbe(
            name or module.__class__.__name__,
            module,
            capture=capture,
            dtype=dtype,
            accumulation_device=accumulation_device,
            freq_table=freq_table,
            expected_hidden_dim=expected_hidden_dim,
            tensor_extractor=tensor_extractor,
        )
        for name, module in model.named_modules()
        if selector(name, module)
    ]
    return ProbeSet(
        probes,
        model,
        max_tokens_per_forward=max_tokens_per_forward,
        sampling_seed=sampling_seed,
    )
