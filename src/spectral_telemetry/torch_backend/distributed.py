"""Cross-rank reduction of streaming covariance state.

Two exact reductions are provided:

* :func:`all_reduce_merge` (preferred) -- two collective phases: (1)
  all-reduce ``n`` and ``n * mean`` to obtain the global mean, then (2)
  all-reduce each rank's *corrected scatter* ``M2_r + n_r (mu_r - mu)(mu_r -
  mu)^T``. Peak extra memory is one D x D tensor per probe regardless of
  world size (law of total covariance; algebraically exact).
* :func:`all_gather_merge` (compatibility / debugging) -- gathers every
  rank's full state and merges pairwise. Transiently holds ``world_size``
  copies of each D x D matrix; prefer :func:`all_reduce_merge` at scale.

Communication device is chosen from the process-group backend: Gloo/MPI
communicate CPU tensors directly; NCCL groups stage tensors on the current
CUDA device and move the merged result back to the accumulator's device.
Only data-parallel reduction is supported: every rank must hold
full-hidden-dimension statistics (see ``guards.ensure_full_dim_activation``).
"""

from __future__ import annotations

import torch

from spectral_telemetry.torch_backend.streaming import StreamingCovariance

__all__ = ["merge_states", "all_gather_merge", "all_reduce_merge", "gather_object_list"]


def _dist():
    import torch.distributed as dist

    return dist


def _is_active(group) -> bool:
    dist = _dist()
    return dist.is_available() and dist.is_initialized() and dist.get_world_size(group=group) > 1


def _comm_device(group) -> torch.device:
    """Pick a tensor device valid for collectives on this group's backend."""
    dist = _dist()
    backend = str(dist.get_backend(group)).lower()
    if "gloo" in backend or "mpi" in backend:
        return torch.device("cpu")
    if "nccl" in backend:
        if not torch.cuda.is_available():  # pragma: no cover
            raise RuntimeError(
                "process group backend is NCCL but CUDA is unavailable; "
                "use a Gloo group for CPU covariance states"
            )
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def merge_states(states) -> StreamingCovariance:
    """Merge an iterable of ``(n, mean, m2)`` states into one accumulator."""
    accs = [StreamingCovariance.from_state(n, mean, m2) for n, mean, m2 in states]
    return StreamingCovariance.merged(accs)


def gather_object_list(obj, group=None) -> list:
    """All-gather an arbitrary picklable object from every rank (ws-1 safe)."""
    if not _is_active(group):
        return [obj]
    dist = _dist()
    out = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(out, obj, group=group)
    return out


def all_reduce_merge(acc: StreamingCovariance, group=None) -> StreamingCovariance:
    """Exact global covariance state via two all-reduces (preferred path).

    Safe for empty local shards (``acc.n == 0``). Falls back to a clone when
    torch.distributed is unavailable/uninitialized or world size is 1.
    """
    if not _is_active(group):
        return acc.clone()
    dist = _dist()
    dev = _comm_device(group)
    dt = acc.mean.dtype
    home = acc.mean.device

    # Phase 1: one fused all-reduce carrying [n, n * mean].
    vec = torch.empty(acc.dim + 1, dtype=dt, device=dev)
    vec[0] = float(acc.n)
    vec[1:] = (acc.mean * float(acc.n)).to(dev)
    dist.all_reduce(vec, op=dist.ReduceOp.SUM, group=group)
    n_global = int(round(float(vec[0].item())))
    if n_global == 0:
        return StreamingCovariance(acc.dim, dtype=dt, device=home)
    global_mean = vec[1:] / float(n_global)

    # Phase 2: all-reduce the between-means-corrected scatter.
    delta = acc.mean.to(dev) - global_mean
    corrected = acc.m2.to(dev) + float(acc.n) * torch.outer(delta, delta)
    dist.all_reduce(corrected, op=dist.ReduceOp.SUM, group=group)

    return StreamingCovariance.from_state(n_global, global_mean.to(home), corrected.to(home))


def all_gather_merge(acc: StreamingCovariance, group=None) -> StreamingCovariance:
    """Gather every rank's state and merge (compat path; O(world_size * D^2))."""
    if not _is_active(group):
        return acc.clone()
    dist = _dist()
    dev = _comm_device(group)
    home = acc.mean.device
    world_size = dist.get_world_size(group=group)

    n_local = torch.tensor([float(acc.n)], dtype=acc.mean.dtype, device=dev)
    mean_local = acc.mean.to(dev)
    m2_local = acc.m2.to(dev)

    n_all = [torch.zeros_like(n_local) for _ in range(world_size)]
    mean_all = [torch.zeros_like(mean_local) for _ in range(world_size)]
    m2_all = [torch.zeros_like(m2_local) for _ in range(world_size)]
    dist.all_gather(n_all, n_local, group=group)
    dist.all_gather(mean_all, mean_local, group=group)
    dist.all_gather(m2_all, m2_local, group=group)

    return merge_states(
        (int(round(float(n.item()))), mean.to(home), m2.to(home))
        for n, mean, m2 in zip(n_all, mean_all, m2_all)
    )
