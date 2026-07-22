import pytest

torch = pytest.importorskip("torch")

from spectral_telemetry.torch_backend.distributed import all_gather_merge, merge_states
from spectral_telemetry.torch_backend.streaming import StreamingCovariance


def test_merge_states_equals_direct():
    torch.manual_seed(0)
    parts = [torch.randn(n, 6, dtype=torch.float64) for n in (30, 70, 11)]
    accs = [StreamingCovariance(6).update(p) for p in parts]
    merged = merge_states(a.state() for a in accs)
    full = StreamingCovariance(6).update(torch.cat(parts))
    assert merged.n == full.n
    assert torch.allclose(merged.covariance(), full.covariance(), atol=1e-10)


def test_all_gather_merge_without_dist_returns_clone():
    acc = StreamingCovariance(4).update(torch.randn(10, 4, dtype=torch.float64))
    out = all_gather_merge(acc)
    assert out is not acc and out.n == acc.n
    assert torch.allclose(out.covariance(), acc.covariance())


def test_all_gather_merge_single_process_gloo():
    dist = pytest.importorskip("torch.distributed")
    if not dist.is_available():
        pytest.skip("torch.distributed unavailable")
    if not dist.is_initialized():
        dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    acc = StreamingCovariance(4).update(torch.randn(12, 4, dtype=torch.float64))
    out = all_gather_merge(acc)
    assert out.n == acc.n
    assert torch.allclose(out.covariance(), acc.covariance(), atol=1e-12)


# ---- pass-2: two-process Gloo tests -----------------------------------------
import os
import tempfile

import numpy as np
import torch.multiprocessing as mp
import torch.nn as nn

from spectral_telemetry.core.frequency import FrequencyTable
from spectral_telemetry.torch_backend.distributed import all_reduce_merge
from spectral_telemetry.torch_backend.probe import attach_probes

WORLD = 2


def _init(rank, store_path):
    import torch.distributed as dist

    store = dist.FileStore(store_path, WORLD)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=WORLD)
    return dist


def _chunk(rank):
    g = torch.Generator().manual_seed(1000 + rank)
    n = 37 if rank == 0 else 81
    return torch.randn(n, 6, dtype=torch.float64, generator=g) + 2.0 * rank


def _worker_allreduce(rank, store_path):
    dist = _init(rank, store_path)
    try:
        local = StreamingCovariance(6).update(_chunk(rank))
        red = all_reduce_merge(local)
        full = StreamingCovariance(6).update(torch.cat([_chunk(0), _chunk(1)]))
        assert red.n == full.n == 118
        assert torch.allclose(red.mean, full.mean, atol=1e-12)
        assert torch.allclose(red.covariance(), full.covariance(), atol=1e-10)
        gat = all_gather_merge(local)
        assert torch.allclose(gat.covariance(), full.covariance(), atol=1e-10)
    finally:
        dist.destroy_process_group()


def _worker_empty_shard(rank, store_path):
    dist = _init(rank, store_path)
    try:
        acc = StreamingCovariance(5)
        if rank == 0:
            g = torch.Generator().manual_seed(7)
            acc.update(torch.randn(40, 5, dtype=torch.float64, generator=g))
        red = all_reduce_merge(acc)
        g = torch.Generator().manual_seed(7)
        ref = StreamingCovariance(5).update(torch.randn(40, 5, dtype=torch.float64, generator=g))
        assert red.n == 40
        assert torch.allclose(red.covariance(), ref.covariance(), atol=1e-10)
    finally:
        dist.destroy_process_group()


def _rank_ids(rank):
    return np.random.default_rng(10 + rank).integers(0, 50, size=20)


def _worker_probeset(rank, store_path):
    dist = _init(rank, store_path)
    try:
        ranks_ = np.arange(1, 51, dtype=np.float64)
        table = FrequencyTable(np.maximum((1e5 / ranks_).astype(np.int64), 1))
        torch.manual_seed(0)  # identical weights on both ranks
        emb = nn.Embedding(50, 6)
        probes = attach_probes(
            emb, select=lambda n, m: isinstance(m, nn.Embedding), freq_table=table
        )
        ids_np = _rank_ids(rank)
        ids = torch.as_tensor(ids_np.reshape(2, 10))
        with probes.batch_context(token_ids=ids):
            emb(ids)
        out = probes.compute(distributed=True, reset=True)
        rec = out["Embedding"]
        assert rec["world_size"] == 2 and rec["n_samples"] == 40

        both = np.concatenate([_rank_ids(0), _rank_ids(1)])
        ref_buckets = table.get_bucket(both)
        for i, name in enumerate(("head", "mid", "tail")):
            expect = int((ref_buckets == i).sum())
            got = rec["frequency_buckets"][name]["n_samples"]
            assert got == expect, (name, got, expect)

        with torch.no_grad():
            vecs = emb.weight[torch.as_tensor(both)].to(torch.float64)
        ref = StreamingCovariance(6).update(vecs).metrics()
        assert abs(rec["soft_rank"] - ref["soft_rank"]) < 1e-8

        assert probes.n_forwards == 0 and probes.probes[0].pooled is None  # reset
        probes.close()
    finally:
        dist.destroy_process_group()


class _NamedA(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 4)

    def forward(self, x):
        return self.a(x)


class _NamedB(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Linear(4, 4)

    def forward(self, x):
        return self.b(x)


def _worker_name_mismatch(rank, store_path):
    dist = _init(rank, store_path)
    try:
        model = _NamedA() if rank == 0 else _NamedB()
        probes = attach_probes(model, select=lambda n, m: isinstance(m, nn.Linear))
        model(torch.randn(3, 4))
        with pytest.raises(RuntimeError, match="identical probe names"):
            probes.compute(distributed=True)
        probes.close()
    finally:
        dist.destroy_process_group()


def _spawn(fn):
    fd, path = tempfile.mkstemp(prefix="st_dist_")
    os.close(fd)
    os.unlink(path)
    try:
        mp.spawn(fn, args=(path,), nprocs=WORLD, join=True)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_two_process_all_reduce_matches_direct():
    _spawn(_worker_allreduce)


def test_two_process_empty_shard():
    _spawn(_worker_empty_shard)


def test_two_process_probeset_distributed_compute():
    _spawn(_worker_probeset)


def test_two_process_rank_consistency_validation():
    _spawn(_worker_name_mismatch)


def test_all_reduce_merge_without_dist_returns_clone():
    acc = StreamingCovariance(4).update(torch.randn(10, 4, dtype=torch.float64))
    out = all_reduce_merge(acc)
    assert out is not acc and out.n == acc.n
    assert torch.allclose(out.covariance(), acc.covariance())
