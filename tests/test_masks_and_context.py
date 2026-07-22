"""Pass-1 acceptance tests: sample masks, batch-context lifecycle, ID types."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from spectral_telemetry.core.frequency import FrequencyTable
from spectral_telemetry.torch_backend.probe import attach_probes, spectral_rank
from spectral_telemetry.torch_backend.streaming import StreamingCovariance


def _tiny_freq_table(vocab=50):
    ranks = np.arange(1, vocab + 1, dtype=np.float64)
    return FrequencyTable(np.maximum((1e5 / ranks).astype(np.int64), 1))


def test_mask_excludes_rows_from_pooled():
    torch.manual_seed(0)
    x = torch.randn(4, 25, 6, dtype=torch.float64)
    mask = torch.rand(4, 25) > 0.4
    mask[0, :2] = True
    m = spectral_rank(x, sample_mask=mask)
    ref = spectral_rank(x.reshape(-1, 6)[mask.reshape(-1)])
    assert m["n_samples"] == int(mask.sum()) == ref["n_samples"]
    assert m["soft_rank"] == pytest.approx(ref["soft_rank"], rel=1e-9)
    assert m["hard_rank"] == pytest.approx(ref["hard_rank"], rel=1e-9)


def test_mask_excludes_rows_from_bucket_counts():
    torch.manual_seed(1)
    table = _tiny_freq_table()
    x = torch.randn(2, 30, 6)
    tokens = torch.randint(0, 50, (2, 30))
    mask = torch.rand(2, 30) > 0.3
    mask[0, :2] = True
    out = spectral_rank(x, token_ids=tokens, sample_mask=mask, freq_table=table)
    fb = out["frequency_buckets"]
    assert sum(v["n_samples"] for v in fb.values()) == int(mask.sum())
    # per-bucket counts match an explicitly filtered reference
    kept_tokens = tokens.reshape(-1)[mask.reshape(-1)].numpy()
    ref_buckets = table.get_bucket(kept_tokens)
    for idx, name in enumerate(("head", "mid", "tail")):
        assert fb[name]["n_samples"] == int((ref_buckets == idx).sum())


def test_flat_2d_activations_with_numpy_mask_and_tokens():
    torch.manual_seed(2)
    x = torch.randn(40, 5, dtype=torch.float64)
    mask_np = np.ones(40, dtype=bool)
    mask_np[:10] = False
    tok_np = np.random.default_rng(0).integers(0, 50, size=40)
    out = spectral_rank(x, token_ids=tok_np, sample_mask=mask_np, freq_table=_tiny_freq_table())
    assert out["n_samples"] == 30


def test_zero_one_attention_mask_accepted():
    x = torch.randn(3, 8, 4, dtype=torch.float64)
    attn = torch.ones(3, 8, dtype=torch.long)
    attn[:, -3:] = 0
    m = spectral_rank(x, sample_mask=attn)
    assert m["n_samples"] == 15


def test_padding_heavy_batch_matches_filtered_reference_streaming():
    torch.manual_seed(3)
    lin = nn.Linear(6, 8)
    probes = attach_probes(lin, select=lambda n, m: m is lin)
    kept = []
    for _ in range(3):
        x = torch.randn(2, 10, 6)
        mask = torch.rand(2, 10) > 0.5
        mask[0, 0] = True
        with probes.batch_context(sample_mask=mask):
            y = lin(x)
        kept.append(y.detach().reshape(-1, 8)[mask.reshape(-1)])
    m = probes.compute()["Linear"]
    ref = StreamingCovariance(8).update(torch.cat(kept).to(torch.float64)).metrics()
    assert m["n_samples"] == ref["n_samples"]
    assert m["soft_rank"] == pytest.approx(ref["soft_rank"], rel=1e-9)
    assert m["hard_rank"] == pytest.approx(ref["hard_rank"], rel=1e-9)
    probes.close()


def _bucketed_embedding_probes():
    emb = nn.Embedding(50, 6)
    probes = attach_probes(
        emb, select=lambda n, m: isinstance(m, nn.Embedding), freq_table=_tiny_freq_table()
    )
    return emb, probes


def test_stale_context_cannot_be_reused():
    emb, probes = _bucketed_embedding_probes()
    ids = torch.randint(0, 50, (2, 10))
    probes.set_batch_context(token_ids=ids)
    emb(ids)  # consumes the context
    with pytest.raises(RuntimeError, match="fresh batch context"):
        emb(torch.randint(0, 50, (2, 10)))
    probes.close()


def test_context_manager_is_single_use():
    emb, probes = _bucketed_embedding_probes()
    ids = torch.randint(0, 50, (2, 10))
    with probes.batch_context(token_ids=ids):
        emb(ids)
        with pytest.raises(RuntimeError, match="fresh batch context"):
            emb(ids)  # second forward inside the same context: already consumed
    with pytest.raises(RuntimeError, match="fresh batch context"):
        emb(ids)  # after the context exits
    probes.close()


def test_reset_clears_context_and_accumulators():
    emb, probes = _bucketed_embedding_probes()
    ids = torch.randint(0, 50, (2, 10))
    probes.set_batch_context(token_ids=ids)
    probes.reset()
    with pytest.raises(RuntimeError, match="fresh batch context"):
        emb(ids)
    assert probes.probes[0].pooled is None
    with probes.batch_context(token_ids=ids):
        emb(ids)
    assert probes.compute()[next(iter(probes.compute()))]["n_samples"] == 20
    probes.close()


def test_alignment_and_type_errors():
    x = torch.randn(20, 4, dtype=torch.float64)
    with pytest.raises(ValueError, match="sample_mask has"):
        spectral_rank(x, sample_mask=np.ones(19, dtype=bool))
    with pytest.raises(ValueError, match="token_ids has"):
        spectral_rank(x, token_ids=np.arange(19), freq_table=_tiny_freq_table())
    emb, probes = _bucketed_embedding_probes()
    with pytest.raises(ValueError, match="same positions"):
        probes.set_batch_context(token_ids=np.arange(20), sample_mask=np.ones(19, bool))
    probes.set_batch_context(token_ids=torch.randint(0, 50, (5,)))
    with pytest.raises(ValueError, match="token IDs vs"):
        emb(torch.randint(0, 50, (2, 10)))
    probes.close()


def test_fully_masked_forward_contributes_nothing():
    lin = nn.Linear(6, 8)
    probes = attach_probes(lin, select=lambda n, m: m is lin)
    with probes.batch_context(sample_mask=torch.zeros(20, dtype=torch.bool)):
        lin(torch.randn(2, 10, 6))
    lin(torch.randn(2, 10, 6))  # unmasked forward (mask is opt-in for pooled)
    assert probes.compute()["Linear"]["n_samples"] == 20
    probes.close()


def test_numpy_token_ids_in_batch_context():
    emb, probes = _bucketed_embedding_probes()
    ids = np.random.default_rng(1).integers(0, 50, size=(2, 10))
    with probes.batch_context(token_ids=ids):
        emb(torch.as_tensor(ids))
    out = next(iter(probes.compute().values()))
    assert sum(v["n_samples"] for v in out["frequency_buckets"].values()) == 20
    probes.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bucketed_spectral_rank_on_cuda():
    x = torch.randn(2, 30, 6, device="cuda")
    tokens = torch.randint(0, 50, (2, 30), device="cuda")
    mask = torch.rand(2, 30, device="cuda") > 0.3
    out = spectral_rank(x, token_ids=tokens, sample_mask=mask, freq_table=_tiny_freq_table())
    assert set(out["frequency_buckets"]) == {"head", "mid", "tail"}
    assert out["n_samples"] == int(mask.sum())
