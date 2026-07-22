"""Probe layer. First two cases adapted from upstream test_probe_api (MIT)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from spectral_telemetry.core.frequency import FrequencyTable
from spectral_telemetry.torch_backend.probe import attach_probes, spectral_rank


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(8, 32), nn.GELU(), nn.Linear(32, 8)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def test_spectral_rank_tensor_api():
    torch.manual_seed(0)
    m = spectral_rank(torch.randn(64, 16))
    assert m["n_samples"] == 64 and m["hidden_dim"] == 16
    assert m["soft_rank"] >= m["hard_rank"] - 1e-5 > 0


def test_attach_probe_output_capture():
    torch.manual_seed(0)
    model = TinyMLP()
    with attach_probes(model, select="fc1") as probes:
        model(torch.randn(20, 8))
        metrics = probes.compute()
    assert list(metrics) == ["fc1"]
    assert metrics["fc1"]["n_samples"] == 20 and metrics["fc1"]["hidden_dim"] == 32


def test_selector_variants_and_streaming_across_batches():
    model = TinyMLP()
    probes = attach_probes(model, select=lambda name, mod: isinstance(mod, nn.Linear))
    assert len(probes) == 2
    for _ in range(5):
        model(torch.randn(16, 8))
    out = probes.compute()
    probes.close()
    assert out["fc1"]["n_samples"] == 80 and out["fc2"]["hidden_dim"] == 8


def test_input_capture_dim():
    model = TinyMLP()
    probes = attach_probes(model, select="fc2", capture="input")
    model(torch.randn(10, 8))
    assert probes.compute()["fc2"]["hidden_dim"] == 32
    probes.close()


def _tiny_freq_table(vocab=50):
    ranks = np.arange(1, vocab + 1, dtype=np.float64)
    return FrequencyTable(np.maximum((1e5 / ranks).astype(np.int64), 1))


def test_bucketed_spectral_rank():
    torch.manual_seed(0)
    acts = torch.randn(4, 25, 6)  # [B, T, D]
    tokens = torch.randint(0, 50, (4, 25))
    out = spectral_rank(
        acts, token_ids=tokens, freq_table=_tiny_freq_table(), min_samples_per_bucket=2
    )
    fb = out["frequency_buckets"]
    assert set(fb) == {"head", "mid", "tail"}
    total = sum(v["n_samples"] for v in fb.values())
    assert total == 100


def test_probeset_bucketed_streaming_requires_and_uses_context():
    torch.manual_seed(0)
    table = _tiny_freq_table()
    emb = nn.Embedding(50, 6)
    probes = attach_probes(emb, select=lambda n, m: isinstance(m, nn.Embedding), freq_table=table)
    with pytest.raises(RuntimeError):  # no batch context armed
        emb(torch.randint(0, 50, (2, 10)))
    for _ in range(3):
        ids = torch.randint(0, 50, (2, 10))
        with probes.batch_context(token_ids=ids):
            emb(ids)
    out = probes.compute(min_samples_per_bucket=2)
    fb = next(iter(out.values()))["frequency_buckets"]
    counted = sum(v["n_samples"] for v in fb.values())
    assert counted == 60
    probes.close()
