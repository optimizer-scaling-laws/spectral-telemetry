"""Pass-2 ergonomics: lifecycle, extractor, expected dim, policy, subsampling."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from spectral_telemetry.core.frequency import FrequencyTable
from spectral_telemetry.torch_backend.probe import attach_probes


def _table(vocab=50):
    ranks = np.arange(1, vocab + 1, dtype=np.float64)
    return FrequencyTable(np.maximum((1e5 / ranks).astype(np.int64), 1))


class DictOut(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 8)

    def forward(self, x):
        return {"hidden": self.lin(x), "aux": 1}


def test_enable_disable_and_forward_counter():
    lin = nn.Linear(4, 4)
    probes = attach_probes(lin, select=lambda n, m: m is lin)
    lin(torch.randn(5, 4))
    assert probes.n_forwards == 1
    probes.disable()
    lin(torch.randn(5, 4))  # ignored
    assert probes.n_forwards == 1
    assert probes.compute()["Linear"]["n_samples"] == 5
    probes.enable()
    lin(torch.randn(5, 4))
    rec = probes.compute()["Linear"]
    assert rec["n_samples"] == 10 and rec["n_forwards"] == 2
    probes.close()


def test_pending_context_survives_disabled_forward():
    emb = nn.Embedding(50, 6)
    probes = attach_probes(
        emb, select=lambda n, m: isinstance(m, nn.Embedding), freq_table=_table()
    )
    ids = torch.randint(0, 50, (2, 10))
    probes.set_batch_context(token_ids=ids)
    probes.disable()
    emb(ids)  # skipped; pending context preserved
    probes.enable()
    emb(ids)  # consumes the pending context
    fb = next(iter(probes.compute().values()))["frequency_buckets"]
    assert sum(v["n_samples"] for v in fb.values()) == 20
    with pytest.raises(RuntimeError, match="fresh batch context"):
        emb(ids)
    probes.close()


def test_close_is_idempotent_and_flagged():
    lin = nn.Linear(4, 4)
    probes = attach_probes(lin, select=lambda n, m: m is lin)
    lin(torch.randn(3, 4))
    assert not probes.is_closed
    probes.close()
    probes.close()  # no error
    assert probes.is_closed
    lin(torch.randn(3, 4))  # hooks removed
    assert probes.compute()["Linear"]["n_samples"] == 3


def test_compute_reset_clears_everything():
    lin = nn.Linear(4, 4)
    probes = attach_probes(lin, select=lambda n, m: m is lin)
    lin(torch.randn(5, 4))
    rec = probes.compute(reset=True)["Linear"]
    assert rec["n_samples"] == 5
    assert probes.n_forwards == 0 and probes.probes[0].pooled is None
    with pytest.raises(RuntimeError, match="no activations"):
        probes.compute()
    probes.close()


def test_tensor_extractor_dict_output_and_input_capture():
    m = DictOut()
    probes = attach_probes(
        m, select=lambda n, mod: mod is m, tensor_extractor=lambda mod, inp, out: out["hidden"]
    )
    m(torch.randn(6, 4))
    assert probes.compute()["DictOut"]["hidden_dim"] == 8
    probes.close()

    m2 = DictOut()
    probes2 = attach_probes(
        m2, select="lin", capture="input", tensor_extractor=lambda mod, inp, out: inp[0]
    )
    m2(torch.randn(6, 4))
    assert probes2.compute()["lin"]["hidden_dim"] == 4
    probes2.close()


def test_extractor_and_dtype_errors():
    m = DictOut()
    probes = attach_probes(m, select=lambda n, mod: mod is m)  # no extractor
    with pytest.raises(TypeError, match="not a tensor"):
        m(torch.randn(3, 4))
    probes.close()

    emb = nn.Embedding(10, 4)
    p2 = attach_probes(emb, select=lambda n, mod: isinstance(mod, nn.Embedding), capture="input")
    with pytest.raises(TypeError, match="non-floating"):
        emb(torch.randint(0, 10, (3, 2)))
    p2.close()


def test_expected_hidden_dim_guard():
    lin = nn.Linear(4, 8)
    probes = attach_probes(lin, select=lambda n, m: m is lin, expected_hidden_dim=16)
    with pytest.raises(ValueError, match="expected_hidden_dim"):
        lin(torch.randn(3, 4))
    probes.close()


def test_accumulation_policy_recorded_and_respected():
    lin = nn.Linear(4, 4)
    probes = attach_probes(lin, select=lambda n, m: m is lin, dtype=torch.float32)
    lin(torch.randn(4, 4))
    rec = probes.compute()["Linear"]
    assert rec["accumulation_dtype"] == "torch.float32"
    assert rec["accumulation_device"] == "cpu"
    assert probes.probes[0].pooled.mean.dtype == torch.float32
    probes.close()


def _capped_run():
    torch.manual_seed(0)
    lin = nn.Linear(4, 4)
    probes = attach_probes(
        lin, select=lambda n, m: m is lin, max_tokens_per_forward=10, sampling_seed=123
    )
    lin(torch.randn(100, 4))
    rec = probes.compute()["Linear"]
    probes.close()
    return rec


def test_max_tokens_per_forward_is_deterministic():
    a, b = _capped_run(), _capped_run()
    assert a["n_samples"] == 10 and a["max_tokens_per_forward"] == 10
    assert a["soft_rank"] == pytest.approx(b["soft_rank"], rel=1e-12)


def test_subsampling_applies_to_tokens_too():
    emb = nn.Embedding(50, 6)
    probes = attach_probes(
        emb,
        select=lambda n, m: isinstance(m, nn.Embedding),
        freq_table=_table(),
        max_tokens_per_forward=8,
        sampling_seed=7,
    )
    ids = torch.randint(0, 50, (2, 20))
    with probes.batch_context(token_ids=ids):
        emb(ids)
    fb = next(iter(probes.compute().values()))["frequency_buckets"]
    assert sum(v["n_samples"] for v in fb.values()) == 8
    probes.close()
