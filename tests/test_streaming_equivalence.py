"""The library's core guarantee: streaming == batch covariance, exactly."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spectral_telemetry.core.ranks import (
    covariance_from_activations,
    eigs_from_covariance,
    rank_metrics,
)
from spectral_telemetry.torch_backend.streaming import StreamingCovariance


def _direct_cov(x: torch.Tensor) -> torch.Tensor:
    c = x - x.mean(dim=0, keepdim=True)
    return (c.t() @ c) / (x.shape[0] - 1)


def test_chunked_updates_match_batch_covariance():
    torch.manual_seed(0)
    x = torch.randn(300, 16, dtype=torch.float64)
    acc = StreamingCovariance(16)
    for chunk in torch.split(x, [50, 120, 130]):
        acc.update(chunk)
    assert acc.n == 300
    assert torch.allclose(acc.covariance(), _direct_cov(x), atol=1e-10)


def test_merge_matches_full_stream_and_is_symmetric():
    torch.manual_seed(1)
    a = torch.randn(80, 8, dtype=torch.float64) + 3.0  # deliberate mean offset
    b = torch.randn(200, 8, dtype=torch.float64) - 1.0
    full = StreamingCovariance(8).update(torch.cat([a, b]))
    left = StreamingCovariance(8).update(a)
    right = StreamingCovariance(8).update(b)
    merged = StreamingCovariance.merged([left.clone(), right.clone()])
    merged_rev = StreamingCovariance.merged([right, left])
    assert torch.allclose(merged.covariance(), full.covariance(), atol=1e-10)
    assert torch.allclose(merged_rev.covariance(), full.covariance(), atol=1e-10)


def test_metrics_match_numpy_core_path():
    torch.manual_seed(2)
    x = torch.randn(256, 12, dtype=torch.float64)
    m = StreamingCovariance(12).update(x).metrics()
    ref = rank_metrics(eigs_from_covariance(covariance_from_activations(x.numpy())))
    assert m["soft_rank"] == pytest.approx(ref["soft_rank"], rel=1e-8)
    assert m["hard_rank"] == pytest.approx(ref["hard_rank"], rel=1e-8)
    assert m["n_samples"] == 256 and m["hidden_dim"] == 12


def test_nd_input_and_error_paths():
    acc = StreamingCovariance(4)
    acc.update(torch.randn(2, 3, 4))  # [B, T, D] flattens
    assert acc.n == 6
    with pytest.raises(ValueError):
        acc.update(torch.randn(5, 3))  # wrong trailing dim
    with pytest.raises(ValueError):
        StreamingCovariance(4).covariance()  # n < 2


# ---- pass-3: finite/state validation and diagnostics-rich records ------------


def test_non_finite_activations_fail_near_source():
    acc = StreamingCovariance(4)
    bad = torch.randn(6, 4, dtype=torch.float64)
    bad[2, 1] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        acc.update(bad)
    relaxed = StreamingCovariance(4, check_finite=False)
    relaxed.update(bad)  # explicitly opted out; no raise


def test_probe_error_names_the_module():
    import torch.nn as nn

    from spectral_telemetry.torch_backend.probe import attach_probes

    lin = nn.Linear(4, 4)
    probes = attach_probes(lin, select=lambda n, m: m is lin)
    with pytest.raises(ValueError, match=r"probe\[Linear\].*non-finite"):
        lin(torch.full((3, 4), float("nan")))
    probes.close()


def test_from_state_validation():
    good = StreamingCovariance(3).update(torch.randn(10, 3, dtype=torch.float64))
    n, mean, m2 = good.state()
    with pytest.raises(ValueError, match="non-negative"):
        StreamingCovariance.from_state(-1, mean, m2)
    with pytest.raises(ValueError, match="1-D"):
        StreamingCovariance.from_state(n, mean.reshape(1, 3), m2)
    with pytest.raises(ValueError, match="square"):
        StreamingCovariance.from_state(n, mean, m2[:2])
    mean_bad = mean.clone()
    mean_bad[0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        StreamingCovariance.from_state(n, mean_bad, m2)


def test_metrics_and_insufficient_buckets_carry_diagnostics():
    torch.manual_seed(0)
    rec = StreamingCovariance(16).update(torch.randn(5, 16, dtype=torch.float64)).metrics()
    assert rec["estimation_status"] == "severely_undersampled"
    assert rec["max_empirical_rank"] == 4
    assert "min_raw_eigenvalue" in rec and "soft_rank_fraction_of_dim" in rec

    from spectral_telemetry.core.frequency import FrequencyTable
    from spectral_telemetry.torch_backend.probe import spectral_rank

    ranks_ = np.arange(1, 51, dtype=np.float64)
    table = FrequencyTable(np.maximum((1e5 / ranks_).astype(np.int64), 1))
    x = torch.randn(1, 4, 6)  # 4 rows: some bucket will starve
    tokens = torch.tensor([[0, 0, 1, 49]])
    out = spectral_rank(x, token_ids=tokens, freq_table=table, min_samples_per_bucket=3)
    starved = [v for v in out["frequency_buckets"].values() if "status" in v]
    assert starved, "expected at least one insufficient bucket"
    for entry in starved:
        assert entry["estimation_status"] == "insufficient_samples" or entry["n_samples"] >= 2
        assert entry["hidden_dim"] == 6 and "sample_to_dim_ratio" in entry


# ---- regression: fp32 accumulation must tolerate its own rounding floor ------
def test_negative_tolerance_scales_with_accumulation_dtype():
    from spectral_telemetry.torch_backend.streaming import _negative_rtol_for

    assert _negative_rtol_for(torch.float64) == 1e-6
    assert _negative_rtol_for(torch.float32) == 1e-3


def test_fp32_noise_floor_matrix_passes_fp32_policy():
    from spectral_telemetry.core.ranks import eigs_from_covariance

    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    cov = q @ np.diag([1.7, 1.0, 0.5, 0.2, 0.1, 0.05, 1e-6, -6.9e-5]) @ q.T
    cov = (cov + cov.T) / 2
    with pytest.raises(ValueError, match="materially negative"):
        eigs_from_covariance(cov)  # float64 policy (1e-6) rejects this
    eigs, diag = eigs_from_covariance(cov, materially_negative_rtol=1e-3, return_diagnostics=True)
    assert eigs.min() >= 1e-12
    assert diag["min_raw_eigenvalue"] == pytest.approx(-6.9e-5, rel=1e-2)


def test_fp32_accumulator_metrics_end_to_end():
    torch.manual_seed(0)
    acc = StreamingCovariance(64, dtype=torch.float32)
    for _ in range(20):
        acc.update(torch.randn(512, 64) * 3 + 1)
    rec = acc.metrics()  # previously could raise on the fp32 rounding floor
    assert rec["n_samples"] == 10240 and "min_raw_eigenvalue" in rec
