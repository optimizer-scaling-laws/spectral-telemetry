import numpy as np

from spectral_telemetry.core.frequency import BUCKET_NAMES, FrequencyTable


def _zipf_table(vocab=100, seed=0):
    ranks = np.arange(1, vocab + 1, dtype=np.float64)
    counts = np.maximum((1e6 / ranks).astype(np.int64), 1)
    return FrequencyTable(counts)


def test_buckets_partition_vocab_and_balance_occurrences():
    t = _zipf_table()
    assert sum(t.tokens_per_bucket[b] for b in BUCKET_NAMES) == t.vocab_size
    total = sum(t.occurrences_per_bucket[b] for b in BUCKET_NAMES)
    assert total == t.total_tokens
    # occurrence-balanced: head is far fewer *tokens* than tail under Zipf
    assert t.tokens_per_bucket["head"] < t.tokens_per_bucket["tail"]


def test_bucket_assignment_ordering():
    t = _zipf_table()
    buckets = t.get_bucket(np.array([0, 1, 50, 99]))
    assert buckets[0] == 0  # most frequent token -> HEAD
    assert buckets[-1] == 2  # rarest token -> TAIL
    assert set(np.unique(t.get_bucket(np.arange(100)))) == {0, 1, 2}


def test_get_bucket_accepts_any_shape():
    t = _zipf_table()
    out = t.get_bucket(np.arange(12).reshape(3, 4))
    assert out.shape == (12,)


# ---- pass-1 validation cases -------------------------------------------------
import pytest


def test_invalid_frequency_vectors_rejected():
    with pytest.raises(ValueError, match="1-D"):
        FrequencyTable(np.ones((4, 4)))
    with pytest.raises(ValueError, match="too small"):
        FrequencyTable(np.array([5]))
    with pytest.raises(ValueError, match="too small"):
        FrequencyTable(np.array([5, 3]))
    with pytest.raises(ValueError, match="non-negative"):
        FrequencyTable(np.array([5, -1, 3]))
    with pytest.raises(ValueError, match="positive"):
        FrequencyTable(np.zeros(10, dtype=np.int64))
    with pytest.raises(ValueError, match="NaN or Inf"):
        FrequencyTable(np.array([1.0, np.nan, 2.0]))


def test_token_id_range_and_type_validation():
    t = _zipf_table()
    with pytest.raises(ValueError, match="out of range"):
        t.get_bucket(np.array([-1, 3]))
    with pytest.raises(ValueError, match="out of range"):
        t.get_bucket(np.array([0, 100]))
    with pytest.raises(ValueError, match="integers"):
        t.get_bucket(np.array([0.5, 1.5]))
    assert t.get_bucket(np.array([], dtype=np.int64)).size == 0


def test_degenerate_buckets_are_reported_not_hidden():
    t = FrequencyTable(np.full(10, 7, dtype=np.int64))  # all-equal frequencies
    assert t.is_degenerate
    assert set(t.empty_buckets) == {"mid", "tail"}
    assert t.occurrence_shares["head"] == pytest.approx(1.0)
    d = t.bucket_diagnostics()
    assert d["is_degenerate"] and d["tokens_per_bucket"]["head"] == 10
    assert "DEGENERATE" in t.summary()


def test_healthy_table_not_flagged():
    t = _zipf_table()
    assert not t.is_degenerate and t.empty_buckets == ()
    assert sum(t.occurrence_shares.values()) == pytest.approx(1.0)
