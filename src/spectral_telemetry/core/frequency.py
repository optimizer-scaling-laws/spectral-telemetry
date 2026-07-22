"""Occurrence-balanced HEAD/MID/TAIL token-frequency buckets (numpy-only).

Port of upstream ``TokenFrequencyTable`` boundary logic (MIT): tokens are
sorted by corpus frequency and cut so each bucket covers roughly one third of
token *occurrences* (not one third of the vocabulary), with the same cutoff
clamping as upstream so bucket assignments are bit-identical for valid input.

This module validates its inputs: frequency vectors must be 1-D, non-negative,
finite, with positive total mass and at least :data:`MIN_VOCAB_SIZE` entries;
token IDs must lie in ``[0, vocab_size)``. Tied or tiny vocabularies can make
occurrence-balanced thirds impossible -- in that case buckets are *reported as
degenerate* (``empty_buckets`` / ``is_degenerate`` / ``occurrence_shares``)
rather than silently presented as balanced.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["BUCKET_NAMES", "MIN_VOCAB_SIZE", "FrequencyTable", "load_frequency_vector"]

BUCKET_NAMES = ("head", "mid", "tail")
MIN_VOCAB_SIZE = 3


def load_frequency_vector(filepath: str | Path) -> np.ndarray:
    path = Path(filepath)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        freq = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as archive:
            key = "frequencies" if "frequencies" in archive.files else archive.files[0]
            freq = archive[key]
    else:
        raise ValueError(f"Unsupported token-frequency file format: {path.suffix}")
    return np.asarray(freq)


class FrequencyTable:
    """Token frequencies with occurrence-balanced head/mid/tail assignment."""

    def __init__(self, frequencies) -> None:
        freq = np.asarray(frequencies)
        if freq.ndim != 1:
            raise ValueError(f"frequencies must be a 1-D vector, got shape {freq.shape}")
        if freq.size < MIN_VOCAB_SIZE:
            raise ValueError(
                f"vocabulary too small for {len(BUCKET_NAMES)} frequency buckets: "
                f"{freq.size} tokens < MIN_VOCAB_SIZE={MIN_VOCAB_SIZE}"
            )
        if not np.issubdtype(freq.dtype, np.number):
            raise ValueError(f"frequencies must be numeric, got dtype {freq.dtype}")
        as_float = freq.astype(np.float64)
        if not np.all(np.isfinite(as_float)):
            raise ValueError("frequencies contain NaN or Inf values")
        if np.any(as_float < 0):
            raise ValueError("frequencies must be non-negative")

        self.frequencies = freq.astype(np.int64)
        self.vocab_size = int(self.frequencies.size)
        self.total_tokens = int(self.frequencies.sum())
        if self.total_tokens <= 0:
            raise ValueError("total token frequency must be positive (all-zero vector)")
        self._compute_boundaries()
        self._compute_statistics()

    @classmethod
    def from_file(cls, filepath: str | Path) -> "FrequencyTable":
        return cls(load_frequency_vector(filepath))

    def _compute_boundaries(self) -> None:
        sorted_freqs = np.sort(self.frequencies)[::-1]
        cumsum = np.cumsum(sorted_freqs).astype(np.float64)
        total = float(cumsum[-1])

        head_cutoff_idx = int(np.sum(cumsum <= total * 0.33))
        mid_cutoff_idx = int(np.sum(cumsum <= total * 0.67))
        head_cutoff_idx = max(1, min(head_cutoff_idx, len(sorted_freqs) - 1))
        mid_cutoff_idx = max(head_cutoff_idx + 1, min(mid_cutoff_idx, len(sorted_freqs) - 1))

        self.head_min_freq = int(sorted_freqs[head_cutoff_idx - 1])
        self.mid_min_freq = int(sorted_freqs[mid_cutoff_idx - 1])

    def _compute_statistics(self) -> None:
        f = self.frequencies
        head = f >= self.head_min_freq
        mid = (f >= self.mid_min_freq) & (f < self.head_min_freq)
        tail = f < self.mid_min_freq
        self.tokens_per_bucket = {
            "head": int(head.sum()),
            "mid": int(mid.sum()),
            "tail": int(tail.sum()),
        }
        self.occurrences_per_bucket = {
            "head": int(f[head].sum()),
            "mid": int(f[mid].sum()),
            "tail": int(f[tail].sum()),
        }
        self.occurrence_shares = {
            b: self.occurrences_per_bucket[b] / self.total_tokens for b in BUCKET_NAMES
        }
        self.empty_buckets = tuple(b for b in BUCKET_NAMES if self.tokens_per_bucket[b] == 0)
        self.is_degenerate = len(self.empty_buckets) > 0

    def get_bucket(self, token_ids) -> np.ndarray:
        """Map token IDs -> 0=head, 1=mid, 2=tail (upstream threshold semantics).

        Raises ``ValueError`` for non-integer IDs or IDs outside
        ``[0, vocab_size)`` -- negative IDs are rejected rather than silently
        indexing from the end of the vocabulary.
        """
        ids = np.asarray(token_ids)
        if ids.size == 0:
            return np.zeros(0, dtype=np.int64)
        if not np.issubdtype(ids.dtype, np.integer):
            raise ValueError(f"token IDs must be integers, got dtype {ids.dtype}")
        ids = ids.reshape(-1).astype(np.int64)
        lo, hi = int(ids.min()), int(ids.max())
        if lo < 0 or hi >= self.vocab_size:
            raise ValueError(
                f"token IDs out of range [0, {self.vocab_size}): found min={lo}, max={hi}"
            )
        freqs = self.frequencies[ids]
        buckets = np.full(ids.shape, 2, dtype=np.int64)
        buckets[freqs >= self.mid_min_freq] = 1
        buckets[freqs >= self.head_min_freq] = 0
        return buckets

    def bucket_diagnostics(self) -> dict:
        """Actual (not idealized) bucket composition and degeneracy flags."""
        return {
            "tokens_per_bucket": dict(self.tokens_per_bucket),
            "occurrences_per_bucket": dict(self.occurrences_per_bucket),
            "occurrence_shares": dict(self.occurrence_shares),
            "empty_buckets": self.empty_buckets,
            "is_degenerate": self.is_degenerate,
            "head_min_freq": self.head_min_freq,
            "mid_min_freq": self.mid_min_freq,
        }

    def summary(self) -> str:
        flag = " DEGENERATE(" + ",".join(self.empty_buckets) + ")" if self.is_degenerate else ""
        return (
            f"FrequencyTable(vocab={self.vocab_size:,}, total={self.total_tokens:,}, "
            f"head>= {self.head_min_freq:,}, mid>= {self.mid_min_freq:,}; "
            "occ share head/mid/tail = "
            + "/".join(f"{100 * self.occurrence_shares[b]:.1f}%" for b in BUCKET_NAMES)
            + f"{flag})"
        )
