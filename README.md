<div align="center">

# Spectral Telemetry

**Streaming, mergeable spectral-capacity diagnostics for neural-network activations.**

[![CI](https://github.com/optimizer-scaling-laws/spectral-telemetry/actions/workflows/ci.yml/badge.svg)](https://github.com/optimizer-scaling-laws/spectral-telemetry/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2605.21803-b31b1b.svg)](https://arxiv.org/abs/2605.21803)

[Paper](https://arxiv.org/abs/2605.21803) ·
[Project page](https://optimizer-scaling-laws.github.io/) ·
[Examples](examples/) ·
[Benchmarks](docs/benchmarks.md)

</div>

`spectral-telemetry` measures the eigenspectrum of centered activation covariance and reports
Shannon effective rank (**soft rank**), participation ratio (**hard rank**), and general Rényi
effective ranks. Its streaming accumulator uses **O(D²) memory per accumulator, independent of
the number of streamed samples**, and supports mergeable statistics across micro-batches,
HEAD/MID/TAIL token-frequency buckets, and data-parallel ranks.

The library is extracted from
[*Same Architecture, Different Capacity: Optimizer-Induced Spectral Scaling Laws*](https://arxiv.org/abs/2605.21803),
with the paper repository's numerical conventions preserved so measurements remain directly
comparable with the published telemetry.

> [!IMPORTANT]
> These metrics characterize **data-conditioned utilization of activation directions** for a
> specified model, layer, dataset, mask, and measurement window. They are not an absolute or
> task-independent measure of model capacity.

## Highlights

- **Streaming full covariance:** no activation-row buffer; memory is constant in sample count.
- **Soft, hard, and Rényi ranks:** one API for concentration-sensitive spectral summaries.
- **Reliability diagnostics:** every record reports `N/D`, empirical-rank limits, sampling status,
  and eigenvalue-health checks.
- **Token-regime telemetry:** occurrence-balanced HEAD/MID/TAIL statistics using the same
  conventions as the paper.
- **Distributed reduction:** algebraically exact two-stage reduction with between-means correction.
- **Model-agnostic probes:** attach to arbitrary PyTorch modules by name pattern, predicate, or
  module instance.
- **NumPy-only core:** ranks, fits, frequency buckets, and legacy-log parsing work without PyTorch.

**Project status:** v0.1.0 beta. The exact full-covariance path is tested and usable; the public API
may still evolve before v1.0.

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/optimizer-scaling-laws/spectral-telemetry.git
cd spectral-telemetry

pip install -e ".[torch]"   # probes + streaming covariance
# or
pip install -e .            # NumPy-only core
```

For development:

```bash
pip install -e ".[dev,torch]"
pytest -q
```

## Quick start

### One-shot measurement

```python
import torch
from spectral_telemetry.telemetry import spectral_rank

acts = torch.randn(4, 128, 512)                    # [batch, tokens, hidden]
mask = torch.ones(4, 128, dtype=torch.bool)        # exclude padding as needed

record = spectral_rank(acts, sample_mask=mask)
print(record["soft_rank"], record["hard_rank"])
print(record["sample_to_dim_ratio"], record["estimation_status"])
```

### Streaming probes

```python
from spectral_telemetry.telemetry import attach_probes

probes = attach_probes(
    model,
    select="*.mlp.c_proj",
    capture="input",
)

for batch in loader:
    # Batch contexts are single-use: armed before one forward, then cleared.
    with probes.batch_context(
        token_ids=batch["input_ids"],
        sample_mask=batch["attention_mask"],
    ):
        model(**batch)

records = probes.compute(reset=True)   # {module_name: telemetry_record}
probes.close()
```

For CUDA training, the recommended policy is:

```python
probes = attach_probes(
    model,
    select="*.mlp.c_proj",
    capture="input",
    accumulation_device="cuda",
    dtype=torch.float32,
)
```

The default CPU/float64 policy minimizes GPU-memory use and prioritizes numerical robustness, but
can dominate wall-clock time on fast GPUs. The CUDA/fp32 policy keeps accumulation on device and
moves only the completed covariance for eigendecomposition. See the
[benchmark report](docs/benchmarks.md).

## Frequency-conditioned telemetry

Pass a corpus-level frequency table to obtain pooled and HEAD/MID/TAIL spectra:

```python
import numpy as np
from spectral_telemetry.telemetry import FrequencyTable, attach_probes

counts = np.load("token_frequencies.npy")
table = FrequencyTable(counts)

probes = attach_probes(
    model,
    select="*.mlp.c_proj",
    capture="input",
    freq_table=table,
)
```

Buckets are balanced by **token occurrences**, not vocabulary size, matching the paper's frequency
analysis. The library validates out-of-range IDs and explicitly reports degenerate buckets.

## Metrics

Let $\lambda_i$ be covariance eigenvalues and
$p_i = \lambda_i / \sum_j \lambda_j$.

| Metric | Definition | Interpretation |
|---|---|---|
| Soft rank | $R_1 = \exp(-\sum_i p_i \log p_i)$ | Shannon-balanced effective dimensionality; sensitive to diffuse spectral mass |
| Hard rank | $R_2 = 1 / \sum_i p_i^2$ | Participation ratio; more sensitive to dominant modes |
| Rényi rank | $R_\alpha = (\sum_i p_i^\alpha)^{1/(1-\alpha)}$ | Continuum of concentration sensitivities; $\alpha=1$ gives soft rank and $\alpha=2$ gives hard rank |

`renyi_rank(eigs, np.inf)` returns the min-entropy rank. Rényi rank is non-increasing in
$\alpha$, so `hard_rank <= soft_rank`.

## Reliability is part of every record

A sample covariance has empirical rank at most `min(D, N - 1)`. Each record therefore includes:

- `n_samples` and `hidden_dim`;
- `sample_to_dim_ratio`;
- `max_empirical_rank`;
- soft/hard rank as fractions of both ambient dimension and empirical maximum;
- `min_raw_eigenvalue` and `negative_eigenvalue_mass`;
- `estimation_status`: `well_sampled`, `moderately_undersampled`,
  `severely_undersampled`, or `insufficient_samples`.

The default sampling thresholds are documented heuristics, not statistical guarantees. A low rank
at small `N/D` may reflect limited sampling rather than learned compression.

## Distributed use

The distributed path uses two collective phases:

1. reduce sample counts and first moments to obtain the global mean;
2. reduce locally centered scatters with the exact between-means correction.

Peak extra reduction memory is one `D × D` tensor per probe, independent of world size. Gloo/MPI
communicate CPU states directly; NCCL stages on the current CUDA device.

```python
records = probes.compute(distributed=True, reset=True)
```

Probe names, hidden dimensions, and bucket configuration are validated across ranks before
collectives begin. Empty local shards and buckets participate with valid empty states.

## Performance

The streaming design removes activation-row storage and keeps memory bounded in measurement-window
length. It does **not** remove the cost of exact covariance or eigendecomposition.

With the recommended CUDA/fp32 policy on an RTX 3090, median per-step times matched or closely
tracked the baseline in the reported 768- and 3072-dimensional experiments; the remaining cost was
paid at collection time. Host memory stayed near 1.0–1.2 GB versus 2.8–3.1 GB for the buffered
emulation in those runs. Full CPU and GPU tables, configurations, and interpretation are in
[docs/benchmarks.md](docs/benchmarks.md).

Run the benchmark locally with:

```bash
python benchmarks/benchmark_probe_overhead.py --help
```

## Examples

| File | What it demonstrates |
|---|---|
| [`examples/probe_quickstart.py`](examples/probe_quickstart.py) | Minimal one-shot and streaming APIs |
| [`examples/transformer_lm.py`](examples/transformer_lm.py) | Masks, frequency buckets, measurement windows, scheduling, and diagnostics |
| [`examples/huggingface_gpt2.py`](examples/huggingface_gpt2.py) | Real GPT-2 FFN probes with tokenized text |
| [`benchmarks/benchmark_probe_overhead.py`](benchmarks/benchmark_probe_overhead.py) | Baseline, buffered, and streaming overhead comparison |

## Public API

```python
from spectral_telemetry.telemetry import (
    spectral_rank,          # one-shot PyTorch measurement
    attach_probes,          # model hooks + streaming accumulation
    StreamingCovariance,    # mergeable covariance primitive
    FrequencyTable,         # occurrence-balanced frequency buckets
    soft_rank,
    hard_rank,
    renyi_rank,
    fit_power_law,
    fit_power_law_with_ci,
)
```

PyTorch-backed objects are imported lazily, so NumPy-only environments can use the core package.

## Scope and limitations

- Exact covariance is quadratic in hidden dimension. One float64 `D × D` scatter is approximately
  4.5 MiB at `D=768`, 72 MiB at `D=3072`, and 288 MiB at `D=6144`.
- HEAD/MID/TAIL telemetry uses roughly four accumulators per probe; costs also multiply with the
  number of probed layers.
- Padding and special-token masks must be supplied by the caller.
- CPU accumulation can synchronize GPU execution; use CUDA/fp32 accumulation for fast GPU loops
  when memory permits.
- Tensor-parallel detection currently covers PyTorch `DTensor`. Other TP systems may expose plain
  local tensors; callers must guarantee capture of the full feature dimension.
- Supported distributed paths are Gloo/MPI with CPU states and NCCL with CUDA staging.
- Accumulators are not thread-safe.
- Sub-quadratic sketches, asynchronous capture, and TP-aware covariance assembly are not yet
  implemented.

## Design principles

- **One primitive, three jobs.** `StreamingCovariance` handles chunked capture, frequency buckets,
  and cross-rank merging through the same state representation.
- **Fail near the source.** Non-finite activations, materially indefinite covariances, invalid token
  IDs, and likely tensor-parallel shards raise rather than silently producing telemetry.
- **Expose measurement policy.** Accumulation device/dtype, sample counts, hidden dimension,
  forward counts, and sampling diagnostics are stamped into records.
- **Preserve upstream comparability.** Epsilon clamps, rank conventions, bucket boundaries, and
  scaling-fit conventions follow the paper repository for valid inputs.

## Roadmap

Planned v0.2+ work includes:

- versioned `TelemetryRecord` objects and JSONL/Parquet serialization;
- W&B and TensorBoard sinks;
- richer sampling and scheduling policies;
- grouping beyond HEAD/MID/TAIL;
- tensor-parallel-aware covariance assembly;
- asynchronous or pinned-memory capture;
- optional sub-quadratic spectrum approximations;
- JAX support if there is demand.

## Citation

If you use the package or its metrics, please cite:

```bibtex
@article{jha2026optimizer,
  title   = {Same Architecture, Different Capacity: Optimizer-Induced
             Spectral Scaling Laws},
  author  = {Jha, Nandan Kumar and Reagen, Brandon},
  journal = {arXiv preprint arXiv:2605.21803},
  year    = {2026}
}
```

A machine-readable citation is available in [`CITATION.cff`](CITATION.cff).

## Provenance and license

Released under the [MIT License](LICENSE). Ported components and preserved numerical conventions
are itemized in [`NOTICE.md`](NOTICE.md); ported test cases are marked in-file.
