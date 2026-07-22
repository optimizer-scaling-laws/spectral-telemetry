# spectral-telemetry

Streaming spectral-capacity telemetry for neural-network activations: Shannon
effective rank ("soft rank") and participation ratio ("hard rank") of centered
activation-covariance eigenspectra. The exact accumulator uses **O(D²) memory
per accumulator — independent of the number of streamed samples** — and
supports algebraically exact merging across chunks, frequency buckets, and
data-parallel shards. Extracted from the paper
[*Same Architecture, Different Capacity: Optimizer-Induced Spectral Scaling
Laws*](https://arxiv.org/abs/2605.21803) and its
[paper repository](https://github.com/optimizer-scaling-laws/spectral-scaling-laws),
with numerical conventions preserved so results are directly comparable with
published telemetry.

These metrics characterize **data-conditioned utilization of activation
directions** under a specified model, layer, dataset, masking policy, and
sampling window. They are not an absolute or task-independent measure of
model capacity.

## Install

```bash
pip install -e ".[torch]"     # probes + streaming (needs torch)
pip install -e .              # numpy-only core: ranks, fits, buckets, legacy logs
```

## 60 seconds

```python
import torch
from spectral_telemetry.telemetry import spectral_rank, attach_probes

# One-shot, any activation tensor; mask out padding:
m = spectral_rank(acts, sample_mask=attention_mask)        # [B, T, D] + [B, T]
print(m["soft_rank"], m["hard_rank"], m["estimation_status"])

# Streaming probes — memory independent of token count, O(D^2) per accumulator:
probes = attach_probes(model, select="*.mlp.c_proj", capture="input")
for batch in loader:
    with probes.batch_context(token_ids=batch["input_ids"],
                              sample_mask=batch["attention_mask"]):
        model(**batch)
records = probes.compute(reset=True)     # one record per probed module
probes.close()
```

On CUDA machines pass `accumulation_device="cuda", dtype=torch.float32` to
`attach_probes` — measured below to make per-step cost indistinguishable from
baseline. Batch contexts are **single-use per forward** — armed before, cleared after,
never silently reused. Add `freq_table=FrequencyTable(counts)` to
`attach_probes` for HEAD/MID/TAIL frequency-conditioned spectra (the paper's
tail-capacity metrics). See `examples/transformer_lm.py` for the full
workflow (padding, buckets, measurement windows, scheduling) and
`examples/huggingface_gpt2.py` for a real GPT-2.

## What the numbers mean

With normalized eigenvalues `p_i = λ_i / Σ_j λ_j`:
**R₁ = exp(−Σ p_i log p_i)** (soft rank) is Shannon-balanced effective
dimensionality, sensitive to diffuse spectral mass; **R₂ = 1 / Σ p_i²**
(hard rank) is concentration-sensitive and dominated by larger modes.
`renyi_rank(eigs, α)` generalizes both (α = 1 → soft, α = 2 → hard, α = ∞ →
min-entropy rank), and R₂ ≤ R₁ always. Neither scalar alone is literal
semantic or behavioral capacity.

**Every record annotates its own reliability.** The empirical rank of a
sample covariance is at most `min(D, N − 1)`, so each record carries
`n_samples`, `hidden_dim`, `sample_to_dim_ratio`, `max_empirical_rank`, rank
fractions of both, eigenvalue diagnostics, and an `estimation_status`
(`well_sampled` / `moderately_undersampled` / `severely_undersampled` /
`insufficient_samples`; thresholds documented and configurable — heuristics,
not statistical laws). A low rank at small N/D may reflect the sample count,
not learned compression; the library will not let that ambiguity pass
silently.

## Distributed use

Reduction is exact and scalable: two `all_reduce` phases (global mean, then
between-means-corrected scatters), with peak extra memory of **one D×D tensor
per probe regardless of world size**. Gloo/MPI groups communicate CPU states
directly; NCCL groups stage on the current CUDA device.

```python
# identical probes on every rank; inside your DDP training loop:
records = probes.compute(distributed=True, reset=True)   # pooled + all buckets
```

Probe names, hidden dimensions, and bucketing are validated across ranks
before any collective fires; ranks that never saw a bucket participate with
empty states. Tested in real two-process Gloo runs (unequal counts, unequal
means, empty shards). Cross-rank merging also fixes an estimation problem:
tail statistics can use every rank's samples (N/D ≈ 7 → ≈ 28 in the paper's
4-GPU, 8× width setting) at no additional memory cost.

## Why streaming? (measured)

The paper repository's tracker buffers every activation row on the host,
transfers the buffer back to the device at each log event, and recomputes
covariance from scratch — measured there at 2.6→14.1 s per event (pooled)
across 1×→8× FFN widths, with 1–2 GB of host buffering
([tables](https://github.com/optimizer-scaling-laws/spectral-scaling-laws/blob/main/docs/telemetry_overhead.md)).
Streaming accumulation removes the row storage, the concatenation, the
buffer transfer, and the at-log-time recomputation. **It does not remove the
cost of exact covariance** — the batch scatter `XᵀX` is paid inside every
forward instead — so the honest comparison is total training overhead, which
this repository benchmarks directly against a buffered emulation
(`benchmarks/benchmark_probe_overhead.py`):

| mode | mean ms/step | overhead | collect s/window | peak RSS MB |
|---|---:|---:|---:|---:|
| baseline | 13.7 | — | — | 669 |
| buffered (old design) | 38.6 | +182% | 0.269 | 729 |
| streaming (pooled) | 18.0 | **+31%** | 0.007 | 678 |
| streaming + buckets | 26.6 | +95% | 0.025 | 682 |

*CPU, fp32, d_model 64, probed hidden 256, 2 layers, seq 64, batch 8,
window 10 steps. Small-scale CPU medians are noisy across subprocess runs;
treat totals as the signal.*

**GPU (RTX 3090, bf16 training; library-default accumulation policy
`cpu`/`float64`), probed widths matching the paper's sweep:**

| probed hidden | mode | mean ms/step | collect s/window | peak host RSS MB |
|---:|---|---:|---:|---:|
| 768 (4 probes) | baseline | 13.7 | — | 924 |
| | buffered | 154.3 | 1.24 | 5,571 |
| | streaming | 243.7 | 0.32 | **1,391** |
| | streaming + buckets | 554.4 | 1.52 | 1,444 |
| 3072 (2 probes) | baseline | 7.7 | — | 892 |
| | buffered | 187.0 | 1.70 | 3,263 |
| | streaming | 467.9 | 2.22 | **1,525** |
| | streaming + buckets | 1,751.0 | 10.9 | 1,967 |
| 6144 (2 probes) | baseline | 6.0 | — | 889 |
| | buffered | 599.2 | 6.95 | 3,663 |
| | streaming | 1,753.1 | 13.9 | **2,606** |
| | streaming + buckets | 6,226.6 | 53.1 | 4,344 |

*Configs: 768 → 4 layers, seq 512, batch 16, 30 steps; 3072 → 2 layers,
seq 256, batch 8, 20 steps; 6144 → 2 layers, seq 256, batch 4, 12 steps;
window 10 steps throughout. Buffered was measured with a shared per-window
buffer across layers (one eigendecomposition per window); the shipped
emulator is per-module, so buffered costs above are mild underestimates.*

Two honest readings. First, on a fast GPU **the default accumulation policy,
not the streaming algorithm, dominates total time**: every row is moved to
host as float64 (4× the bytes of the buffered emulation's bf16 copies) and
the exact scatter runs on CPU inside every step, so streaming's wall-clock
exceeds buffered's at all three widths under this policy. Second, what the
default policy still buys, in the same measurements: **2–4× lower peak host
memory at every width**, and memory bounded in window length — the buffered
design's buffer grows linearly with tokens per window (tens of gigabytes per
rank at the paper's production 131k tokens/event and 8× width), which these
short 10-step windows understate. The remedy for the wall-clock is a policy
switch, not a redesign — measured next.

**Same benchmark, recommended GPU policy (`--accum-device cuda
--accum-dtype fp32`):**

| probed hidden | mode | mean ms/step | median ms/step | collect s/window | peak host MB | peak CUDA MB |
|---:|---|---:|---:|---:|---:|---:|
| 768 (4 probes) | baseline | 14.2 | 14.3 | — | 915 | 601 |
| | buffered | 199.7 | 28.9 | 1.70 | 3,095 | 601 |
| | streaming | 47.2 | **14.2** | 0.32 | 997 | 609 |
| | streaming + buckets | 174.6 | 22.2 | 1.53 | 1,005 | 638 |
| 3072 (2 probes) | baseline | 7.3 | 8.0 | — | 892 | 300 |
| | buffered | 270.0 | 14.4 | 2.53 | 2,831 | 300 |
| | streaming | 247.5 | **8.7** | 2.39 | 1,159 | 374 |
| | streaming + buckets | 1,102.8 | 13.7 | 10.9 | 1,179 | 612 |

With on-device accumulation the per-step cost vanishes — **medians match
baseline** — and streaming's remaining cost is the per-window collect
(covariance transfer plus CPU-float64 eigendecomposition), the irreducible
term any design pays, here at or below the buffered emulation's (5× lower at
768; parity at 3072, where the eigendecomposition dominates both). That term
amortizes with the measurement window: `overhead ≈ collect_s / (window_steps
× per_step_s)` — these 10-step windows against 7–14 ms toy baselines are
deliberately adversarial, so treat absolute collect seconds as the
transferable quantity (production step times are one to two orders larger).
On-device accumulators cost the visible CUDA increments (one fp32 D×D per
accumulator), and host RSS stays at ~1.0–1.2 GB versus buffered's
2.8–3.1 GB, with buffered's buffer growing linearly in window length while
streaming's state stays flat. **Recommended GPU policy:**
`attach_probes(..., accumulation_device="cuda", dtype=torch.float32)` — the
PSD-validation tolerance scales with the accumulation dtype.

The merge rule is algebraically exact; in floating point, streaming and batch
covariance agree within the tested tolerance (`atol=1e-10`).

## Limitations

Exact full covariance scales quadratically in hidden dimension: one float64
D×D scatter is ~4.5 MiB at D=768, ~72 MiB at D=3072, ~288 MiB at D=6144 —
multiplied by ~4 with HEAD/MID/TAIL buckets and again by the number of
probed layers. CPU accumulation (the default, explicit in every record) can
synchronize GPU execution in the forward path and is measured to dominate
wall-clock on fast GPUs (see the benchmark tables); prefer
`accumulation_device="cuda"` there. Tensor-parallel detection
covers PyTorch `DTensor` only; other TP frameworks expose plain local
tensors, so users must guarantee full-feature-dimension capture
(`expected_hidden_dim=` catches many, not all, silent shardings). Empirical
ranks are bounded by `min(D, N − 1)` and covariance is data- and
window-dependent; padding and special-token masks must be supplied by the
caller. Supported reduction configurations: Gloo/MPI with CPU states, NCCL
with CUDA staging. Accumulators are not thread-safe. Sub-quadratic/sketched
estimation, asynchronous capture, and TP-aware assembly are roadmap items,
not current features.

## Design notes

* **One primitive, three jobs.** `StreamingCovariance` (Chan–Golub–LeVeque
  pairwise updates) makes chunked capture, per-bucket accumulation, and
  cross-rank reduction the same exact `merge`/reduce operation.
* **Fail near the source.** Non-finite activations raise at `update()` with
  the probe name; materially negative covariance eigenvalues raise rather
  than being silently clipped (threshold scaled to the accumulator dtype so
  float32's rounding floor is admitted; the raw minimum is always reported); frequency tables validate inputs and report
  degenerate buckets instead of pretending balance.
* **Explicit policies.** Accumulation device/dtype, sampling caps, and
  forward counts are stamped into every record; `core/schema.py` is a
  legacy-log compatibility parser for the paper repo's outputs, not this
  library's wire format (a versioned telemetry record is v0.2).
* **Upstream numeric parity.** Eps clamps, entropy/participation-ratio
  conventions, occurrence-balanced tertile boundaries, and t-interval fits
  match the paper repository for valid inputs.

## Roadmap (v0.2+)

Versioned `TelemetryRecord` + JSONL/Parquet serialization with W&B and
TensorBoard sinks; richer sampling/scheduling policies; a grouping protocol
beyond HEAD/MID/TAIL; TP-aware covariance assembly; asynchronous/pinned
capture; optional spectrum summaries; JAX backend on demand.

## Citation

If you use these metrics, please cite the paper:

```bibtex
@article{jha2026optimizer,
  title   = {Same Architecture, Different Capacity: Optimizer-Induced
             Spectral Scaling Laws},
  author  = {Jha, Nandan Kumar and Reagen, Brandon},
  journal = {arXiv preprint arXiv:2605.21803},
  year    = {2026}
}
```

## Provenance & license

MIT. Ported components and preserved conventions are itemized in
[NOTICE.md](NOTICE.md); ported test cases are marked in-file.
