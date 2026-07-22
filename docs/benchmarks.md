# Benchmark results

This document contains the full benchmark tables summarized in the main README. All measurements
compare a baseline training loop, a buffered emulation of the paper repository's original design,
and the streaming implementation shipped here.

The benchmark script is:

```bash
python benchmarks/benchmark_probe_overhead.py --help
```

## What streaming changes

The paper repository's tracker buffers activation rows on the host, transfers the buffer back to the
device at each log event, and recomputes covariance from scratch. The original implementation was
measured at 2.6–14.1 seconds per pooled event across the paper's 1×–8× FFN sweep, with 1–2 GB of
host buffering in the benchmark configuration. The complete upstream tables are available in the
[paper repository](https://github.com/optimizer-scaling-laws/spectral-scaling-laws/blob/main/docs/telemetry_overhead.md).

Streaming accumulation removes:

- activation-row storage;
- per-window concatenation;
- the buffered host-to-device transfer;
- covariance recomputation from the full row buffer.

It does **not** remove the cost of exact covariance. The batch scatter `XᵀX` is paid during each
forward, and eigendecomposition remains a collection-time cost.

## CPU benchmark

| mode | mean ms/step | overhead | collect s/window | peak RSS MB |
|---|---:|---:|---:|---:|
| baseline | 13.7 | — | — | 669 |
| buffered (old design) | 38.6 | +182% | 0.269 | 729 |
| streaming (pooled) | 18.0 | **+31%** | 0.007 | 678 |
| streaming + buckets | 26.6 | +95% | 0.025 | 682 |

Configuration: CPU, fp32, `d_model=64`, probed hidden dimension 256, two layers, sequence length
64, batch size 8, and a 10-step measurement window. Small CPU medians are noisy across subprocess
runs; treat the totals as the primary signal.

## GPU benchmark: default CPU/float64 accumulation

RTX 3090 with bf16 training and the library-default accumulation policy (`cpu`/`float64`).

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

Configurations:

- 768: four layers, sequence length 512, batch size 16, 30 steps;
- 3072: two layers, sequence length 256, batch size 8, 20 steps;
- 6144: two layers, sequence length 256, batch size 4, 12 steps;
- 10-step measurement windows throughout.

Buffered measurements used a shared per-window buffer across layers and one eigendecomposition per
window. The shipped emulator is per-module, so the buffered costs above are mild underestimates.

### Interpretation

On a fast GPU, the conservative default policy dominates total time: every activation row moves to
host as float64, and the exact scatter runs on CPU inside the forward path. Under this policy,
streaming is slower than the buffered emulation at all three widths.

What the default policy still provides is bounded window memory and 2–4× lower peak host memory in
these runs. The buffered design grows linearly with tokens per window; the reported 10-step windows
understate the gap at larger production logging intervals.

## GPU benchmark: recommended CUDA/fp32 accumulation

Same benchmark with:

```text
--accum-device cuda --accum-dtype fp32
```

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

### Interpretation

With on-device accumulation, median per-step time matches the baseline at hidden dimension 768 and
remains close at 3072. The remaining overhead is concentrated at the measurement-window boundary:
covariance transfer plus CPU/float64 eigendecomposition.

A useful approximation is:

```text
overhead ≈ collect_seconds / (window_steps × baseline_step_seconds)
```

The 10-step windows and 7–14 ms toy baselines are deliberately adversarial. Production steps are
often one to two orders of magnitude longer, so the collection seconds and measurement cadence are
more transferable than the reported percentage overhead.

On-device accumulation requires one fp32 `D × D` matrix per accumulator. In these runs, host RSS
remained approximately 1.0–1.2 GB, compared with 2.8–3.1 GB for the buffered emulation, whose
memory continues growing with window length.

## Recommended policy

For CUDA training, start with:

```python
attach_probes(
    ...,
    accumulation_device="cuda",
    dtype=torch.float32,
)
```

Then choose:

- a limited set of layers;
- a measurement window long enough to amortize collection;
- HEAD/MID/TAIL buckets only where the scientific question requires them;
- enough unmasked samples to make the `N/D` diagnostics meaningful.

The PSD-validation tolerance is scaled to accumulation precision, and every record reports the
actual accumulation device and dtype.
