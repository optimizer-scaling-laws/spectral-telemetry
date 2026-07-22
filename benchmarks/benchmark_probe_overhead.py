#!/usr/bin/env python3
"""Benchmark spectral-telemetry's streaming probes against a no-telemetry
baseline and a buffered (store-all-rows, compute-at-log-time) emulation of the
paper repository's tracker.

Modes
-----
baseline           train loop only
buffered           hooks append activation rows to a host-side buffer; at each
                   window boundary the buffer is concatenated, covariance is
                   computed from scratch, and eigenvalues are taken (emulates
                   the paper tracker's cost structure)
streaming          attach_probes, pooled StreamingCovariance
streaming_buckets  streaming + HEAD/MID/TAIL frequency buckets via batch_context

Streaming moves exact-covariance work *into* every step (batch scatter
X^T X per forward); buffered concentrates cost at window boundaries and in
host memory. This benchmark reports both faces: mean step time (total
overhead) and the window-boundary collect cost, plus peak host RSS and peak
CUDA memory.

With ``--mode all`` (default) each mode runs in its own subprocess so peak
RSS is attributable per mode; child stderr is surfaced on failure.

Example (CPU, small):
    python benchmarks/benchmark_probe_overhead.py --steps 30 --log-every 10
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

MODES = ("baseline", "buffered", "streaming", "streaming_buckets")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=MODES + ("all",), default="all")
    p.add_argument("--steps", type=int, default=30, help="timed training steps")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument(
        "--log-every", type=int, default=10, help="window length: compute()/flush every N steps"
    )
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--dim", type=int, default=64, help="model width d_model")
    p.add_argument("--ffn-mult", type=int, default=4, help="probed hidden dim = ffn_mult * dim")
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--vocab", type=int, default=2048)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default=None, help="cpu|cuda (default: auto)")
    p.add_argument(
        "--dtype",
        choices=["auto", "fp32", "bf16"],
        default="auto",
        help="'auto' = bf16 autocast on cuda, fp32 on cpu",
    )
    p.add_argument(
        "--accum-device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="streaming accumulator device (library default: cpu)",
    )
    p.add_argument(
        "--accum-dtype",
        choices=["fp64", "fp32"],
        default="fp64",
        help="streaming accumulator dtype (library default: fp64)",
    )
    p.add_argument("--json-out", default=None)
    return p.parse_args(argv)


def peak_rss_mb() -> float:
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ---------------------------------------------------------------- tiny model
def build_model(args, torch):
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, d, h):
            super().__init__()
            self.norm = nn.LayerNorm(d)
            self.fc1 = nn.Linear(d, h)
            self.fc2 = nn.Linear(h, d)

        def forward(self, x):
            y = self.fc1(self.norm(x))
            y = torch.relu(y) ** 2
            return x + self.fc2(y)

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            d, h = args.dim, args.dim * args.ffn_mult
            self.emb = nn.Embedding(args.vocab, d)
            self.blocks = nn.ModuleList(Block(d, h) for _ in range(args.layers))
            self.head = nn.Linear(d, args.vocab, bias=False)

        def forward(self, ids, targets):
            x = self.emb(ids)
            for b in self.blocks:
                x = b(x)
            logits = self.head(x).float()
            return torch.nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]), targets.view(-1)
            )

    return TinyLM()


def zipf_table(vocab):
    import numpy as np

    from spectral_telemetry.core.frequency import FrequencyTable

    ranks = np.arange(1, vocab + 1, dtype=np.float64)
    return FrequencyTable(np.maximum((1e7 / ranks).astype(np.int64), 1))


class BufferedEmulator:
    """Old-tracker cost structure: per-module host buffers, recompute at flush."""

    def __init__(self, modules, torch):
        self.torch = torch
        self.rows: dict = {id(m): [] for m in modules}
        self.handles = [m.register_forward_hook(self._grab) for m in modules]

    def _grab(self, module, inputs, output):
        self.rows[id(module)].append(output.detach().reshape(-1, output.shape[-1]).to("cpu"))

    def flush(self):
        t = self.torch
        out = []
        for key, rows in self.rows.items():
            x = t.cat(rows).to(t.float64)
            self.rows[key] = []
            c = x - x.mean(dim=0, keepdim=True)
            cov = (c.t() @ c) / (x.shape[0] - 1)
            out.append(t.linalg.eigvalsh(cov))
        return out

    def close(self):
        for h in self.handles:
            h.remove()


def run_one_mode(args) -> dict:
    import torch

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.dtype == "bf16" or (args.dtype == "auto" and device == "cuda"):
        precision, amp = "bf16", torch.amp.autocast(device_type=device, dtype=torch.bfloat16)
    else:
        precision, amp = "fp32", contextlib.nullcontext()

    model = build_model(args, torch).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    hidden = args.dim * args.ffn_mult

    probes, buffered, table = None, None, None
    if args.mode in ("streaming", "streaming_buckets"):
        from spectral_telemetry.torch_backend.probe import attach_probes

        table = zipf_table(args.vocab) if args.mode == "streaming_buckets" else None
        probes = attach_probes(
            model,
            select=lambda name, m: name.endswith("fc1"),
            freq_table=table,
            expected_hidden_dim=hidden,
            accumulation_device=args.accum_device,
            dtype=torch.float64 if args.accum_dtype == "fp64" else torch.float32,
        )
    elif args.mode == "buffered":
        buffered = BufferedEmulator([b.fc1 for b in model.blocks], torch)

    def batch():
        ids = torch.randint(0, args.vocab, (args.batch, args.seq), device=device)
        tgt = torch.randint(0, args.vocab, (args.batch, args.seq), device=device)
        return ids, tgt

    collect_times: list[float] = []

    def one_step(step: int):
        opt.zero_grad(set_to_none=True)
        ids, tgt = batch()
        if probes is not None and table is not None:
            probes.set_batch_context(token_ids=ids)
        with amp:
            loss = model(ids, tgt)
        loss.backward()
        opt.step()
        if step % args.log_every == 0:
            t0 = time.perf_counter()
            if probes is not None:
                probes.compute(reset=True)
            elif buffered is not None:
                for b in model.blocks:
                    pass
                buffered.flush()
            if device == "cuda":
                torch.cuda.synchronize()
            collect_times.append(time.perf_counter() - t0)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step in range(1, args.warmup + 1):
        one_step(step)
    if device == "cuda":
        torch.cuda.synchronize()
    collect_times.clear()

    times = []
    first = args.warmup + 1
    for step in range(first, first + args.steps):
        t0 = time.perf_counter()
        one_step(step)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    if probes is not None:
        probes.close()
    if buffered is not None:
        buffered.close()

    tokens_per_step = args.batch * args.seq
    mean_s = statistics.fmean(times)
    result = {
        "mode": args.mode,
        "device": device,
        "precision": precision,
        "accum_device": args.accum_device,
        "accum_dtype": args.accum_dtype,
        "steps": args.steps,
        "warmup": args.warmup,
        "log_every": args.log_every,
        "batch": args.batch,
        "seq": args.seq,
        "dim": args.dim,
        "ffn_mult": args.ffn_mult,
        "probed_hidden_dim": hidden,
        "layers": args.layers,
        "vocab": args.vocab,
        "mean_step_s": mean_s,
        "median_step_s": statistics.median(times),
        "tokens_per_s": tokens_per_step / mean_s,
        "collect_s_per_window": statistics.fmean(collect_times) if collect_times else 0.0,
        "n_windows": len(collect_times),
        "peak_rss_mb": peak_rss_mb(),
    }
    if device == "cuda":
        result["peak_cuda_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
    return result


def run_all(args) -> list[dict]:
    results = []
    for mode in MODES:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = f.name
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--mode",
            mode,
            "--steps",
            str(args.steps),
            "--warmup",
            str(args.warmup),
            "--log-every",
            str(args.log_every),
            "--batch",
            str(args.batch),
            "--seq",
            str(args.seq),
            "--dim",
            str(args.dim),
            "--ffn-mult",
            str(args.ffn_mult),
            "--layers",
            str(args.layers),
            "--vocab",
            str(args.vocab),
            "--seed",
            str(args.seed),
            "--dtype",
            args.dtype,
            "--accum-device",
            args.accum_device,
            "--accum-dtype",
            args.accum_dtype,
            "--json-out",
            json_path,
        ]
        if args.device:
            cmd += ["--device", args.device]
        proc = subprocess.run(
            cmd, cwd=str(REPO_ROOT), text=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        if proc.returncode != 0:
            tail = "\n".join((proc.stderr or "").strip().splitlines()[-30:])
            print(
                f"\n[{mode}] child failed (exit {proc.returncode}). stderr tail:\n{tail}",
                file=sys.stderr,
            )
            raise SystemExit(proc.returncode)
        with open(json_path) as f:
            results.append(json.load(f))
        os.unlink(json_path)
    return results


def print_table(results: list[dict]) -> None:
    base = next(r for r in results if r["mode"] == "baseline")
    hdr = (
        f"{'mode':<18} {'mean ms/step':>12} {'median ms':>10} {'overhead %':>11} "
        f"{'collect s/win':>13} {'peak RSS MB':>12}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        ovh = 100.0 * (r["mean_step_s"] - base["mean_step_s"]) / base["mean_step_s"]
        line = (
            f"{r['mode']:<18} {1e3 * r['mean_step_s']:>12.2f} "
            f"{1e3 * r['median_step_s']:>10.2f} {ovh:>10.2f}% "
            f"{r['collect_s_per_window']:>13.3f} {r['peak_rss_mb']:>12.1f}"
        )
        if "peak_cuda_mb" in r:
            line += f"  cuda {r['peak_cuda_mb']:.1f} MB"
        print(line)
    b = base
    print(
        f"\nconfig: device={b['device']} precision={b['precision']} dim={b['dim']} "
        f"ffn_mult={b['ffn_mult']} (probed hidden {b['probed_hidden_dim']}) "
        f"layers={b['layers']} seq={b['seq']} batch={b['batch']} steps={b['steps']} "
        f"log_every={b['log_every']} accum={b['accum_device']}/{b['accum_dtype']}"
    )
    print(
        "note: streaming pays exact-covariance accumulation inside every step; "
        "buffered concentrates cost and memory at window boundaries. Compare "
        "TOTAL overhead (mean-step column), not only collect time."
    )


def main():
    args = parse_args()
    if args.mode == "all":
        results = run_all(args)
        print_table(results)
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(results, f, indent=2)
    else:
        result = run_one_mode(args)
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(result, f, indent=2)
        else:
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
