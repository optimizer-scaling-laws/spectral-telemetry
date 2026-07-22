#!/usr/bin/env python3
"""Self-contained transformer example: masks, buckets, windows, diagnostics.

Runs on CPU in seconds. Demonstrates the full research workflow on a tiny
causal LM: padded variable-length batches (attention masks), a token-frequency
table with HEAD/MID/TAIL buckets, per-forward batch contexts, windowed
measurement with ``compute(reset=True)``, enable/disable scheduling, and the
estimation diagnostics that annotate every record.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise SystemExit("This example needs torch: pip install 'spectral-telemetry[torch]'")

from spectral_telemetry.telemetry import FrequencyTable, attach_probes

VOCAB, DIM, HID, LAYERS, PAD = 512, 32, 128, 2, 0


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(DIM)
        self.fc1 = nn.Linear(DIM, HID)
        self.fc2 = nn.Linear(HID, DIM)

    def forward(self, x):
        return x + self.fc2(torch.relu(self.fc1(self.norm(x))) ** 2)


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DIM)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.head = nn.Linear(DIM, VOCAB, bias=False)

    def forward(self, ids):
        x = self.emb(ids)
        for b in self.blocks:
            x = b(x)
        return self.head(x)


def padded_batch(rng, batch=8, max_len=48):
    """Variable-length sequences padded to max_len; Zipf-ish token draws."""
    ids = torch.full((batch, max_len), PAD, dtype=torch.long)
    mask = torch.zeros(batch, max_len, dtype=torch.bool)
    for i in range(batch):
        n = int(rng.integers(max_len // 2, max_len + 1))
        ids[i, :n] = torch.as_tensor(rng.zipf(1.4, size=n).clip(1, VOCAB - 1))
        mask[i, :n] = True
    return ids, mask


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = TinyLM()
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)

    # Corpus frequency table (here: synthetic Zipf counts over the vocab).
    counts = np.maximum((1e6 / np.arange(1, VOCAB + 1) ** 1.2).astype(np.int64), 1)
    table = FrequencyTable(counts)
    print(table.summary())

    # Probe the FFN hidden activations of every block, with buckets.
    probes = attach_probes(model, select="*.fc1", freq_table=table, expected_hidden_dim=HID)

    for window in range(2):
        for _ in range(5):  # measurement window: 5 training steps
            ids, mask = padded_batch(rng)
            with probes.batch_context(token_ids=ids, sample_mask=mask):
                logits = model(ids)
            loss = F.cross_entropy(
                logits.view(-1, VOCAB)[mask.view(-1)], ids.view(-1)[mask.view(-1)]
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        records = probes.compute(reset=True)  # window-local spectra, then clear
        print(f"\nwindow {window} (loss {loss.item():.3f}):")
        for name, rec in records.items():
            print(
                f"  {name}: soft={rec['soft_rank']:.1f} hard={rec['hard_rank']:.1f} "
                f"N/D={rec['sample_to_dim_ratio']:.1f} [{rec['estimation_status']}]"
            )
            tail = rec["frequency_buckets"]["tail"]
            status = tail.get("status", tail.get("estimation_status"))
            print(f"    tail bucket: n={tail['n_samples']} ({status})")

    probes.disable()  # e.g. skip measurement during evaluation
    ids, mask = padded_batch(rng)
    model(ids)
    assert probes.n_forwards == 0
    probes.close()
    print("\ndone: probes closed, hooks removed")


if __name__ == "__main__":
    main()
