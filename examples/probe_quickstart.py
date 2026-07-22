#!/usr/bin/env python3
"""60-second quickstart: probe any module, streaming, O(D^2) memory."""

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("This example needs torch: pip install 'spectral-telemetry[torch]'")

from spectral_telemetry.telemetry import attach_probes, spectral_rank


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(32, 128), nn.GELU(), nn.Linear(128, 32)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


model = TinyMLP()

x = torch.randn(256, 32)
print("input-space:", spectral_rank(x))

probes = attach_probes(model, select="fc1", capture="output")
for _ in range(10):
    model(torch.randn(64, 32))
for name, m in probes.compute().items():
    print(f"{name}: soft={m['soft_rank']:.2f} hard={m['hard_rank']:.2f} n={m['n_samples']}")
probes.close()
