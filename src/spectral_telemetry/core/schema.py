"""Legacy-log compatibility parser (NOT this library's wire format).

Parses the paper repository's text logs into the released metric vocabulary
so published runs remain machine-readable. A versioned, typed TelemetryRecord
with JSONL/Parquet serialization is the planned wire format (v0.2).

Port of upstream ``optimizer_ssl.analysis.log_schema`` (MIT). The canonical
vocabulary is the released public one; legacy paper-run names (``SE_*``,
``PR_*``) normalize into it, and retired diagnostics (``EEE``, ``JS``) are
dropped. Torch-free by design.
"""

from __future__ import annotations

import math
import re
from typing import Any

__all__ = [
    "CANONICAL_METRICS",
    "LEGACY_KEY_MAP",
    "IGNORED_LEGACY_KEYS",
    "parse_metric_pairs",
    "parse_layer_metric_line",
]

CANONICAL_METRICS = (
    "soft_rank",
    "hard_rank",
    "spectral_entropy",
    "soft_rank_pre",
    "hard_rank_pre",
    "spectral_entropy_pre",
    "soft_rank_post",
    "hard_rank_post",
    "spectral_entropy_post",
)

_STEP_RE = re.compile(r"Step\s+(?P<step>\d+)\s*:")
_PAIR_RE = re.compile(r"(?P<key>[A-Za-z_]+)=(?P<value>[-+0-9.eE]+)")

LEGACY_KEY_MAP = {
    "SE_pre": "spectral_entropy_pre",
    "SE_post": "spectral_entropy_post",
    "PR_pre": "hard_rank_pre",
    "PR_post": "hard_rank_post",
}
IGNORED_LEGACY_KEYS = {"EEE_pre", "EEE_post", "JS"}


def parse_metric_pairs(text: str) -> dict[str, float]:
    """Parse ``key=value`` pairs, normalizing legacy names; derive soft ranks."""
    row: dict[str, float] = {}
    for match in _PAIR_RE.finditer(text):
        key = match.group("key")
        if key in IGNORED_LEGACY_KEYS:
            continue
        row[LEGACY_KEY_MAP.get(key, key)] = float(match.group("value"))

    if "soft_rank_pre" not in row and "spectral_entropy_pre" in row:
        row["soft_rank_pre"] = math.exp(float(row["spectral_entropy_pre"]))
    if "soft_rank_post" not in row and "spectral_entropy_post" in row:
        row["soft_rank_post"] = math.exp(float(row["spectral_entropy_post"]))
    return row


def parse_layer_metric_line(line: str) -> dict[str, Any] | None:
    """Parse one layer-metric log line (released or legacy schema)."""
    step_match = _STEP_RE.search(line)
    if not step_match:
        return None
    row: dict[str, Any] = {"step": int(step_match.group("step"))}
    row.update(parse_metric_pairs(line))
    return row
