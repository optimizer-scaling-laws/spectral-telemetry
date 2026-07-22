import math

from spectral_telemetry.core.schema import parse_layer_metric_line, parse_metric_pairs


def test_legacy_line_normalizes_and_derives_soft_rank():
    row = parse_layer_metric_line("Step 100: SE_post=2.0 PR_post=5.0 EEE_post=9.0 JS=1.0")
    assert row["step"] == 100
    assert row["hard_rank_post"] == 5.0
    assert row["soft_rank_post"] == math.exp(2.0)
    assert "EEE_post" not in row and "JS" not in row


def test_released_names_pass_through():
    row = parse_metric_pairs("soft_rank_pre=7.5 hard_rank_pre=4.0")
    assert row == {"soft_rank_pre": 7.5, "hard_rank_pre": 4.0}


def test_non_metric_line_returns_none():
    assert parse_layer_metric_line("no step marker here") is None
