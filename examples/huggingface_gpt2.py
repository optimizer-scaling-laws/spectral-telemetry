#!/usr/bin/env python3
"""GPT-2 example (requires `pip install transformers`; downloads the model).

Probes the input to every block's ``mlp.c_proj`` -- the post-activation FFN
features analyzed in the paper -- on real tokenized text with attention-mask
filtering and HEAD/MID/TAIL buckets. The frequency table here is built from
the sample corpus itself, which is tiny; expect a degeneracy warning from
``FrequencyTable`` diagnostics on such small corpora, and substitute corpus-
level counts (e.g. token_frequencies.npy from the paper repo) for real use.

Not executed in CI (network + model download); run locally:
    python examples/huggingface_gpt2.py
"""

import numpy as np

try:
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
except ImportError:
    raise SystemExit("This example needs transformers: pip install transformers")

from spectral_telemetry.telemetry import FrequencyTable, attach_probes

TEXTS = [
    "The optimizer determines how much of the architecture's capacity is realized.",
    "Two checkpoints can match in loss while differing sharply in spectral rank.",
    "Rare tokens live in the tail of the frequency distribution.",
    "Effective rank summarizes how many directions the representation uses.",
] * 8


def main():
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()

    enc = tok(TEXTS, return_tensors="pt", padding=True)
    ids, mask = enc["input_ids"], enc["attention_mask"].bool()

    counts = np.bincount(ids[mask].reshape(-1).numpy(), minlength=len(tok))
    table = FrequencyTable(np.maximum(counts, 0) + (counts.sum() == 0))
    print(table.summary())  # tiny corpus -> likely degenerate; that is the point

    probes = attach_probes(
        model,
        select="transformer.h.*.mlp.c_proj",
        capture="input",  # post-activation FFN features
        freq_table=None if table.is_degenerate else table,
        expected_hidden_dim=4 * model.config.n_embd,
    )
    with torch.no_grad():
        if table.is_degenerate:
            with probes.batch_context(sample_mask=mask):
                model(ids)
        else:
            with probes.batch_context(token_ids=ids, sample_mask=mask):
                model(ids)

    for name, rec in probes.compute().items():
        print(
            f"{name}: soft={rec['soft_rank']:.1f} hard={rec['hard_rank']:.1f} "
            f"N/D={rec['sample_to_dim_ratio']:.2f} [{rec['estimation_status']}]"
        )
    probes.close()


if __name__ == "__main__":
    main()
