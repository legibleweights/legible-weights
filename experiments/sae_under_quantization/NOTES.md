# SAE features under quantization — Qwen2.5-0.5B

**Date:** 2026-05-18 (v0.1)

## Question

Sparse autoencoders are increasingly used as deployment-time interpretability
and steering tools for LLMs. In deployment, those LLMs are almost always
**quantized** (int8 / nf4 / int4 GPTQ / AWQ / …) to reduce memory and
latency. The SAE itself is typically trained on full-precision (fp16 /
bf16) activations.

Open empirical question: **does a fp16-trained SAE's feature dictionary
still describe the quantized model's residual stream?**

Concretely: if a feature fires on token `t` in the fp16 forward pass with
activation value `v_fp16`, what's the corresponding activation value
`v_quant` in the *quantized* forward pass through the *same* SAE? Are the
firings still on the same tokens? Do some features go silent?

The LessWrong post "Monosemanticity and Quantization" explicitly flags this
as unexplored. One 2025 arXiv paper (`arxiv:2504.04215`) examines safety
*circuits* under compression, but not SAE features. No public artifact at
the sub-1B-model scale.

## Setup

- **Base model**: `Qwen/Qwen2.5-0.5B`, layer 9 residual stream.
- **SAE**: `legible-weights/sae-qwen2.5-0.5b-l9` v0.2 — TopK k=32, d_hidden
  14336 (16× expansion), trained on 10 M fp16 activations from FineWeb-Edu
  with exclude_first_n=4. Held-out CE recovered 0.98 (see Qwen2.5
  NOTES.md).
- **Quantization variants** (via `bitsandbytes` 0.49.2):
  - `fp16` — reference
  - `int8` — `load_in_8bit=True` (linear int8 with smooth scaling)
  - `nf4` — `load_in_4bit=True` + `bnb_4bit_quant_type='nf4'`
    + `bnb_4bit_compute_dtype=torch.float16`
- **Held-out data**: 50,000 layer-9 activations + 8 CE-recovery batches
  from FineWeb-Edu (offsets disjoint from training and disjoint per metric).
- **Same texts** through all three precisions, *same SAE checkpoint* for
  encoding.

## Results

### Reconstruction & CE recovery (all through the fp16 SAE)

| precision | MSE    | EV    | L0   | Dead | CE clean | CE recon | CE recovered |
|-----------|--------|-------|------|------|----------|----------|---------------|
| fp16      | 0.033  | 0.846 | 32.0 | 225  | 2.693    | 2.821    | **0.9815**    |
| int8      | 0.033  | 0.845 | 32.0 | 219  | 2.702    | 2.830    | **0.9816**    |
| nf4       | 0.036  | 0.834 | 32.0 | 236  | 2.832    | 2.961    | **0.9817**    |

Reconstruction quality on the quantized models' activations through the
fp16 SAE is essentially identical to reconstruction on fp16 activations
themselves. CE recovery is invariant across precisions: even though both
the clean and SAE-spliced losses *rise* under nf4 (from 2.69 to 2.83 nats
absolute), they rise in lockstep, so the *fraction of next-token signal
preserved by the SAE* stays at 0.98.

### Per-feature stability against the fp16 reference

For features active at fp16 (defined as firing on ≥ 30 of the 50,000
tokens — 7,485 of 14,336 features qualify):

| precision | stable (r > 0.9)   | drifted (0.5 < r ≤ 0.9) | died (r ≤ 0.5) | median r | mean r |
|-----------|--------------------|-------------------------|-----------------|----------|--------|
| int8      | **7,482 (99.96%)** | 3                       | **0**           | 0.991    | 0.988  |
| nf4       | **6,539 (87.4%)**  | 946 (12.6%)             | **0**           | 0.961    | 0.950  |

Among the 946 drifted features under nf4, the correlation distribution is
heavily skewed toward the top of the bucket:

| r range       | count |
|---------------|-------|
| (0.5, 0.6]    | 0     |
| (0.6, 0.7]    | 10    |
| (0.7, 0.8]    | 44    |
| (0.8, 0.9]    | 892   |

**The minimum activation correlation across all 7,485 fp16-active features
under nf4 quantization is r = 0.618.** No feature in the dictionary
"dies" under nf4 — every feature that fires on fp16 also fires on the
quantized model, with at worst moderate amplitude drift.

## Falsifiable claim

**On Qwen2.5-0.5B layer 9, a fp16-trained TopK sparse autoencoder transfers
to int8 and nf4 quantized versions of the same base model with no feature
deaths, ≥ 87 % of active features retaining Pearson r > 0.9 against the
fp16 reference at nf4 (≥ 99.96 % at int8), and SAE-spliced CE recovery
unchanged at 0.98 across all three precisions.**

In plain terms: SAE-based interpretability and steering tools developed
against full-precision weights transfer directly to int8 and nf4 deployment
at this model scale, with only mild amplitude drift on a minority of
features.

## What's NOT in this v0.1

- **One model, one layer.** Qwen2.5-0.5B at L9 specifically. The same
  experiment on other depths and other architectures (Llama-3.2-1B, GPT-2,
  …) is the obvious extension and might show that *late*-layer features
  (where the residual stream is high-magnitude and more sensitive to
  rounding) are less robust than mid-layer ones.
- **Two precisions below fp16.** We tested int8 and nf4. We did not test
  GPTQ / AWQ / SmoothQuant (which use calibration data and might preserve
  features better), nor int2 / ternary / BitNet-1.58 (which would
  certainly stress the dictionary harder). bitnet-style 1.58-bit weights
  are the natural stress test for "where does this break".
- **No per-feature taxonomy.** We did not ask whether the 946 nf4-drifted
  features share a *kind* (e.g., morphological suffixes, high-frequency
  function words, low-density semantic features). Doing so would require
  re-running the feature inspector on both precisions and grouping. v0.2
  follow-up.
- **No causal validation.** We measured firing-pattern correlation, not
  whether ablating feature X has the same downstream effect on the fp16
  and nf4 models. That's the next claim ("dictionary transfers in *what
  it encodes*" → "dictionary transfers in *what it causes*").

## Public artifacts

- `report.json` — the headline metrics
- `per_feature_corr_int8.npy`, `per_feature_corr_nf4.npy` — per-feature
  Pearson correlation against fp16, shape (14336,)
- `feature_active_mask_int8.npy`, `feature_active_mask_nf4.npy` —
  boolean masks marking which features were "active" at the fp16 reference
- This NOTES.md
- All code in `experiments/sae_under_quantization/` and
  `src/legible_weights/`

The SAE checkpoint used as the reference dictionary is the existing
[legible-weights/sae-qwen2.5-0.5b-l9](https://huggingface.co/legible-weights/sae-qwen2.5-0.5b-l9)
v0.2 — no new SAE was trained for this experiment.
