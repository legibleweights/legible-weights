# Diffusion vs. autoregressive — matched-scale SAE feature comparison

**Date:** 2026-05-18 (v0.1, draft, training in progress)

## Question

If you train sparse autoencoders on the residual streams of two transformers
that are matched on **architecture, scale, and training corpus** but differ
only in **training objective** — autoregressive next-token prediction vs.
discrete masked-diffusion denoising — how similar are the feature
dictionaries they learn?

## Matched pair

|                   | GPT-2 small                          | MDLM-OWT                              |
|-------------------|--------------------------------------|---------------------------------------|
| HuggingFace ID    | `openai-community/gpt2`              | `kuleshov-group/mdlm-owt`             |
| Reference         | Radford et al. 2019                  | Sahoo et al. NeurIPS 2024             |
| Training objective| Causal next-token (LM loss)          | Discrete masked diffusion (substitution-based) |
| Blocks            | 12                                   | 12                                    |
| Hidden dim        | 768                                  | 768                                   |
| Heads             | 12                                   | 12                                    |
| Tokenizer         | GPT-2 BPE (vocab 50257)              | GPT-2 BPE + 1 mask token (vocab 50258) |
| Training corpus   | WebText                              | OpenWebText                           |
| Attention         | Causal                               | Bidirectional                         |
| Time conditioning | (n/a)                                | False on this checkpoint              |
| Total params      | 124 M                                | 130 M (non-embedding)                 |

WebText is OpenAI's closed-source corpus; OpenWebText is the open
recreation. The two are drawn from the same distribution (Reddit-submitted
URLs ≥ 3 karma), with different scraping passes. As close to a controlled
corpus as we can get without retraining one of them.

## Methodology

Both SAEs trained with identical hyperparameters on activations collected
from layer 6 (mid-depth of 12) of each base model on OpenWebText:

- d_in = 768 (matched)
- d_hidden = 12 288 (16× expansion)
- TopK with k = 32
- 10 M tokens of OpenWebText activations
- `exclude_first_n = 4` (skip outlier prefix positions, same as Qwen2.5 v0.2)
- Adam, lr 5e-4, batch 4 096, 4 epochs
- For MDLM: noise level t = 0 (clean text). This MDLM checkpoint has
  `time_conditioning = False` so the value is mechanically inert; the only
  effect is what fraction of input tokens are replaced by `[MASK]`. We use
  zero. Documenting the choice because it is the obvious axis of variation
  someone reproducing this would ask about.

Dictionary alignment (`src/legible_weights/eval/alignment.py`):

1. **Decoder cosine similarity** over all (feat_A, feat_B) pairs of decoder
   rows. *Methodological caveat:* the two 768-dim residual streams are not
   the same vector space, so high decoder cosine alone is not principled
   evidence of feature equivalence. We report it as a diagnostic.
2. **Activation correlation** on a shared OpenWebText slice (held-out from
   training data). Same texts fed through both base models, both SAEs'
   feature activations gathered at matched positions in matched order.
   Per-pair Pearson correlation. **This is the load-bearing metric.**
3. **Hungarian assignment** for one-to-one matching.

## Results

### Training metrics (final step)

| metric        | GPT-2 small SAE | MDLM SAE |
|---------------|-----------------|----------|
| Final MSE     | 0.85            | 5.58     |
| Final EV      | 0.86            | 0.90     |
| Final L0      | 32              | 32       |
| Wall-time     | ~3 min          | ~3 min   |

MSE differs by ~6× because MDLM activations have ~2× the mean-absolute
magnitude (mean-abs ~4.3 vs. ~2.0). EV is scale-normalized and comparable —
both SAEs explain about the same fraction of their respective activation
variance.

### Dictionary alignment

**Matching basis: activation correlation on 30,000 held-out OpenWebText
tokens.** Decoder-row cosine is reported as a diagnostic only — see the
methodological caveat above.

| metric                                       | value |
|----------------------------------------------|-------|
| Active features (fire ≥30× on 30k tokens)    | GPT-2: 4,697 / 12,288 (38%)  ·  MDLM: 3,715 / 12,288 (30%) |
| Max activation correlation                   | **0.989**                                                  |
| Median activation correlation                | 0.332                                                      |
| Median decoder cosine                        | −0.002                                                     |
| Max decoder cosine                           | 0.133                                                      |

Hungarian-matched pair bucketing (3,715 pairs, one per MDLM active
feature):

| bucket                          | count       | share |
|---------------------------------|-------------|-------|
| **Shared** (act_corr > 0.30)    | **2,043**   | **55 %** |
| Weak (0.15 < act_corr ≤ 0.30)   | 957         | 26 %  |
| Divergent (act_corr ≤ 0.15)     | 715         | 19 %  |

### Feature inspection of representative pairs

Top 12 pairs by activation correlation (all r > 0.98) are **all** closed-
class function-word or punctuation features that fire on identical
contexts across the two models, including in several cases on
*literally the same activating sentence* drawn from the eval slice:

| rank | act_corr | concept (consistent across both sides) |
|------|----------|-----------------------------------------|
| 1    | 0.989    | ` which` (relative pronoun)             |
| 2    | 0.988    | ` the` (definite article)               |
| 3    | 0.987    | `↵` (newline)                           |
| 4    | 0.986    | ` with` / ` With`                       |
| 5    | 0.985    | ` but`                                  |
| 6    | 0.984    | ` come`                                 |
| 7    | 0.984    | ` from` / ` From`                       |
| 8    | 0.984    | ` there` / ` There`                     |
| 9    | 0.984    | ` or`                                   |
| 10   | 0.984    | ` by`                                   |
| 11   | 0.983    | ` when` / ` When`                       |
| 12   | 0.983    | ` we` / ` We`                           |

Decoder cosines for these pairs are essentially zero (range −0.106 to
+0.064), confirming that **the geometric encoding of each concept differs
completely between the two models even though the activation pattern is
identical**.

Bottom-4 pairs (act_corr ≈ 0.04) reveal architecture-specific features. The
most coherent example is GPT-2 feature 10847, which cleanly captures
financial-lending verbs (`borrowing`, `loans`, `borrow`), where Hungarian
assigned it MDLM feature 1889 as a best match — but MDLM 1889 fires on
heterogeneous tokens (`large`, `that`, `collection`, `I`) with no shared
theme. Likely interpretation: the "financial-lending" feature exists in
GPT-2 but not at all in MDLM at this layer/scale, and the matching algorithm
reports the closest available non-match.

See `paired_features_v0.1.md` for the full activating-context tables.

## Falsifiable claim (committed after data)

**At matched scale (~125 M parameters), matched depth (layer 6 of 12),
matched corpus (OpenWebText), and matched SAE recipe (TopK k = 32, 16×
expansion, 10 M tokens, exclude_first_n = 4), 55 % of MDLM SAE features
have a GPT-2 SAE counterpart with Pearson r > 0.30 on 30 k held-out tokens.
Decoder-row cosine across the two SAEs is centered at zero (median
−0.002, max +0.133), so the shared features occupy geometrically
unrelated subspaces of the two models' residual streams.**

In plain terms: causal next-token prediction and bidirectional masked-
diffusion denoising, trained on the same data at the same scale, build
**the same concepts in different coordinate systems**. The geometry of the
representation is paradigm-specific; the inventory of represented concepts
is substantially shared.

The top-correlated pairs are dominated by closed-class function words and
punctuation; the divergent bucket is the more interesting one for future
work, since it is where genuine paradigm-specific features would live.

## What's NOT in this v0.1

- Only **one layer** (mid-depth, layer 6 of 12) and **one pair of models**.
  The claim is layer-6-specific; depth-resolved comparison is a natural
  follow-up.
- Only **one SAE recipe**. Different `k`, expansion factors, or training
  objectives could change the overlap fraction.
- Only **one diffusion noise level** (t = 0). The shipped MDLM has
  time_conditioning = False so t doesn't enter the forward pass, but
  collecting at non-zero t (partially-masked inputs) probes a different
  computational regime — also a natural follow-up.
- No causal experiments: we have not yet tested whether ablating a shared
  feature has analogous behavioral effects in both models. That would be the
  next-step claim ("shared in *what they encode*" → "shared in *what they
  use*").

## Public artifacts (planned)

- `legible-weights/sae-gpt2-small-l6-v0.1` on HuggingFace
- `legible-weights/sae-mdlm-owt-l6-v0.1` on HuggingFace
- `legible-weights/legible-weights` on GitHub — full code + writeup
- This NOTES.md as the formal report
