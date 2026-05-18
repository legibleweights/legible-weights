# legible-weights

Interpretability research on small open-weight language models. Umbrella for
a thread of small, focused empirical projects, each living in its own
project repo with a single falsifiable claim and a public, reproducible
artifact.

## Projects

| repo                                                                            | one-line claim                                                                             |
|----------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| [diffusion-vs-ar-saes](https://github.com/legibleweights/diffusion-vs-ar-saes)   | At matched scale and corpus, 55 % of MDLM SAE features have a GPT-2 SAE counterpart at activation-correlation r > 0.30, despite cross-decoder cosine ~ 0 — the two paradigms build the same concepts in different coordinate systems. |
| [sae-under-quantization](https://github.com/legibleweights/sae-under-quantization) | A fp16-trained TopK SAE on Qwen2.5-0.5B L9 transfers losslessly to int8 (99.96 % features stable at r > 0.9) and near-losslessly to nf4 (87.4 %, 0 deaths); SAE-spliced CE recovery unchanged at 0.98. |

## SAE checkpoints (HuggingFace)

All artifacts live under the [`legible-weights` HF organization](https://huggingface.co/legible-weights):

- [`sae-qwen2.5-0.5b-l9`](https://huggingface.co/legible-weights/sae-qwen2.5-0.5b-l9) (v0.1 baseline, v0.2 outlier-aware) — the foundational Qwen2.5-0.5B SAE
- [`sae-gpt2-small-l6-v0.1`](https://huggingface.co/legible-weights/sae-gpt2-small-l6-v0.1) — from [diffusion-vs-ar-saes](https://github.com/legibleweights/diffusion-vs-ar-saes)
- [`sae-mdlm-owt-l6-v0.1`](https://huggingface.co/legible-weights/sae-mdlm-owt-l6-v0.1) — from [diffusion-vs-ar-saes](https://github.com/legibleweights/diffusion-vs-ar-saes)

## What's in this repo

This repo contains the **foundational Qwen2.5-0.5B SAE training and feature-
inspection work**, plus the shared library code (TopK SAE, training loop,
activation collection, model adapters, eval / CE recovery, feature
inspection) that the project repos vendor in.

- `src/legible_weights/` — the shared library
- `experiments/qwen2_5_0_5b/` — original Qwen2.5-0.5B v0.1 and v0.2 SAE
  training, feature inspection, and the outlier-position-trap finding
  (NOTES.md)

When a sub-project matures past v0.1 — e.g. when it has its own narrative
arc, paper, or sustained release cycle — it gets spun out into its own repo
listed above. Vendored copies of the shared library go with it so each
project repo is self-contained and reproducible without depending on the
umbrella.

## Status

Pre-release. The two project repos above are the substantive output so far;
this umbrella is the index + library + the original v0.1/v0.2 Qwen2.5-0.5B
work.

## License

MIT.
