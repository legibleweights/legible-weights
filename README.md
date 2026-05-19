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
| [small-sae-bench](https://github.com/legibleweights/small-sae-bench) | **v0.4:** Register-Subtracted TopK SAE is a **strict Pareto-improvement** over vanilla TopK across 3 models × 5 (model, layer) configs: never regresses on CE recovery, gives perfect prefix reconstruction (EV ≥ 0.9997 across all configs vs TopK's −0.47 to +0.94 range), pays 0.2–1.5pt mid-sequence EV cost. CE-recovery gain over TopK scales with how badly TopK's prefix is broken: +2.4pts at Qwen2.5-0.5B L15 where TopK collapses (−0.47 EV) → tied at Pythia-1.4B L12 where TopK isn't broken. Zero extra learnable parameters — uses the fixed register direction from `outlier-position-anatomy`. |
| [outlier-position-anatomy](https://github.com/legibleweights/outlier-position-anatomy) | The position-0 attention-sink in small open transformers is a load-bearing 3-component write-and-erase circuit (writers in early layers + an eraser before the output) that maintains a **fixed scaffolding vector** — not memory. Cross-architecture: 99.8–99.99% of register energy is in the input-independent mean across Qwen2.5-0.5B / GPT-2 small / Pythia-1.4B (mean pairwise cosine ≥ 0.9996 over hundreds of inputs each). MLPs do most of the work, not attention heads. Ablating the writers costs +3.3 nats on Qwen — the model has *learned to depend on* the fixed anchor. |

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
