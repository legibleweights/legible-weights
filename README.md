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
| [small-sae-bench](https://github.com/legibleweights/small-sae-bench) | **v0.4.3:** Register-Subtracted TopK SAE wins on CE recovery in **6 of 8** (model, layer) configs across 3 small open transformers (gains 0.3 to +3.9 pts), ties once, regresses once. **The single regression is predicted by the eraser-mechanism rule**: at layers one-step-before an attention-head-mediated eraser (GPT-2 L11), RS regresses; at layers one-step-before MLP-mediated erasers (Qwen L21, Pythia L23), RS wins. This **bridges all three project repos**: the mechanistic finding (eraser type) from `outlier-position-anatomy` v0.5 *predicts* the architectural result. Practitioner rule: use RS unless one step before an attention-head eraser. |
| [outlier-position-anatomy](https://github.com/legibleweights/outlier-position-anatomy) | **v0.7:** The position-0 attention-sink is a load-bearing 3-component write-and-erase circuit maintaining a **fixed scaffolding vector** — not memory. Per-component dissection across 3 models shows writers are always MLP-dominated; erasers split: **MLP-mediated in Qwen L21 and Pythia L23, attention-head-mediated in GPT-2 L11 (head 8 = 77%)**. This eraser-mechanism distinction *predicts* the one configuration where `small-sae-bench` v0.4.3's Register-Subtracted SAE regresses (GPT-2 L10, one before the attention eraser). Mechanism → engineering → predictive theory in one loop. |
| [tunable-residual-finetune](https://github.com/legibleweights/tunable-residual-finetune) | **v0.1:** A single learnable residual-stream vector (**896 params, 0.0002 % of Qwen2.5-0.5B**) inserted at one mid-sequence position of one layer reduces GSM8K loss by **8.7 %** while leaving FineWeb-Edu loss unchanged (+0.14 %, within noise) — a *tiny task-conditioning vector* that touches no weight matrix and uses about half the parameter budget of the smallest possible LoRA. Position 0 (the fixed attention-sink register from `outlier-position-anatomy`) is tunable only weakly (5.5 %); the sweet spot is positions 30-50 (8.5 %); positions ≥ 100 progressively less tunable. Multi-position tuning does **not** compound (replacing multiple per-input residuals with shared learnable vectors damages the model). Cleanly tests and partially refutes the "tune the position-0 register" hypothesis motivated by `outlier-position-anatomy` v0.7. |

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
