# Qwen2.5-0.5B — SAE training notes

## v0.1-smoke (2026-05-18)

First end-to-end pipeline verification. Trained a TopK SAE on the layer 9
residual stream of `Qwen/Qwen2.5-0.5B`.

| field            | value     |
|------------------|-----------|
| n_tokens         | 1,000,000 |
| seq_len          | 512       |
| d_in / d_hidden  | 896 / 14336 (16× expansion) |
| k                | 32        |
| epochs           | 4         |
| batch_size       | 4096      |
| lr               | 5e-4      |
| final MSE        | 0.093     |
| final EV         | 0.988     |
| final L0         | 32        |
| wall time        | ~40 s on 1× RTX 4090 |

Checkpoint: [Legiblex/sae-qwen2.5-0.5b-l9](https://huggingface.co/Legiblex/sae-qwen2.5-0.5b-l9)

### What this run does and does not tell us

- It tells us the pipeline runs end-to-end: hook capture, activation buffer,
  TopK forward, decoder renormalization, save/load round-trip.
- It does **not** tell us we have a usable feature dictionary. EV 0.988 on
  1M tokens almost certainly reflects memorization of the activation
  distribution rather than learned features. A meaningful run needs ≥100M
  tokens and the EV-vs-step curve should show extended plateau behavior.

### Next steps

- [ ] Scale to 100M tokens — at 60k tok/s collection speed, that's ~30 min
      of data collection. Buffer would be ~180 GB in fp16, so collection
      needs to be streamed-to-disk (zarr or memmap) rather than held in RAM.
- [ ] Run a held-out reconstruction eval on a fresh sample of tokens to
      separate train-set memorization from generalization.
- [ ] Compute dead-feature counts (how many of the 14336 features never fire
      across the validation set) — for TopK this should be near zero with
      proper decoder renormalization, but worth verifying.
- [ ] Browse a handful of features by activation strength and check whether
      they look like coherent concepts. If they don't, that's a signal the
      token budget is still too small.
