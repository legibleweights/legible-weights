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
| wall time        | ~40 s on 1× RTX 4090 |

Checkpoint: [legible-weights/sae-qwen2.5-0.5b-l9](https://huggingface.co/legible-weights/sae-qwen2.5-0.5b-l9)

### Training metrics (final step)

- MSE 0.093, EV 0.988, L0 32.

### Held-out evaluation (100k fresh tokens from a later slice of FineWeb-Edu)

| metric                          | value           |
|---------------------------------|-----------------|
| MSE                             | 0.103           |
| Explained variance              | 0.987           |
| L0 (mean ± std)                 | 32.0 ± 0.0      |
| Dead features                   | 6 / 14336 (0.04%) |
| CE clean (no intervention)      | 2.709           |
| CE w/ SAE reconstruction spliced| 3.119 (+0.410)  |
| CE w/ mean-activation baseline  | 10.910          |
| **CE recovered**                | **0.950**       |

Held-out EV essentially matches training EV → not overfit. CE recovery of
**0.950** is the load-bearing number: splicing the SAE's reconstruction in
place of the real layer-9 residual stream costs only 5% of the next-token
predictive signal that the model would otherwise have. For a 1M-token, 16×
expansion, k=32 SAE on a 0.5B model this is unexpectedly usable — earlier
note in this file calling it a "memorization artifact" was wrong, the eval
demonstrably refutes it.

### Next steps

- [ ] Scale to 50–100M tokens with streamed-to-disk activations (180GB fp16
      buffer doesn't fit RAM, so memmap on the NVMe). Goal: push CE recovery
      above 0.97 and harden against any remaining variance from data slice.
- [ ] Inspect top features: pick 8–10 high-firing features and find the
      top-k activating tokens for each. If they cluster into coherent
      concepts (a date format, a code construct, a sentiment polarity) we
      have something publishable. If they look like noise, the smoke is
      not as good as the metrics suggest.
- [ ] Replicate on layer 5 and layer 15 to compare mid-vs-late residual
      streams; pick the most interpretable layer for the first blog post.
- [ ] Train an MLP-output SAE (vs. residual-stream) at the same layer for a
      coverage comparison.
