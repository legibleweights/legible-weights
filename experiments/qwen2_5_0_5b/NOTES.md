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

### Feature inspection

256 sequences × 256 tokens of held-out FineWeb-Edu, top-16 features by total
activation mass.

**With first-4-positions included** (`features_v0.1.md`):

All 16 top features show the *same* top-5 activating contexts in slightly
shuffled order, all with activations in a tight 380–416 range. This is the
classic **outlier-position / attention-sink** phenomenon — the first content
token of each sequence has anomalously high residual-stream norm, and a
naive SAE spends a large fraction of its features reconstructing those few
dominant directions. The high EV and CE-recovered numbers are partly being
earned on these positions rather than on learned concepts.

**Excluding the first 4 positions** (`features_v0.1_excl4.md`):

Top features become individually distinguishable and several are
clearly interpretable. Tentative labels (single-pass, not yet replicated):

| feature | hypothesis                                          |
|---------|-----------------------------------------------------|
| 1131    | end-of-sentence period followed by newline          |
| 12348   | title-case / headline-style capitalized word        |
| 6934    | paragraph-initial content token                     |
| 5940    | preposition ` of` in noun-phrase post-modifiers     |
| 2795    | morphological suffix / subword-completion (`iers`, `ages`, `s`) |
| 1063 / 14233 | two distinct ` ,` features in different syntactic contexts |
| 2204 / 11944 | two distinct ` the` features in different noun-class contexts |
| 11480   | copular / auxiliary ` are`, ` is`                   |
| 7215    | tokens within a single specific Rosa Parks passage  |
| 6845    | ` the` again (a third splinter — feature redundancy) |

The fact that ` the` and ` ,` show up under multiple feature IDs suggests
the dictionary has redundancy / partial splits — a real run with more tokens
should pull these into single canonical features.

### Next steps

- [ ] Modify activation collection to **skip the first N positions** of each
      sequence during training (not just inspection). This should produce a
      cleaner dictionary on the next run since SAE capacity won't be spent
      on outlier reconstruction.
- [ ] Scale to 50–100M tokens with streamed-to-disk activations (180GB fp16
      buffer doesn't fit RAM, so memmap on the NVMe).
- [ ] Add a feature-deduplication metric: features whose decoder rows have
      cosine similarity > 0.9 are essentially the same direction split
      across the codebook. Track this number per run.
- [ ] Replicate on layer 5 and layer 15 to compare mid-vs-late residual
      streams; pick the most interpretable layer for the first blog post.
- [ ] Train an MLP-output SAE (vs. residual-stream) at the same layer for a
      coverage comparison.
