# legible-weights

Interpretability research on small open-weight language models.

Most public SAE and feature-dictionary work targets a handful of specific models
at the 7B+ scale. This project trains and publishes dictionaries for smaller
open-weight models (0.5B–3B), where iteration is fast and ablations are cheap,
and uses them to trace specific behaviors back to specific weights.

## Output

- **Sparse autoencoders** for residual-stream and MLP-output activations across
  small open-weight LLMs (Qwen2.5, Llama 3.2, Gemma 2 at ≤3B parameters).
  Checkpoints are released under the HuggingFace organization
  [legible-weights](https://huggingface.co/legible-weights).
- **Monthly writeups** that pick one concrete model behavior and trace it
  through the dictionary to the specific features (and weights) that cause it.
- **Tooling** for loading dictionaries, browsing features, and running
  attribution experiments.

## Status

Pre-v0.1. Repository scaffolding only — no trained dictionaries yet.

## License

MIT. See `LICENSE`.
