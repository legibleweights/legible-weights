"""Inspect features in the v0.1-smoke SAE.

Runs a small eval set through Qwen2.5-0.5B + the layer-9 SAE, then for the
top-N most active features prints/saves their top activating contexts. The
output markdown lands in features_v0.1.md next to NOTES.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset

from legible_weights.data.activations import load_model
from legible_weights.eval.features import (
    collect_feature_activations,
    format_examples_markdown,
    top_activating_examples,
)
from legible_weights.sae.model import SAEConfig, TopKSAE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("checkpoints/sae-qwen2.5-0.5b-l9"))
    ap.add_argument("--n-sequences", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--top-features", type=int, default=20,
                    help="How many features (by total activation mass) to inspect")
    ap.add_argument("--top-k", type=int, default=5,
                    help="Examples per feature")
    ap.add_argument("--context-radius", type=int, default=8)
    ap.add_argument("--dataset-split-offset", type=int, default=20_000,
                    help="Skip even further into the dataset than the eval set")
    ap.add_argument("--exclude-first-n", type=int, default=0,
                    help="Mask out the first N token positions in each sequence "
                         "(outlier-position workaround for residual-stream SAEs)")
    ap.add_argument("--out",
                    type=Path,
                    default=Path("experiments/qwen2_5_0_5b/features_v0.1.md"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.loads((args.checkpoint / "config.json").read_text())

    sae_cfg = SAEConfig(
        d_in=cfg["sae"]["d_in"],
        d_hidden=cfg["sae"]["d_hidden"],
        k=cfg["sae"]["k"],
    )
    sae = TopKSAE(sae_cfg)
    sae.load_state_dict(torch.load(args.checkpoint / "sae.pt", map_location="cpu", weights_only=True))
    sae.to(device).eval()

    model, tok = load_model(cfg["base_model"], device=device, dtype=torch.float16)

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    ds = ds.skip(args.dataset_split_offset)
    texts = [t["text"] for _, t in zip(range(args.n_sequences + 32), ds)]

    acts, ids, mask = collect_feature_activations(
        model=model,
        tokenizer=tok,
        sae=sae,
        texts=iter(texts),
        layer_idx=cfg["layer"],
        n_sequences=args.n_sequences,
        seq_len=args.seq_len,
        batch_size=8,
        device=device,
    )
    print(f"[features] acts shape: {tuple(acts.shape)}")

    if args.exclude_first_n > 0:
        mask[:, : args.exclude_first_n] = 0
        print(f"[features] masked out first {args.exclude_first_n} positions of each sequence")

    # Rank features by total mass (sum of activations) across the eval set
    mask_flat = mask.reshape(-1).bool()
    acts_flat = acts.reshape(-1, acts.size(-1))
    total_mass = acts_flat[mask_flat].sum(dim=0)
    fire_count = (acts_flat[mask_flat] > 0).sum(dim=0)

    # Pick features that fire on at least a few tokens but aren't trivially-everywhere
    # to bias toward interpretable ones. Sort by total mass among qualifying features.
    n_tokens = int(mask_flat.sum().item())
    min_fires = max(3, n_tokens // 5000)        # ~0.02% of tokens
    max_fires = max(min_fires + 1, n_tokens // 4)  # not in >25% of tokens
    qualifying = (fire_count >= min_fires) & (fire_count <= max_fires)
    masked_mass = torch.where(qualifying, total_mass, torch.full_like(total_mass, -1.0))
    top_vals, top_ids = masked_mass.topk(args.top_features)
    feature_ids = [int(i) for i in top_ids.tolist()]

    print(f"[features] n_tokens={n_tokens}, min_fires={min_fires}, "
          f"max_fires={max_fires}, top features: {feature_ids[:5]}...")

    examples = top_activating_examples(
        acts=acts,
        token_ids=ids,
        attn_mask=mask,
        tokenizer=tok,
        feature_ids=feature_ids,
        top_k=args.top_k,
        context_radius=args.context_radius,
    )

    header = (
        "# v0.1-smoke feature inspection\n\n"
        f"Eval set: {args.n_sequences} sequences × {args.seq_len} tokens from a held-out "
        f"FineWeb-Edu slice (offset {args.dataset_split_offset}+ rows).\n\n"
        f"Total positions: {acts.shape[0] * acts.shape[1]} "
        f"({n_tokens} non-pad).\n\n"
        f"Features ranked by total activation mass among features firing on "
        f"[{min_fires}, {max_fires}] tokens "
        f"(filters out always-on directions and one-shot noise).\n\n"
        f"---\n\n"
    )
    body = format_examples_markdown(examples)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(header + body)
    print(f"[features] wrote {args.out}")

    # Also save a small JSON summary for later programmatic use
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "feature_ids": feature_ids,
                "n_tokens": n_tokens,
                "min_fires": int(min_fires),
                "max_fires": int(max_fires),
                "fire_counts": [int(fire_count[i].item()) for i in feature_ids],
                "total_masses": [float(total_mass[i].item()) for i in feature_ids],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
