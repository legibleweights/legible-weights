"""Held-out evaluation for a trained SAE.

Reports:
- reconstruction MSE and explained variance on held-out activations
- L0 (active features per token) and L0 stddev
- dead-feature count (features never active across the held-out set)
- CE recovery via splice intervention on the base model
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from datasets import load_dataset

from legible_weights.data.activations import collect_activations, load_model
from legible_weights.eval.recovery import ce_recovery
from legible_weights.sae.model import SAEConfig, TopKSAE


@dataclass
class EvalReport:
    n_eval_tokens: int
    mse: float
    explained_variance: float
    l0_mean: float
    l0_std: float
    dead_features: int
    total_features: int
    ce_clean: float
    ce_recon: float
    ce_zero: float
    ce_recovered: float
    ce_n_tokens: int


def reconstruction_metrics(sae: TopKSAE, activations: torch.Tensor, batch_size: int = 4096):
    sae.eval()
    device = next(sae.parameters()).device
    n = activations.shape[0]

    total_sq_err = 0.0
    total_count = 0
    feature_fire_count = torch.zeros(sae.cfg.d_hidden, device=device)
    l0_per_batch: list[torch.Tensor] = []
    sum_y = torch.zeros(sae.cfg.d_in, device=device, dtype=torch.float64)
    sum_y2 = torch.zeros(sae.cfg.d_in, device=device, dtype=torch.float64)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            x = activations[start : start + batch_size].to(device, dtype=torch.float32)
            recon, acts = sae(x)
            sq_err = (recon - x).pow(2).sum().item()
            total_sq_err += sq_err
            total_count += x.numel()
            feature_fire_count += (acts > 0).sum(dim=0).float()
            l0_per_batch.append((acts > 0).sum(dim=-1).float().cpu())
            sum_y += x.double().sum(dim=0)
            sum_y2 += x.double().pow(2).sum(dim=0)

    mean_y = (sum_y / n).to(torch.float32)
    var_y = (sum_y2 / n).to(torch.float32) - mean_y.pow(2)
    total_var = var_y.sum().item() * n  # because we summed across dims and over n
    mse = total_sq_err / total_count
    explained_var = 1.0 - (total_sq_err / total_var) if total_var > 0 else float("nan")

    l0_all = torch.cat(l0_per_batch)
    l0_mean = l0_all.mean().item()
    l0_std = l0_all.std().item()
    dead = int((feature_fire_count == 0).sum().item())

    return {
        "mse": mse,
        "explained_variance": explained_var,
        "l0_mean": l0_mean,
        "l0_std": l0_std,
        "dead_features": dead,
        "total_features": int(sae.cfg.d_hidden),
        "n_eval_tokens": int(n),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True, help="Directory with sae.pt + config.json")
    ap.add_argument("--n-eval-tokens", type=int, default=200_000)
    ap.add_argument("--n-ce-batches", type=int, default=16)
    ap.add_argument("--ce-batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--dataset-split-offset", type=int, default=10_000,
                    help="Skip this many rows of the streaming dataset to avoid overlap with training")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = json.loads((args.checkpoint / "config.json").read_text())
    exclude_first_n = int(cfg.get("training", {}).get("exclude_first_n", 0))
    print(f"[eval] checkpoint config: layer={cfg['layer']}, base={cfg['base_model']}, "
          f"exclude_first_n={exclude_first_n}")

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
    # Skip the prefix used during training for a true held-out evaluation
    ds = ds.skip(args.dataset_split_offset)
    # Pull enough held-out texts for both activation collection and CE recovery.
    # ~1000 texts at avg seq_len ~ a few hundred tokens easily covers 200k eval tokens
    # plus 16*4 = 64 texts for CE.
    n_texts_total = 1000 + args.n_ce_batches * args.ce_batch_size
    texts = [t["text"] for _, t in zip(range(n_texts_total), ds)]
    n_act = n_texts_total - args.n_ce_batches * args.ce_batch_size
    eval_text_iter_1 = iter(texts[:n_act])
    eval_text_iter_2 = iter(texts[n_act:])

    print("[eval] collecting held-out activations")
    activations = collect_activations(
        model=model,
        tokenizer=tok,
        texts=eval_text_iter_1,
        layer_idx=cfg["layer"],
        n_tokens=args.n_eval_tokens,
        seq_len=args.seq_len,
        batch_size=8,
        device=device,
        exclude_first_n=exclude_first_n,
    )
    print(f"[eval] held-out activations: {activations.shape}")

    rec_metrics = reconstruction_metrics(sae, activations)
    print(f"[eval] reconstruction: mse={rec_metrics['mse']:.4f} "
          f"ev={rec_metrics['explained_variance']:.3f} "
          f"l0={rec_metrics['l0_mean']:.1f}±{rec_metrics['l0_std']:.1f} "
          f"dead={rec_metrics['dead_features']}/{rec_metrics['total_features']}")

    print("[eval] CE recovery (splice intervention)")
    rec = ce_recovery(
        model=model,
        tokenizer=tok,
        sae=sae,
        texts=eval_text_iter_2,
        layer_idx=cfg["layer"],
        n_batches=args.n_ce_batches,
        batch_size=args.ce_batch_size,
        seq_len=args.seq_len,
        device=device,
        exclude_first_n=exclude_first_n,
    )
    print(f"[eval] CE: clean={rec.ce_clean:.3f} recon={rec.ce_recon:.3f} "
          f"zero={rec.ce_zero:.3f} recovered={rec.recovered:.3f}")

    report = EvalReport(
        n_eval_tokens=rec_metrics["n_eval_tokens"],
        mse=rec_metrics["mse"],
        explained_variance=rec_metrics["explained_variance"],
        l0_mean=rec_metrics["l0_mean"],
        l0_std=rec_metrics["l0_std"],
        dead_features=rec_metrics["dead_features"],
        total_features=rec_metrics["total_features"],
        ce_clean=rec.ce_clean,
        ce_recon=rec.ce_recon,
        ce_zero=rec.ce_zero,
        ce_recovered=rec.recovered,
        ce_n_tokens=rec.n_tokens,
    )
    (args.checkpoint / "eval.json").write_text(json.dumps(asdict(report), indent=2))
    print(f"[eval] wrote {args.checkpoint}/eval.json")


if __name__ == "__main__":
    main()
