"""SAE feature stability under quantization.

Setup: take a single base model (Qwen2.5-0.5B) and a single fp16-trained SAE
checkpoint (the v0.2 layer-9 dictionary). Run the *same* held-out tokens
through the base model at multiple precisions (fp16, int8, nf4), capture
layer-9 residual activations from each, encode all of them through the
*same* fp16 SAE, and quantify how stable the feature activations are.

Outputs (per-precision):
- Reconstruction MSE / explained variance via the fp16 SAE
- Per-feature Pearson correlation against the fp16 reference activations
- Dead-feature emergence (features active on fp16 → silent on quantized)
- CE recovery via splice intervention on the quantized model

Saves a single combined JSON report plus a numpy array of per-feature
correlations so downstream feature inspection can target specific
"survived" / "died" / "drifted" features.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from legible_weights.data.activations import collect_activations
from legible_weights.data.adapters import QWEN_LLAMA
from legible_weights.eval.recovery import ce_recovery
from legible_weights.sae.model import SAEConfig, TopKSAE


PRECISIONS = ["fp16", "int8", "nf4"]


def load_base(precision: str, device: str):
    name = "Qwen/Qwen2.5-0.5B"
    if precision == "fp16":
        m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16)
        m.to(device).eval()
    elif precision == "int8":
        cfg = BitsAndBytesConfig(load_in_8bit=True)
        m = AutoModelForCausalLM.from_pretrained(name, quantization_config=cfg).eval()
    elif precision == "nf4":
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        m = AutoModelForCausalLM.from_pretrained(name, quantization_config=cfg).eval()
    else:
        raise ValueError(f"unknown precision: {precision}")
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def reconstruction_metrics(sae, x):
    """MSE and explained variance of sae(x) vs x."""
    with torch.no_grad():
        recon, acts = sae(x)
        mse = (recon - x).pow(2).mean().item()
        var = x.var(dim=0).sum().item()
        residual_var = (x - recon).var(dim=0).sum().item()
        ev = 1.0 - residual_var / var if var > 0 else float("nan")
        l0_mean = (acts > 0).float().sum(dim=-1).mean().item()
        return mse, ev, l0_mean, acts


def encode_in_chunks(sae, activations, chunk=4096):
    out = []
    device = next(sae.parameters()).device
    with torch.no_grad():
        for i in range(0, activations.shape[0], chunk):
            x = activations[i : i + chunk].to(device, dtype=torch.float32)
            out.append(sae.encode(x).cpu())
    return torch.cat(out, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", type=Path, required=True,
                    help="fp16 SAE checkpoint directory")
    ap.add_argument("--n-tokens", type=int, default=50_000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--n-ce-batches", type=int, default=8)
    ap.add_argument("--ce-batch-size", type=int, default=4)
    ap.add_argument("--dataset-offset", type=int, default=50_000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--precisions", nargs="+", default=PRECISIONS)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- load SAE ----
    meta = json.loads((args.sae / "config.json").read_text())
    sae_cfg = SAEConfig(
        d_in=meta["sae"]["d_in"],
        d_hidden=meta["sae"]["d_hidden"],
        k=meta["sae"]["k"],
    )
    sae = TopKSAE(sae_cfg)
    sae.load_state_dict(torch.load(args.sae / "sae.pt", map_location="cpu", weights_only=True))
    sae.to(device).eval()
    layer_idx = meta["layer"]
    exclude_first_n = int(meta["training"].get("exclude_first_n", 0))
    print(f"[setup] SAE layer={layer_idx} d_hidden={sae_cfg.d_hidden} k={sae_cfg.k} "
          f"exclude_first_n={exclude_first_n}")

    # ---- shared text slice ----
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                       split="train", streaming=True)
    ds = ds.skip(args.dataset_offset)
    texts = [t["text"] for _, t in zip(range(2048), ds)]
    print(f"[setup] using {len(texts)} held-out FineWeb-Edu sequences")

    results: dict = {
        "sae_checkpoint": str(args.sae),
        "n_tokens": args.n_tokens,
        "seq_len": args.seq_len,
        "precisions": {},
    }

    all_features_per_precision: dict = {}

    for precision in args.precisions:
        print(f"\n=== precision: {precision} ===")
        t0 = time.time()
        model = load_base(precision, device)
        # NOTE: device for nf4/int8 is set internally by bnb (CUDA-only)

        activations = collect_activations(
            model=model, tokenizer=tok, texts=iter(texts),
            layer_idx=layer_idx, n_tokens=args.n_tokens, seq_len=args.seq_len,
            batch_size=8, device=device, exclude_first_n=exclude_first_n,
            adapter=QWEN_LLAMA, shuffle=False,
        )
        t_collect = time.time() - t0
        print(f"[{precision}] activations: {tuple(activations.shape)} in {t_collect:.1f}s")

        # Reconstruction & feature acts on the same buffer
        sample = activations.to(device, dtype=torch.float32)
        with torch.no_grad():
            mse_full, ev_full, l0_full, _ = reconstruction_metrics(sae, sample)
        feats = encode_in_chunks(sae, activations)  # (N, d_hidden)
        all_features_per_precision[precision] = feats
        n_dead = int((feats.sum(dim=0) == 0).sum().item())
        print(f"[{precision}] MSE={mse_full:.4f}  EV={ev_full:.3f}  "
              f"L0={l0_full:.1f}  dead_features={n_dead}")

        # CE recovery via splice intervention
        ds_ce = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                             split="train", streaming=True)
        ds_ce = ds_ce.skip(args.dataset_offset + 5000)  # disjoint slice for CE
        ce_texts = (t["text"] for _, t in zip(range(args.n_ce_batches * args.ce_batch_size + 32), ds_ce))
        rec = ce_recovery(
            model=model, tokenizer=tok, sae=sae, texts=ce_texts,
            layer_idx=layer_idx, n_batches=args.n_ce_batches,
            batch_size=args.ce_batch_size, seq_len=args.seq_len,
            device=device, exclude_first_n=exclude_first_n,
        )
        print(f"[{precision}] CE clean={rec.ce_clean:.3f} recon={rec.ce_recon:.3f} "
              f"zero={rec.ce_zero:.3f} recovered={rec.recovered:.3f}")

        results["precisions"][precision] = {
            "mse": mse_full,
            "ev": ev_full,
            "l0_mean": l0_full,
            "dead_features": n_dead,
            "ce_clean": rec.ce_clean,
            "ce_recon": rec.ce_recon,
            "ce_zero": rec.ce_zero,
            "ce_recovered": rec.recovered,
            "ce_n_tokens": rec.n_tokens,
            "collect_seconds": t_collect,
        }

        del model
        torch.cuda.empty_cache()

    # ---- per-feature correlations against fp16 reference ----
    if "fp16" in all_features_per_precision:
        ref = all_features_per_precision["fp16"].float()
        n = ref.shape[0]
        ref_centered = ref - ref.mean(dim=0, keepdim=True)
        ref_std = ref.std(dim=0).clamp_min(1e-8)
        ref_norm = ref_centered / ref_std

        per_feat_results: dict = {}
        for precision, feats in all_features_per_precision.items():
            if precision == "fp16":
                continue
            other = feats.float()
            other_centered = other - other.mean(dim=0, keepdim=True)
            other_std = other.std(dim=0).clamp_min(1e-8)
            other_norm = other_centered / other_std
            corr = (ref_norm * other_norm).mean(dim=0)  # (d_hidden,)
            # Track which features were "active" at fp16 (fired on >=30 tokens)
            active = ((ref > 0).sum(dim=0) >= 30)
            corr_active = corr[active]

            # Buckets
            stable = int(((corr_active > 0.9) | ((feats > 0).sum(dim=0)[active] >= 30) & (corr_active > 0.9)).sum().item())
            stable = int((corr_active > 0.9).sum().item())
            drifted = int(((corr_active > 0.5) & (corr_active <= 0.9)).sum().item())
            died = int((corr_active <= 0.5).sum().item())
            per_feat_results[precision] = {
                "n_active_at_fp16": int(active.sum().item()),
                "n_stable_corr_gt_0.9": stable,
                "n_drifted_corr_0.5_0.9": drifted,
                "n_died_corr_le_0.5": died,
                "median_corr_active": float(corr_active.median().item()),
                "mean_corr_active": float(corr_active.mean().item()),
            }
            print(f"\n[per-feat vs fp16] {precision}: "
                  f"active@fp16={int(active.sum())}  "
                  f"stable (r>0.9)={stable}  "
                  f"drifted (0.5<r<=0.9)={drifted}  "
                  f"died (r<=0.5)={died}  "
                  f"median r={corr_active.median().item():.3f}")

            np.save(args.out / f"per_feature_corr_{precision}.npy", corr.numpy())
            np.save(args.out / f"feature_active_mask_{precision}.npy", active.numpy())

        results["per_feature_vs_fp16"] = per_feat_results

    (args.out / "report.json").write_text(json.dumps(results, indent=2))
    print(f"\n[save] wrote {args.out}/report.json")


if __name__ == "__main__":
    main()
