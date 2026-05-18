"""Train a TopK SAE on Qwen2.5-0.5B layer 9 residual stream.

Smoke-scale run: ~1M tokens of activations, ~3 minutes on a single 4090.
The full run swaps n_tokens and n_epochs for larger values; everything else
stays the same.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset

from legible_weights.data.activations import collect_activations, load_model
from legible_weights.sae.model import SAEConfig, TopKSAE
from legible_weights.sae.train import TrainConfig, train_sae


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--layer", type=int, default=9)
    ap.add_argument("--n-tokens", type=int, default=1_000_000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--expansion", type=int, default=16)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("checkpoints/sae-qwen2.5-0.5b-l9"),
    )
    ap.add_argument("--smoke", action="store_true", help="Use a tiny dataset slice")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}, model={args.model}, layer={args.layer}")

    model, tok = load_model(args.model, device=device, dtype=torch.float16)
    d_model = model.config.hidden_size
    print(f"[setup] d_model={d_model}")

    print("[data] streaming FineWeb-Edu sample-10BT")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    texts = (row["text"] for row in ds)

    t0 = time.time()
    activations = collect_activations(
        model=model,
        tokenizer=tok,
        texts=texts,
        layer_idx=args.layer,
        n_tokens=args.n_tokens,
        seq_len=args.seq_len,
        batch_size=8,
        device=device,
    )
    print(f"[data] collected {activations.shape} in {time.time() - t0:.1f}s")

    # Free model weights — SAE training only needs activations
    del model
    torch.cuda.empty_cache()

    sae_cfg = SAEConfig(d_in=d_model, d_hidden=d_model * args.expansion, k=args.k)
    sae = TopKSAE(sae_cfg)
    print(f"[sae] d_in={sae_cfg.d_in}, d_hidden={sae_cfg.d_hidden}, k={sae_cfg.k}")

    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        n_epochs=args.epochs,
        device=device,
    )

    t0 = time.time()
    history = train_sae(sae, activations, train_cfg)
    print(f"[train] {len(history)} log points in {time.time() - t0:.1f}s")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(sae.state_dict(), args.out / "sae.pt")
    (args.out / "config.json").write_text(
        json.dumps(
            {
                "base_model": args.model,
                "layer": args.layer,
                "hook": f"model.layers[{args.layer}] output[0] (residual stream)",
                "sae": {
                    "d_in": sae_cfg.d_in,
                    "d_hidden": sae_cfg.d_hidden,
                    "k": sae_cfg.k,
                    "arch": "TopK",
                },
                "training": {
                    "n_tokens": args.n_tokens,
                    "seq_len": args.seq_len,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "n_epochs": args.epochs,
                    "seed": args.seed,
                    "dataset": "HuggingFaceFW/fineweb-edu:sample-10BT",
                },
            },
            indent=2,
        )
    )
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[save] wrote {args.out}/sae.pt + config.json + history.json")

    if history:
        last = history[-1]
        print(
            f"[final] step={last['step']} mse={last['mse']:.4f} "
            f"ev={last['explained_var']:.3f} l0={last['l0']:.1f}"
        )


if __name__ == "__main__":
    main()
