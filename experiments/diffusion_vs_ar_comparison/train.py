"""Train SAEs on GPT-2 small and MDLM at matched relative depth.

Same hyperparameters for both runs — only the model differs. The intent is
to make the resulting SAEs comparable so that subsequent dictionary
alignment can attribute differences to the training-objective axis, not to
SAE hyperparameter drift.

Run twice:
    python train.py --model gpt2 --out ../../checkpoints/sae-gpt2-small-l6-v0.1
    python train.py --model mdlm --out ../../checkpoints/sae-mdlm-owt-l6-v0.1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Local mdlm package (patched modeling file lives next to this script)
sys.path.insert(0, str(Path(__file__).parent))

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from legible_weights.data.activations import collect_activations
from legible_weights.data.adapters import GPT2, MDLM, QWEN_LLAMA, ModelAdapter
from legible_weights.sae.model import SAEConfig, TopKSAE
from legible_weights.sae.train import TrainConfig, train_sae


def load_gpt2_small(device: str):
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "openai-community/gpt2", torch_dtype=torch.float16
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok, GPT2, "openai-community/gpt2"


def load_mdlm_owt(device: str):
    from mdlm.configuration_mdlm import MDLMConfig
    from mdlm.modeling_mdlm import MDLM as MDLMModel

    cfg = MDLMConfig.from_pretrained("kuleshov-group/mdlm-owt")
    # MDLM has an internal autocast(bfloat16); load fp32 weights and let it
    # manage precision. Activations are downcast to fp16 inside the hook.
    model = MDLMModel(cfg).to(device).eval()
    model.load_state_dict(
        load_file(hf_hub_download("kuleshov-group/mdlm-owt", "model.safetensors"))
    )
    for p in model.parameters():
        p.requires_grad_(False)

    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tok.pad_token = tok.eos_token
    return model, tok, MDLM, "kuleshov-group/mdlm-owt"


MODELS = {
    "gpt2": load_gpt2_small,
    "mdlm": load_mdlm_owt,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--layer", type=int, default=6,
                    help="Layer index (both models have 12 blocks; layer 6 is mid-depth)")
    ap.add_argument("--n-tokens", type=int, default=10_000_000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--expansion", type=int, default=16)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--exclude-first-n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}, model={args.model}, layer={args.layer}, "
          f"n_tokens={args.n_tokens}")

    model, tok, adapter, model_id = MODELS[args.model](device)
    d_model = (model.config.hidden_size
               if hasattr(model.config, "hidden_size")
               else model.config.hidden_dim)
    print(f"[setup] d_model={d_model}, adapter={adapter.name}, model_id={model_id}")

    print("[data] streaming OpenWebText")
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    texts = (row["text"] for row in ds)

    t0 = time.time()
    activations = collect_activations(
        model=model, tokenizer=tok, texts=texts,
        layer_idx=args.layer, n_tokens=args.n_tokens, seq_len=args.seq_len,
        batch_size=8, device=device, exclude_first_n=args.exclude_first_n,
        adapter=adapter,
    )
    print(f"[data] collected {tuple(activations.shape)} in {time.time() - t0:.1f}s")

    del model
    torch.cuda.empty_cache()

    sae_cfg = SAEConfig(d_in=d_model, d_hidden=d_model * args.expansion, k=args.k)
    sae = TopKSAE(sae_cfg)
    print(f"[sae] d_in={sae_cfg.d_in} d_hidden={sae_cfg.d_hidden} k={sae_cfg.k}")

    train_cfg = TrainConfig(
        batch_size=args.batch_size, lr=args.lr,
        n_epochs=args.epochs, device=device,
    )

    t0 = time.time()
    history = train_sae(sae, activations, train_cfg)
    print(f"[train] {len(history)} log points in {time.time() - t0:.1f}s")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(sae.state_dict(), args.out / "sae.pt")
    (args.out / "config.json").write_text(json.dumps(
        {
            "base_model": model_id,
            "adapter": adapter.name,
            "layer": args.layer,
            "hook": f"model.{adapter.name}_layer[{args.layer}] output (residual stream)",
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
                "dataset": "Skylion007/openwebtext",
                "exclude_first_n": args.exclude_first_n,
                "diffusion_noise_level": 0 if adapter.name == "mdlm" else None,
            },
        }, indent=2))
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    if history:
        last = history[-1]
        print(f"[final] step={last['step']} mse={last['mse']:.4f} "
              f"ev={last['explained_var']:.3f} l0={last['l0']:.1f}")


if __name__ == "__main__":
    main()
