"""Inspect activation-matched feature pairs across GPT-2 and MDLM.

For the top-N most correlated pairs from alignment.json (and optionally for
a selection of divergent pairs), runs both base models over a shared text
slice and prints the top-activating contexts for the GPT-2 feature and the
MDLM feature side by side. If the pair's act_corr is high, the contexts
should overlap.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))

from legible_weights.data.adapters import GPT2, MDLM
from legible_weights.eval.features import (
    collect_feature_activations,
    top_activating_examples,
)
from legible_weights.sae.model import SAEConfig, TopKSAE


def _load_sae(checkpoint_dir: Path) -> tuple[TopKSAE, dict]:
    cfg_meta = json.loads((checkpoint_dir / "config.json").read_text())
    sae_cfg = SAEConfig(
        d_in=cfg_meta["sae"]["d_in"],
        d_hidden=cfg_meta["sae"]["d_hidden"],
        k=cfg_meta["sae"]["k"],
    )
    sae = TopKSAE(sae_cfg)
    sae.load_state_dict(
        torch.load(checkpoint_dir / "sae.pt", map_location="cpu", weights_only=True)
    )
    sae.eval()
    return sae, cfg_meta


def _load_base(meta: dict, device: str):
    name = meta["adapter"]
    if name == "gpt2":
        tok = AutoTokenizer.from_pretrained(meta["base_model"])
        tok.pad_token = tok.eos_token
        m = AutoModelForCausalLM.from_pretrained(
            meta["base_model"], torch_dtype=torch.float16
        ).to(device).eval()
        return m, tok, GPT2
    elif name == "mdlm":
        from mdlm.configuration_mdlm import MDLMConfig
        from mdlm.modeling_mdlm import MDLM as MDLMModel

        cfg = MDLMConfig.from_pretrained(meta["base_model"])
        m = MDLMModel(cfg).to(device).eval()
        m.load_state_dict(
            load_file(hf_hub_download(meta["base_model"], "model.safetensors"))
        )
        tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
        tok.pad_token = tok.eos_token
        return m, tok, MDLM
    raise ValueError(f"Unknown adapter: {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alignment", type=Path, required=True,
                    help="Path to alignment.json")
    ap.add_argument("--sae-a", type=Path, required=True)
    ap.add_argument("--sae-b", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-top", type=int, default=15,
                    help="Top-N pairs by act_corr to inspect")
    ap.add_argument("--n-divergent", type=int, default=5,
                    help="Lowest-corr pairs to inspect for contrast")
    ap.add_argument("--n-sequences", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--context-radius", type=int, default=8)
    ap.add_argument("--dataset-offset", type=int, default=40_000,
                    help="Skip past the alignment-correlation slice (which used offset 30000)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    matches = json.loads(args.alignment.read_text())
    matches = [m for m in matches if m.get("act_corr") is not None]
    matches.sort(key=lambda m: -m["act_corr"])
    top_pairs = matches[: args.n_top]
    divergent_pairs = sorted(matches, key=lambda m: m["act_corr"])[: args.n_divergent]
    pairs_to_inspect = top_pairs + divergent_pairs
    feature_ids_a = [m["a"] for m in pairs_to_inspect]
    feature_ids_b = [m["b"] for m in pairs_to_inspect]

    print(f"[load] alignment: {len(matches)} matches in file")
    print(f"[load] inspecting top {len(top_pairs)} and bottom {len(divergent_pairs)}")

    sae_a, meta_a = _load_sae(args.sae_a)
    sae_b, meta_b = _load_sae(args.sae_b)
    sae_a.to(device).eval()
    sae_b.to(device).eval()

    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    ds = ds.skip(args.dataset_offset)
    texts = [t["text"] for _, t in zip(range(args.n_sequences + 64), ds)]

    print("[A] collecting feature activations")
    model_a, tok_a, adapter_a = _load_base(meta_a, device)
    acts_a, ids_a, mask_a = collect_feature_activations(
        model=model_a, tokenizer=tok_a, sae=sae_a, texts=iter(texts),
        layer_idx=meta_a["layer"], n_sequences=args.n_sequences,
        seq_len=args.seq_len, batch_size=8, device=device, adapter=adapter_a,
    )
    if meta_a["training"].get("exclude_first_n", 0) > 0:
        mask_a[:, : meta_a["training"]["exclude_first_n"]] = 0
    del model_a
    torch.cuda.empty_cache()
    ex_a = top_activating_examples(
        acts=acts_a, token_ids=ids_a, attn_mask=mask_a, tokenizer=tok_a,
        feature_ids=feature_ids_a, top_k=args.top_k, context_radius=args.context_radius,
    )

    print("[B] collecting feature activations")
    model_b, tok_b, adapter_b = _load_base(meta_b, device)
    acts_b, ids_b, mask_b = collect_feature_activations(
        model=model_b, tokenizer=tok_b, sae=sae_b, texts=iter(texts),
        layer_idx=meta_b["layer"], n_sequences=args.n_sequences,
        seq_len=args.seq_len, batch_size=8, device=device, adapter=adapter_b,
    )
    if meta_b["training"].get("exclude_first_n", 0) > 0:
        mask_b[:, : meta_b["training"]["exclude_first_n"]] = 0
    del model_b
    torch.cuda.empty_cache()
    ex_b = top_activating_examples(
        acts=acts_b, token_ids=ids_b, attn_mask=mask_b, tokenizer=tok_b,
        feature_ids=feature_ids_b, top_k=args.top_k, context_radius=args.context_radius,
    )

    # Render markdown
    lines: list[str] = []
    lines.append("# GPT-2 vs MDLM — paired feature inspection")
    lines.append("")
    lines.append(
        f"Eval set: {args.n_sequences} sequences × {args.seq_len} tokens from "
        f"OpenWebText (offset {args.dataset_offset}, disjoint from training and "
        f"correlation slices)."
    )
    lines.append("")
    lines.append(
        f"**A = GPT-2 small layer {meta_a['layer']}**, "
        f"**B = MDLM-OWT layer {meta_b['layer']}**. "
        f"Same tokenizer for both (GPT-2 BPE), so token strings are directly comparable."
    )
    lines.append("")

    def render_pair(m, label):
        a, b = m["a"], m["b"]
        lines.append(f"### {label} — A feat {a}  ↔  B feat {b}")
        lines.append(f"act_corr = **{m['act_corr']:.3f}**, decoder_cos = {m['decoder_cos']:+.3f}")
        lines.append("")
        a_examples = ex_a.get(a, [])
        b_examples = ex_b.get(b, [])
        if not a_examples and not b_examples:
            lines.append("_neither side has positive activations on this slice_\n")
            return
        lines.append("| rank | GPT-2 act | GPT-2 context | MDLM act | MDLM context |")
        lines.append("|------|-----------|---------------|----------|--------------|")
        for r in range(max(len(a_examples), len(b_examples))):
            def fmt(exs, r):
                if r >= len(exs):
                    return "—", ""
                e = exs[r]
                ctx = ""
                for i, tok in enumerate(e.context_token_strs):
                    tok_clean = tok.replace("|", "\\|").replace("\n", "↵")
                    if i == e.context_highlight_pos:
                        ctx += f"**[{tok_clean}]**"
                    else:
                        ctx += tok_clean
                return f"{e.activation:.2f}", ctx
            a_act, a_ctx = fmt(a_examples, r)
            b_act, b_ctx = fmt(b_examples, r)
            lines.append(f"| {r + 1} | {a_act} | {a_ctx} | {b_act} | {b_ctx} |")
        lines.append("")

    lines.append("## Top correlated pairs")
    lines.append("")
    for i, m in enumerate(top_pairs, 1):
        render_pair(m, f"#{i}")

    lines.append("## Divergent pairs (lowest correlation in the matching)")
    lines.append("")
    for i, m in enumerate(divergent_pairs, 1):
        render_pair(m, f"#{len(top_pairs) + i}")

    args.out.write_text("\n".join(lines))
    print(f"[save] wrote {args.out}")


if __name__ == "__main__":
    main()
