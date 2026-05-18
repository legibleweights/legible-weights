"""Compare two trained SAEs by aligning their dictionaries.

Loads SAE_A (e.g., GPT-2) and SAE_B (e.g., MDLM), computes cosine
similarity between all pairs of decoder rows, runs Hungarian assignment to
get one-to-one matches, then optionally runs both base models over a shared
text slice to also compute per-pair activation correlation.

Outputs:
- alignment.json (sorted matches with decoder_cos and act_corr)
- alignment_summary.json (bucketed counts + headline stats)
- A markdown report (paired top examples for the highest-cos and
  lowest-cos pairs, plus per-side exclusive top features) so a human can
  judge whether high-cos pairs really do represent the same concept.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))

from legible_weights.data.activations import collect_activations
from legible_weights.data.adapters import GPT2, MDLM, ModelAdapter
from legible_weights.eval.alignment import (
    FeatureMatch,
    activation_correlation_matrix,
    decoder_cosine_matrix,
    hungarian_match,
    summarize,
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


def _load_base_for_meta(meta: dict, device: str):
    name = meta["adapter"]
    if name == "gpt2":
        tok = AutoTokenizer.from_pretrained(meta["base_model"])
        tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            meta["base_model"], torch_dtype=torch.float16
        ).to(device).eval()
        return model, tok, GPT2
    elif name == "mdlm":
        from mdlm.configuration_mdlm import MDLMConfig
        from mdlm.modeling_mdlm import MDLM as MDLMModel

        cfg = MDLMConfig.from_pretrained(meta["base_model"])
        model = MDLMModel(cfg).to(device).eval()
        model.load_state_dict(
            load_file(hf_hub_download(meta["base_model"], "model.safetensors"))
        )
        tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
        tok.pad_token = tok.eos_token
        return model, tok, MDLM
    raise ValueError(f"Unknown adapter: {name}")


@torch.no_grad()
def _sae_acts_on_texts(
    sae: TopKSAE,
    base_model,
    tokenizer,
    texts: list[str],
    layer_idx: int,
    seq_len: int,
    n_tokens: int,
    device: str,
    adapter: ModelAdapter,
    exclude_first_n: int,
) -> torch.Tensor:
    """Collect SAE feature activations on a shared text slice (N tokens, d_hidden)."""
    raw = collect_activations(
        model=base_model, tokenizer=tokenizer, texts=iter(texts),
        layer_idx=layer_idx, n_tokens=n_tokens, seq_len=seq_len,
        batch_size=8, device=device, exclude_first_n=exclude_first_n,
        adapter=adapter, shuffle=False,  # preserve token order across A/B for correlation
    )
    sae.to(device).eval()
    out: list[torch.Tensor] = []
    bs = 4096
    for i in range(0, raw.shape[0], bs):
        x = raw[i:i + bs].to(device, dtype=torch.float32)
        acts = sae.encode(x)
        out.append(acts.cpu())
    return torch.cat(out, dim=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-a", type=Path, required=True,
                    help="Checkpoint dir of side A (e.g., GPT-2 SAE)")
    ap.add_argument("--sae-b", type=Path, required=True,
                    help="Checkpoint dir of side B (e.g., MDLM SAE)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--corr-n-tokens", type=int, default=20_000,
                    help="Tokens of shared text for activation correlation. Set to 0 to skip.")
    ap.add_argument("--corr-seq-len", type=int, default=256)
    ap.add_argument("--dataset-offset", type=int, default=30_000)
    ap.add_argument("--cos-threshold", type=float, default=0.5)
    ap.add_argument("--corr-threshold", type=float, default=0.3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[load] sae A: {args.sae_a}")
    sae_a, meta_a = _load_sae(args.sae_a)
    print(f"[load] sae B: {args.sae_b}")
    sae_b, meta_b = _load_sae(args.sae_b)

    if meta_a["sae"]["d_in"] != meta_b["sae"]["d_in"]:
        raise ValueError(
            f"d_in mismatch: A={meta_a['sae']['d_in']} B={meta_b['sae']['d_in']} — "
            "alignment requires both SAEs to operate in the same d_in space."
        )

    print("[acts] loading both base models for shared-text activation collection")
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    ds = ds.skip(args.dataset_offset)
    texts = [t["text"] for _, t in zip(range(2048), ds)]

    model_a, tok_a, adapter_a = _load_base_for_meta(meta_a, device)
    acts_a = _sae_acts_on_texts(
        sae_a, model_a, tok_a, texts,
        layer_idx=meta_a["layer"], seq_len=args.corr_seq_len,
        n_tokens=args.corr_n_tokens, device=device, adapter=adapter_a,
        exclude_first_n=meta_a["training"].get("exclude_first_n", 0),
    )
    del model_a
    torch.cuda.empty_cache()

    model_b, tok_b, adapter_b = _load_base_for_meta(meta_b, device)
    acts_b = _sae_acts_on_texts(
        sae_b, model_b, tok_b, texts,
        layer_idx=meta_b["layer"], seq_len=args.corr_seq_len,
        n_tokens=args.corr_n_tokens, device=device, adapter=adapter_b,
        exclude_first_n=meta_b["training"].get("exclude_first_n", 0),
    )
    del model_b
    torch.cuda.empty_cache()

    n_common = min(acts_a.shape[0], acts_b.shape[0])
    acts_a = acts_a[:n_common]
    acts_b = acts_b[:n_common]
    print(f"[acts] {n_common} matched tokens collected from both sides")

    print("[align] building full activation-correlation matrix between active features")
    corr_mat, active_a, active_b = activation_correlation_matrix(
        acts_a, acts_b, min_fires=30,
    )
    print(f"[align] active features: A={len(active_a)}/{acts_a.shape[1]}, "
          f"B={len(active_b)}/{acts_b.shape[1]}; corr matrix {tuple(corr_mat.shape)}; "
          f"max={corr_mat.max().item():.3f}, "
          f"row-max mean={corr_mat.max(dim=1).values.mean().item():.3f}")

    print("[align] hungarian on activation correlation (primary basis)")
    raw_matches = hungarian_match(corr_mat)
    # Remap reduced indices back to original feature ids, attach decoder cosine
    print("[align] computing decoder cosine (diagnostic only)")
    decoder_sim = decoder_cosine_matrix(sae_a.decoder.weight, sae_b.decoder.weight)
    matches: list[FeatureMatch] = []
    for m in raw_matches:
        orig_a = int(active_a[m.a].item())
        orig_b = int(active_b[m.b].item())
        matches.append(FeatureMatch(
            a=orig_a, b=orig_b,
            decoder_cos=float(decoder_sim[orig_a, orig_b].item()),
            act_corr=m.decoder_cos,  # hungarian_match stored the corr as decoder_cos field
        ))
    matches.sort(key=lambda m: -(m.act_corr or 0.0))
    print(f"[align] top match act_corr={matches[0].act_corr:.3f}, "
          f"median={matches[len(matches)//2].act_corr:.3f}")

    # Bucketing — using activation correlation only. Decoder cosine kept as diagnostic.
    n_shared = sum(1 for m in matches if (m.act_corr or 0.0) > args.corr_threshold)
    n_weak = sum(
        1 for m in matches
        if args.corr_threshold * 0.5 < (m.act_corr or 0.0) <= args.corr_threshold
    )
    n_divergent = len(matches) - n_shared - n_weak
    corrs_arr = sorted([m.act_corr for m in matches if m.act_corr is not None], reverse=True)
    summary = {
        "n_total_pairs": len(matches),
        "n_active_features_a": int(len(active_a)),
        "n_active_features_b": int(len(active_b)),
        "n_shared": n_shared,
        "n_weak": n_weak,
        "n_divergent": n_divergent,
        "max_act_corr": corrs_arr[0] if corrs_arr else None,
        "median_act_corr": corrs_arr[len(corrs_arr) // 2] if corrs_arr else None,
        "median_decoder_cos": float(
            sorted([m.decoder_cos for m in matches])[len(matches) // 2]
        ),
        "max_decoder_cos": float(max(m.decoder_cos for m in matches)),
        "thresholds": {"act_corr_shared": args.corr_threshold,
                       "act_corr_weak": args.corr_threshold * 0.5},
    }
    print(f"[summary] {summary}")

    (args.out / "alignment.json").write_text(json.dumps(
        [asdict(m) for m in matches], indent=2,
    ))
    (args.out / "alignment_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[save] wrote {args.out}/alignment.json and alignment_summary.json")


if __name__ == "__main__":
    main()
