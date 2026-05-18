"""Find top-activating contexts for each SAE feature.

Runs a held-out text stream through the base model + SAE, captures feature
activations alongside the tokenized contexts, and for each feature reports
the top-k (token, surrounding context, activation strength) triples. This is
what lets a human look at a feature and say "feature 4912 is the
'opening-quote' feature, feature 117 is 'years in the 1990s'", etc.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from tqdm import tqdm


@dataclass
class FeatureExample:
    feature_id: int
    activation: float
    token_id: int
    token_str: str
    context_token_strs: list[str]
    context_highlight_pos: int  # index into context_token_strs of the activating token


@torch.no_grad()
def collect_feature_activations(
    model: torch.nn.Module,
    tokenizer,
    sae: torch.nn.Module,
    texts,
    layer_idx: int,
    n_sequences: int,
    seq_len: int = 256,
    batch_size: int = 8,
    device: str | torch.device = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (acts, token_ids, attn_mask).

    acts:      (n_total_tokens, d_hidden) — SAE activations per token (sparse)
    token_ids: (n_sequences, seq_len)
    attn_mask: (n_sequences, seq_len)
    """
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, outputs):
        hs = outputs[0] if isinstance(outputs, tuple) else outputs
        captured.append(hs.detach())

    handle = model.model.layers[layer_idx].register_forward_hook(hook)

    all_acts: list[torch.Tensor] = []
    all_ids: list[torch.Tensor] = []
    all_mask: list[torch.Tensor] = []

    try:
        batch: list[str] = []
        text_iter = iter(texts)
        pbar = tqdm(total=n_sequences, desc="collect features", unit="seq")
        seqs_done = 0
        while seqs_done < n_sequences:
            try:
                batch.append(next(text_iter))
            except StopIteration:
                break
            if len(batch) < batch_size:
                continue

            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=seq_len,
            ).to(device)
            captured.clear()
            model(**enc)
            hs = captured[0]  # (B, L, D)

            flat = hs.reshape(-1, hs.size(-1)).to(torch.float32)
            acts = sae.encode(flat)  # (B*L, d_hidden), sparse via TopK
            acts = acts.reshape(hs.size(0), hs.size(1), -1)

            all_acts.append(acts.cpu())
            all_ids.append(enc.input_ids.cpu())
            all_mask.append(enc.attention_mask.cpu())
            seqs_done += hs.size(0)
            pbar.update(hs.size(0))
            batch = []
        pbar.close()
    finally:
        handle.remove()

    acts = torch.cat(all_acts, dim=0)  # (S, L, d_hidden)
    ids = torch.cat(all_ids, dim=0)
    mask = torch.cat(all_mask, dim=0)
    return acts, ids, mask


def top_activating_examples(
    acts: torch.Tensor,            # (S, L, d_hidden)
    token_ids: torch.Tensor,       # (S, L)
    attn_mask: torch.Tensor,       # (S, L)
    tokenizer,
    feature_ids: list[int],
    top_k: int = 5,
    context_radius: int = 8,
) -> dict[int, list[FeatureExample]]:
    """For each feature in `feature_ids`, return its top-`top_k` activating contexts."""
    s, l, d = acts.shape
    out: dict[int, list[FeatureExample]] = {}

    # Mask out padding positions before ranking
    pad_mask_flat = attn_mask.reshape(-1).bool()
    flat_acts_all = acts.reshape(-1, d)
    flat_ids_all = token_ids.reshape(-1)

    for fid in feature_ids:
        col = flat_acts_all[:, fid].clone()
        col[~pad_mask_flat] = float("-inf")
        if (col == float("-inf")).all():
            out[fid] = []
            continue

        # Pick top-k unique flat positions
        topk = min(top_k, int((col != float("-inf")).sum().item()))
        if topk == 0:
            out[fid] = []
            continue
        vals, idxs = col.topk(topk)

        examples: list[FeatureExample] = []
        for v, flat_idx in zip(vals.tolist(), idxs.tolist()):
            if v <= 0:
                continue
            seq_i = flat_idx // l
            tok_i = flat_idx % l
            lo = max(0, tok_i - context_radius)
            hi = min(l, tok_i + context_radius + 1)
            context_ids = token_ids[seq_i, lo:hi].tolist()
            context_strs = [tokenizer.decode([t]) for t in context_ids]
            tok_id = int(token_ids[seq_i, tok_i].item())
            examples.append(
                FeatureExample(
                    feature_id=fid,
                    activation=float(v),
                    token_id=tok_id,
                    token_str=tokenizer.decode([tok_id]),
                    context_token_strs=context_strs,
                    context_highlight_pos=tok_i - lo,
                )
            )
        out[fid] = examples
    return out


def format_examples_markdown(
    examples_by_feature: dict[int, list[FeatureExample]],
) -> str:
    """Render a markdown report. Each feature gets a section with its top contexts."""
    lines: list[str] = []
    for fid, examples in examples_by_feature.items():
        if not examples:
            lines.append(f"### feature {fid}\n\n_no positive activations on the eval set_\n")
            continue
        lines.append(f"### feature {fid}")
        lines.append(f"top activation: **{examples[0].activation:.2f}**, "
                     f"token: `{examples[0].token_str!r}`\n")
        lines.append("| rank | act | context |")
        lines.append("|------|-----|---------|")
        for rank, ex in enumerate(examples, 1):
            ctx = ""
            for i, tok in enumerate(ex.context_token_strs):
                # Escape pipes and newlines for table cell
                tok_clean = tok.replace("|", "\\|").replace("\n", "↵")
                if i == ex.context_highlight_pos:
                    ctx += f"**[{tok_clean}]**"
                else:
                    ctx += tok_clean
            lines.append(f"| {rank} | {ex.activation:.2f} | {ctx} |")
        lines.append("")
    return "\n".join(lines)
