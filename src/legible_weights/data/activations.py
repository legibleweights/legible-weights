"""Collect residual-stream activations from a HuggingFace causal LM.

The model is run in eval mode with no grad; activations are captured by a
forward hook on the target decoder layer and accumulated into a buffer of
shape (n_tokens, d_model). The buffer is shuffled once before being returned
so downstream training sees IID batches.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(
    model_name: str,
    device: str | torch.device,
    dtype: torch.dtype = torch.float16,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def _get_decoder_layer(model: torch.nn.Module, idx: int) -> torch.nn.Module:
    # Works for Qwen2, Llama, Gemma — all use model.model.layers[i]
    return model.model.layers[idx]


@torch.no_grad()
def collect_activations(
    model: torch.nn.Module,
    tokenizer,
    texts: Iterable[str],
    layer_idx: int,
    n_tokens: int,
    seq_len: int = 512,
    batch_size: int = 8,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Stream `texts` through the model, capturing layer_idx residual stream.

    Returns a tensor of shape (n_tokens, d_model), fp16, on CPU. Caller is
    responsible for moving batches to GPU during SAE training.
    """
    d_model = model.config.hidden_size
    buf = torch.empty((n_tokens, d_model), dtype=torch.float16)
    filled = 0

    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, outputs):
        # Decoder layer output is a tuple; element 0 is the hidden states
        hs = outputs[0] if isinstance(outputs, tuple) else outputs
        captured.append(hs.detach().to(torch.float16).cpu())

    layer = _get_decoder_layer(model, layer_idx)
    handle = layer.register_forward_hook(hook)

    try:
        batch: list[str] = []
        pbar = tqdm(total=n_tokens, desc=f"collect L{layer_idx}", unit="tok")
        for text in texts:
            if filled >= n_tokens:
                break
            batch.append(text)
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
            mask = enc.attention_mask.cpu().bool()
            valid = hs[mask]  # (n_valid_tokens, D)
            take = min(valid.shape[0], n_tokens - filled)
            buf[filled : filled + take] = valid[:take]
            filled += take
            pbar.update(take)
            batch = []
        pbar.close()
    finally:
        handle.remove()

    buf = buf[:filled]
    # Shuffle once
    perm = torch.randperm(buf.shape[0])
    return buf[perm]
