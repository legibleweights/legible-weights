"""Cross-entropy recovery metric for residual-stream SAEs.

Measures how much of the base model's next-token predictive performance is
preserved when the residual stream at a target layer is replaced by the SAE
reconstruction. Reported as a fraction in [0, 1]:

    recovered = (ce_zero - ce_recon) / (ce_zero - ce_clean)

- ce_clean: base model loss, no intervention
- ce_recon: base model loss, target layer output replaced by SAE(x)
- ce_zero:  base model loss, target layer output replaced by the dataset mean
            activation (a no-information baseline)

A perfect SAE has recovered ≈ 1.0. A useless SAE has recovered ≈ 0.0.
Anything below ~0.5 on residual-stream activations is a sign the SAE is not
capturing the directions the next layer actually uses.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm import tqdm


@dataclass
class RecoveryResult:
    ce_clean: float
    ce_recon: float
    ce_zero: float
    recovered: float
    n_tokens: int


@torch.no_grad()
def _compute_ce(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[float, int]:
    """Per-token CE on next-token prediction, masked by attention_mask shifted."""
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :].contiguous()
    targets = input_ids[:, 1:].contiguous()
    mask = attention_mask[:, 1:].contiguous().bool()

    loss = F.cross_entropy(
        logits.float().view(-1, logits.size(-1)),
        targets.view(-1),
        reduction="none",
    ).view(targets.shape)
    valid = loss[mask]
    return valid.sum().item(), valid.numel()


def _install_replacement_hook(
    layer: torch.nn.Module,
    replacement_fn,
):
    def hook(_module, _inputs, outputs):
        if isinstance(outputs, tuple):
            hs = outputs[0]
            new_hs = replacement_fn(hs)
            return (new_hs,) + outputs[1:]
        return replacement_fn(outputs)

    return layer.register_forward_hook(hook)


@torch.no_grad()
def ce_recovery(
    model: torch.nn.Module,
    tokenizer,
    sae: torch.nn.Module,
    texts,
    layer_idx: int,
    n_batches: int = 16,
    batch_size: int = 4,
    seq_len: int = 512,
    device: str | torch.device = "cuda",
    mean_activation: torch.Tensor | None = None,
) -> RecoveryResult:
    """Compute CE recovery on a held-out text iterator.

    `mean_activation` (D,) is used for the zero-information baseline. If not
    given, it's estimated from the first batch.
    """
    layer = model.model.layers[layer_idx]

    # Materialize n_batches worth of inputs first (so all three runs see identical data)
    inputs: list[dict[str, torch.Tensor]] = []
    text_iter = iter(texts)
    pbar_collect = tqdm(total=n_batches, desc="collect eval batches", unit="batch")
    while len(inputs) < n_batches:
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(next(text_iter))
            except StopIteration:
                break
        if not batch:
            break
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seq_len,
        )
        inputs.append({k: v.to(device) for k, v in enc.items()})
        pbar_collect.update(1)
    pbar_collect.close()

    # Estimate mean activation if not provided — use sae.pre_bias as a proxy
    if mean_activation is None:
        mean_activation = sae.pre_bias.detach().to(device)
    mean_activation = mean_activation.to(device)

    def sae_replace(hs: torch.Tensor) -> torch.Tensor:
        flat = hs.reshape(-1, hs.size(-1)).to(torch.float32)
        recon, _ = sae(flat)
        return recon.reshape(hs.shape).to(hs.dtype)

    def mean_replace(hs: torch.Tensor) -> torch.Tensor:
        return mean_activation.to(hs.dtype).expand_as(hs)

    sums = {"clean": 0.0, "recon": 0.0, "zero": 0.0}
    counts = {"clean": 0, "recon": 0, "zero": 0}

    for inp in tqdm(inputs, desc="eval runs"):
        # Clean
        s, c = _compute_ce(model, inp["input_ids"], inp["attention_mask"])
        sums["clean"] += s
        counts["clean"] += c

        # Recon
        h = _install_replacement_hook(layer, sae_replace)
        try:
            s, c = _compute_ce(model, inp["input_ids"], inp["attention_mask"])
        finally:
            h.remove()
        sums["recon"] += s
        counts["recon"] += c

        # Zero (mean replacement)
        h = _install_replacement_hook(layer, mean_replace)
        try:
            s, c = _compute_ce(model, inp["input_ids"], inp["attention_mask"])
        finally:
            h.remove()
        sums["zero"] += s
        counts["zero"] += c

    ce_clean = sums["clean"] / counts["clean"]
    ce_recon = sums["recon"] / counts["recon"]
    ce_zero = sums["zero"] / counts["zero"]
    denom = ce_zero - ce_clean
    recovered = (ce_zero - ce_recon) / denom if denom > 1e-6 else float("nan")

    return RecoveryResult(
        ce_clean=ce_clean,
        ce_recon=ce_recon,
        ce_zero=ce_zero,
        recovered=recovered,
        n_tokens=counts["clean"],
    )
