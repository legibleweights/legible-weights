"""SAE training loop over a pre-collected activation buffer."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from tqdm import tqdm

from legible_weights.sae.model import TopKSAE


@dataclass
class TrainConfig:
    batch_size: int = 4096
    lr: float = 5e-4
    n_epochs: int = 4
    log_every: int = 25
    device: str = "cuda"


def train_sae(
    sae: TopKSAE,
    activations: torch.Tensor,
    cfg: TrainConfig,
) -> list[dict]:
    """Train `sae` on a (N, d_in) activation buffer. Returns step-level metrics."""
    sae.to(cfg.device).train()
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)

    n, _ = activations.shape
    history: list[dict] = []

    # Initialize pre_bias to the mean of activations — standard trick
    with torch.no_grad():
        sae.pre_bias.copy_(activations.mean(dim=0).to(cfg.device).float())

    total_var = activations.float().var(dim=0).sum().item()

    step = 0
    for epoch in range(cfg.n_epochs):
        perm = torch.randperm(n)
        pbar = tqdm(
            range(0, n - cfg.batch_size + 1, cfg.batch_size),
            desc=f"epoch {epoch + 1}/{cfg.n_epochs}",
        )
        for start in pbar:
            idx = perm[start : start + cfg.batch_size]
            x = activations[idx].to(cfg.device, dtype=torch.float32)

            recon, acts = sae(x)
            mse = (recon - x).pow(2).mean()

            opt.zero_grad(set_to_none=True)
            mse.backward()
            opt.step()
            sae.normalize_decoder_()

            if step % cfg.log_every == 0:
                with torch.no_grad():
                    residual_var = (x - recon).var(dim=0).sum().item()
                    explained = 1.0 - residual_var / total_var
                    l0 = (acts > 0).float().sum(dim=-1).mean().item()
                history.append(
                    {
                        "step": step,
                        "epoch": epoch,
                        "mse": mse.item(),
                        "explained_var": explained,
                        "l0": l0,
                    }
                )
                pbar.set_postfix(mse=f"{mse.item():.4f}", ev=f"{explained:.3f}", l0=f"{l0:.1f}")
            step += 1

    return history
