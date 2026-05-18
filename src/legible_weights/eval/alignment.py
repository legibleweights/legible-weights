"""Align two SAE dictionaries trained on different models.

Cross-paradigm feature comparison: given SAE_A and SAE_B trained on the same
(or comparable) activation distribution from two different models, ask which
features in A have an analogue in B. The pipeline:

1. **Cosine-similarity matching** of decoder rows. Each feature in an SAE is
   a direction in d_model space; if two features encode the same concept,
   their decoder directions should be highly aligned (after sign + norm
   normalization).
2. **Hungarian assignment** to pick one-to-one matches that maximize total
   similarity.
3. **Activation correlation** on a shared text corpus: pass the same texts
   through both base models, get both SAEs' activations, and measure
   correlation between hypothesized matched feature pairs. A feature pair
   that has high decoder cosine *and* high activation correlation is a
   strong match.

The output is a list of (feature_id_A, feature_id_B, decoder_cos,
act_corr) triples plus the unmatched features on each side.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


@dataclass
class FeatureMatch:
    a: int
    b: int
    decoder_cos: float
    act_corr: float | None = None


def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def decoder_cosine_matrix(
    decoder_a: torch.Tensor,  # (d_in, d_hidden_a)
    decoder_b: torch.Tensor,  # (d_in, d_hidden_b)
) -> torch.Tensor:
    """(d_hidden_a, d_hidden_b) matrix of cosine similarities between decoder columns.

    For the comparison to be valid, decoder_a and decoder_b must operate in
    *aligned* d_in spaces — i.e., the underlying models share the same
    residual-stream dimensionality and (loosely) the same training corpus
    so that "direction d in residual space" means something comparable.
    """
    da = normalize_rows(decoder_a.detach().T)  # (d_hidden_a, d_in), rows are features
    db = normalize_rows(decoder_b.detach().T)
    return da @ db.T  # (d_hidden_a, d_hidden_b)


def hungarian_match(
    sim: torch.Tensor,
    top_k_a: int | None = None,
    top_k_b: int | None = None,
) -> list[FeatureMatch]:
    """One-to-one matching maximizing total similarity.

    If both SAEs have the same number of features (typical for matched
    training), full assignment is square. For very large dictionaries the
    cost matrix is too big to assign exactly; pass top_k_* to restrict to
    each side's most-firing features.
    """
    if top_k_a is not None and top_k_a < sim.shape[0]:
        keep_a = torch.arange(top_k_a)
    else:
        keep_a = torch.arange(sim.shape[0])
    if top_k_b is not None and top_k_b < sim.shape[1]:
        keep_b = torch.arange(top_k_b)
    else:
        keep_b = torch.arange(sim.shape[1])

    sub = sim[keep_a][:, keep_b].detach().cpu().numpy()
    # linear_sum_assignment minimizes; we want to maximize cos sim
    row_ind, col_ind = linear_sum_assignment(-sub)
    matches: list[FeatureMatch] = []
    for r, c in zip(row_ind, col_ind):
        matches.append(
            FeatureMatch(
                a=int(keep_a[r].item()),
                b=int(keep_b[c].item()),
                decoder_cos=float(sub[r, c]),
            )
        )
    matches.sort(key=lambda m: -m.decoder_cos)
    return matches


@torch.no_grad()
def activation_correlation_matrix(
    acts_a: torch.Tensor,    # (N, d_hidden_a)
    acts_b: torch.Tensor,    # (N, d_hidden_b)
    min_fires: int = 30,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full pairwise Pearson correlation matrix between A and B features.

    Returns (corr, active_a, active_b) where:
    - corr has shape (n_active_a, n_active_b)
    - active_a is an int tensor of shape (n_active_a,) mapping reduced row
      index → original A feature id
    - active_b is the analogous mapping for B

    `min_fires`: features whose activation is non-zero in fewer than this
    many of N tokens are dropped to avoid spurious correlations from rare-fire
    features. With N=30k, min_fires=30 keeps features firing on at least 0.1%
    of tokens.
    """
    a = acts_a.float()
    b = acts_b.float()
    fires_a = (a > 0).sum(dim=0)
    fires_b = (b > 0).sum(dim=0)
    active_a = (fires_a >= min_fires).nonzero(as_tuple=True)[0]
    active_b = (fires_b >= min_fires).nonzero(as_tuple=True)[0]

    a = a[:, active_a]
    b = b[:, active_b]

    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)
    a_std = a.std(dim=0).clamp_min(1e-8)
    b_std = b.std(dim=0).clamp_min(1e-8)
    a = a / a_std
    b = b / b_std
    # Pearson correlation matrix in one matmul (after standardization)
    corr = (a.T @ b) / a.shape[0]
    return corr, active_a, active_b


@torch.no_grad()
def activation_correlation(
    acts_a: torch.Tensor,    # (N, d_hidden_a)
    acts_b: torch.Tensor,    # (N, d_hidden_b)
    pairs: list[FeatureMatch],
) -> list[FeatureMatch]:
    """Annotate each match with Pearson correlation of its activation traces.

    The N rows in both tensors must correspond to *the same* tokens — i.e.,
    same texts fed through both base models then both SAEs at the matched
    layer, in the same shuffle order.
    """
    a = acts_a.float()
    b = acts_b.float()
    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)
    a_std = a.std(dim=0).clamp_min(1e-8)
    b_std = b.std(dim=0).clamp_min(1e-8)

    annotated: list[FeatureMatch] = []
    for m in pairs:
        col_a = a[:, m.a]
        col_b = b[:, m.b]
        corr = (col_a * col_b).mean() / (a_std[m.a] * b_std[m.b])
        annotated.append(
            FeatureMatch(
                a=m.a,
                b=m.b,
                decoder_cos=m.decoder_cos,
                act_corr=float(corr.item()),
            )
        )
    return annotated


def summarize(
    matches: list[FeatureMatch],
    cos_threshold: float = 0.5,
    corr_threshold: float = 0.3,
) -> dict:
    """Bucket the matches into shared / weak / divergent."""
    shared = [
        m for m in matches
        if m.decoder_cos > cos_threshold
        and (m.act_corr is None or m.act_corr > corr_threshold)
    ]
    weak = [m for m in matches if m not in shared and m.decoder_cos > cos_threshold * 0.5]
    divergent = [m for m in matches if m not in shared and m not in weak]
    return {
        "n_shared": len(shared),
        "n_weak": len(weak),
        "n_divergent": len(divergent),
        "median_cos": float(np.median([m.decoder_cos for m in matches])),
        "max_cos": float(max(m.decoder_cos for m in matches)),
        "thresholds": {"cos": cos_threshold, "corr": corr_threshold},
    }
