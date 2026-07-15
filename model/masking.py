"""
Masking utilities for SurgR2-MAE.

"""
from __future__ import annotations

from typing import Tuple

import torch


def random_keep_mask(
    batch_size: int,
    num_tokens: int,
    mask_ratio: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random masking. Returns (keep_idx, mask_idx) of shapes
    (B, N_keep) and (B, N_mask) — flat token indices.
    """
    n_keep = int(num_tokens * (1.0 - mask_ratio))
    n_keep = max(1, n_keep)

    noise = torch.rand(batch_size, num_tokens, device=device)
    perm = torch.argsort(noise, dim=1)
    keep_idx = perm[:, :n_keep]
    mask_idx = perm[:, n_keep:]
    return keep_idx, mask_idx


def complementary_keep_mask(
    support_score: torch.Tensor,    # (B, N) support score for a supporting view
    mask_ratio: float,
    protect_fraction: float = 0.5,  # of the kept tokens, this fraction is protected
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Hybrid keep mask: top-scoring tokens are protected (always kept), the
    remaining keep budget is filled randomly from the unprotected pool.

    Returns:
        keep_idx: (B, N_keep) — the FIRST `n_protect` entries of each row are
            the protected tokens (highest support score first); the rest are
            randomly kept. NOT sorted, by design.
        mask_idx: (B, N_mask)
        n_protect: int
    """
    B, N = support_score.shape
    device = support_score.device
    n_keep = max(1, int(N * (1.0 - mask_ratio)))
    n_protect = max(0, min(n_keep, int(n_keep * protect_fraction)))
    n_random_keep = n_keep - n_protect

    if n_protect > 0:
        _, top_idx = torch.topk(support_score, k=n_protect, dim=1)
    else:
        top_idx = torch.zeros(B, 0, dtype=torch.long, device=device)

    is_protected = torch.zeros(B, N, dtype=torch.bool, device=device)
    if n_protect > 0:
        is_protected.scatter_(1, top_idx, True)

    rand_prio = torch.rand(B, N, device=device)
    rand_among_unprotected = torch.where(
        is_protected, torch.full_like(rand_prio, -float("inf")), rand_prio,
    )
    _, rand_keep_idx = torch.topk(rand_among_unprotected, k=n_random_keep, dim=1)

    keep_idx = torch.cat([top_idx, rand_keep_idx], dim=1)   # protected FIRST

    # Complement -> mask_idx
    full = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
    keep_set = torch.zeros(B, N, dtype=torch.bool, device=device)
    keep_set.scatter_(1, keep_idx, True)
    mask_idx = full[~keep_set].view(B, N - n_keep)
    return keep_idx, mask_idx, n_protect
