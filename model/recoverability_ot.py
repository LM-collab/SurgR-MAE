"""
Sparse unbalanced Sinkhorn-OT with explicit unmatched state.

"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers: convert flat token index to (t, h, w) coordinates
# ---------------------------------------------------------------------------
def flat_to_thw(idx: torch.Tensor, grid: Tuple[int, int, int]) -> torch.Tensor:
   
    T, H, W = grid
    HW = H * W
    t = idx // HW
    rem = idx % HW
    h = rem // W
    w = rem % W
    return torch.stack([t, h, w], dim=-1)


# ---------------------------------------------------------------------------
# Sparse candidate construction
# ---------------------------------------------------------------------------
def build_sparse_candidates(
    z_drv_masked: torch.Tensor,    # (B, N_md, D) — masked driver tokens (momentum)
    drv_idx: torch.Tensor,         # (B, N_md) — flat indices in driver view
    z_sup_all: torch.Tensor,       # (B, N, D) — ALL supporting tokens (momentum)
    grid: Tuple[int, int, int],
    topk: int,
    temporal_window: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """For each masked driver token, find top-k similar supporting tokens
    within a temporal window |dt| <= temporal_window.

    Returns:
        cand_idx: (B, N_md, K)  — flat token indices into z_sup_all
        cand_dt:  (B, N_md, K)  — temporal offsets |tau_j - tau_i|
    """
    B, N_md, D = z_drv_masked.shape
    _, N, _ = z_sup_all.shape
    device = z_drv_masked.device

    zd = F.normalize(z_drv_masked, dim=-1)
    zs = F.normalize(z_sup_all, dim=-1)
    sim = torch.einsum("bid,bjd->bij", zd, zs)

    drv_thw = flat_to_thw(drv_idx, grid)
    all_idx = torch.arange(N, device=device)
    all_thw = flat_to_thw(all_idx, grid).unsqueeze(0)

    drv_t = drv_thw[..., 0:1].float()
    all_t = all_thw[..., 0:1].float()
    dt = (drv_t - all_t.transpose(1, 2)).abs()

    valid = (dt <= float(temporal_window)).float()
    sim_masked = sim - 1e6 * (1.0 - valid)

    topk_actual = min(topk, N)
    _, cand_idx = torch.topk(sim_masked, k=topk_actual, dim=-1)
    cand_dt = torch.gather(dt, dim=-1, index=cand_idx)
    
    cand_valid = cand_dt <= float(temporal_window)

    return cand_idx, cand_dt, cand_valid


# ---------------------------------------------------------------------------
# Cost matrix with unmatched state
# ---------------------------------------------------------------------------
def compute_cost_with_null(
    z_drv: torch.Tensor,         # (B, N_md, D)
    z_sup_cand: torch.Tensor,    # (B, N_md, K, D)
    cand_dt: torch.Tensor,       # (B, N_md, K)
    alpha_f: float,
    alpha_t: float,
    delta_t: float,
    gamma_null: float,
    cand_valid: Optional[torch.Tensor] = None,   # (B, N_md, K) bool
) -> torch.Tensor:
    """Returns cost matrix of shape (B, N_md, K+1)"""
    B, N_md, K, D = z_sup_cand.shape

    zd = F.normalize(z_drv, dim=-1).unsqueeze(2)
    zs = F.normalize(z_sup_cand, dim=-1)
    cos = (zd * zs).sum(dim=-1)

    feat_cost = alpha_f * (1.0 - cos)
    temp_cost = alpha_t * (cand_dt / max(delta_t, 1.0))
    cost_cand = feat_cost + temp_cost
    if cand_valid is not None:
        cost_cand = torch.where(
            cand_valid, cost_cand, torch.full_like(cost_cand, 1e4),
        )

    null_col = torch.full(
        (B, N_md, 1), float(gamma_null),
        device=cost_cand.device, dtype=cost_cand.dtype,
    )
    cost = torch.cat([cost_cand, null_col], dim=-1)
    return cost


# ---------------------------------------------------------------------------
# Unbalanced Sinkhorn (log-domain, Chizat et al. 2018)
# ---------------------------------------------------------------------------
@torch.no_grad()
def unbalanced_sinkhorn(
    cost: torch.Tensor,        # (B, M, J)
    eps: float,
    lambda_r: float,
    lambda_c: float,
    n_iter: int,
    mu_r: Optional[torch.Tensor] = None,   # (B, M)
    nu_c: Optional[torch.Tensor] = None,   # (B, J)
) -> torch.Tensor:
    
    orig_dtype = cost.dtype
    cost = cost.float()
    B, M, J = cost.shape
    device = cost.device

    if mu_r is None:
        mu_r = torch.full((B, M), 1.0 / M, device=device, dtype=cost.dtype)
    if nu_c is None:
        nu_c = torch.full((B, J), 1.0 / J, device=device, dtype=cost.dtype)

    log_mu = torch.log(mu_r.clamp_min(1e-12))
    log_nu = torch.log(nu_c.clamp_min(1e-12))

    f = torch.zeros((B, M), device=device, dtype=cost.dtype)
    g = torch.zeros((B, J), device=device, dtype=cost.dtype)

    tau_r = lambda_r / (lambda_r + eps)
    tau_c = lambda_c / (lambda_c + eps)

    K = -cost / max(eps, 1e-8)

    for _ in range(n_iter):
        log_a = torch.logsumexp(K + g.unsqueeze(1) / max(eps, 1e-8), dim=-1)
        f = tau_r * (eps * (log_mu - log_a))
        log_b = torch.logsumexp(K + f.unsqueeze(2) / max(eps, 1e-8), dim=-2)
        g = tau_c * (eps * (log_nu - log_b))

    log_pi = K + (f.unsqueeze(2) + g.unsqueeze(1)) / max(eps, 1e-8)
    return log_pi.exp()


# ---------------------------------------------------------------------------
# Recoverability + reliability gate
# ---------------------------------------------------------------------------
class ViewPairReliability(nn.Module):
    

    def __init__(self, num_views: int, beta: float = 0.99):
        super().__init__()
        self.num_views = num_views
        self.beta = beta
        # Init at 1.0 so early training treats all pairs as fully reliable.
        self.register_buffer("rho", torch.ones(num_views, num_views))

    @torch.no_grad()
    def update(self, driver: int, support: int, mean_recov: torch.Tensor) -> None:
        if driver == support:
            return
        new = self.beta * self.rho[driver, support] + (1.0 - self.beta) * mean_recov.detach()
        self.rho[driver, support] = new.clamp(0.0, 1.0)

    def get(self, driver: int, support: int) -> torch.Tensor:
        return self.rho[driver, support]

    @torch.no_grad()
    def sync_ddp(self) -> None:
        
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.rho, op=torch.distributed.ReduceOp.SUM)
            self.rho.div_(torch.distributed.get_world_size())


def compute_recoverability_and_gate(
    pi: torch.Tensor,            # (B, M, K+1)
    rho_dv: torch.Tensor,
    eps_div: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    pi_cand = pi[..., :-1].sum(dim=-1)
    pi_null = pi[..., -1]
    total = pi_cand + pi_null + eps_div
    m = pi_cand / total

    rho = rho_dv if rho_dv.ndim > 0 else rho_dv.expand(m.shape[0])
    rho = rho.unsqueeze(1) if rho.ndim == 1 else rho
    g = torch.sqrt(m * rho.clamp(0.0, 1.0).to(m.dtype) + eps_div)
    return m, g


def compute_support_score(
    pi: torch.Tensor,            # (B, M, K+1)
    cand_idx: torch.Tensor,      # (B, M, K)
    g: torch.Tensor,             # (B, M)
    n_sup_tokens: int,
    eps_div: float = 1e-8,
) -> torch.Tensor:
    
    B, M, K1 = pi.shape
    K = K1 - 1
    pi_cand = pi[..., :K]
    row_sum = pi_cand.sum(dim=-1, keepdim=True) + eps_div
    pi_norm = pi_cand / row_sum

    weighted = pi_norm * g.unsqueeze(-1)

    score = torch.zeros(B, n_sup_tokens, device=pi.device, dtype=pi.dtype)
    score.scatter_add_(
        dim=1,
        index=cand_idx.reshape(B, M * K),
        src=weighted.reshape(B, M * K),
    )
    return score


# ---------------------------------------------------------------------------
# Top-level matching module per (driver, support) pair
# ---------------------------------------------------------------------------
class RecoverabilityMatcher(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        p = cfg.pretrain
        self.alpha_f = p.alpha_feat
        self.alpha_t = p.alpha_temp
        self.delta_t = float(p.temporal_window)
        self.gamma_null = p.gamma_unmatched
        self.eps = p.sinkhorn_eps
        self.lambda_r = p.lambda_row
        self.lambda_c = p.lambda_col
        self.n_iter = p.sinkhorn_iters
        self.topk = p.topk_candidates
        self.temporal_window = p.temporal_window
        self.reliability = ViewPairReliability(
            num_views=cfg.data.num_views, beta=p.reliability_beta,
        )
        # Table III ablations (see config.PretrainConfig.ablation)
        self.ablation = getattr(p, "ablation", "full")
        self.softmax_tau = getattr(p, "softmax_tau", 0.1)

    @torch.no_grad()
    def match(
        self,
        z_drv_masked: torch.Tensor,    # (B, N_md, D) momentum
        drv_idx: torch.Tensor,         # (B, N_md)
        z_sup_all: torch.Tensor,       # (B, N, D) momentum
        grid: Tuple[int, int, int],
        driver_view: int,
        support_view: int,
        elem_valid: Optional[torch.Tensor] = None,   # (B,) bool
    ) -> Dict[str, torch.Tensor]:
        
        cand_idx, cand_dt, cand_valid = build_sparse_candidates(
            z_drv_masked, drv_idx, z_sup_all, grid,
            topk=self.topk, temporal_window=self.temporal_window,
        )
        B, N_md, K = cand_idx.shape
        D = z_sup_all.shape[-1]
        gather_idx = cand_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        z_cand = torch.gather(
            z_sup_all.unsqueeze(1).expand(-1, N_md, -1, -1),
            dim=2, index=gather_idx,
        )

        cost = compute_cost_with_null(
            z_drv_masked, z_cand, cand_dt,
            self.alpha_f, self.alpha_t, self.delta_t, self.gamma_null,
            cand_valid=cand_valid,
        )

        if self.ablation == "softmax_null":
           
            pi = torch.softmax(-cost.float() / max(self.softmax_tau, 1e-6), dim=-1)
        elif self.ablation == "balanced_ot_no_null":
            
            pi_real = unbalanced_sinkhorn(
                cost[..., :-1], eps=self.eps,
                lambda_r=1e6, lambda_c=1e6, n_iter=self.n_iter,
            )
            pi = torch.cat([pi_real, torch.zeros_like(pi_real[..., :1])], dim=-1)
        else:
            pi = unbalanced_sinkhorn(
                cost, eps=self.eps,
                lambda_r=self.lambda_r, lambda_c=self.lambda_c,
                n_iter=self.n_iter,
            )

        rho = self.reliability.get(driver_view, support_view).to(pi.dtype)
        m, g = compute_recoverability_and_gate(pi, rho)
        if self.ablation == "no_gate":
            
            g = torch.ones_like(g)

        
        if elem_valid is not None:
            valid_f = elem_valid.to(m.dtype)                      # (B,)
            g = g * valid_f.unsqueeze(1)
            n_valid = valid_f.sum()
            if n_valid > 0:
                mean_recov = (m * valid_f.unsqueeze(1)).sum() / (n_valid * m.shape[1])
                self.reliability.update(driver_view, support_view, mean_recov)
        else:
            self.reliability.update(driver_view, support_view, m.mean())

        n_sup = z_sup_all.shape[1]
        s = compute_support_score(pi, cand_idx, g, n_sup_tokens=n_sup)

        return {
            "pi": pi,                       # (B, N_md, K+1)
            "cand_idx": cand_idx,           # (B, N_md, K)
            "cand_dt": cand_dt,             # (B, N_md, K)
            "recoverability": m,            # (B, N_md)
            "gate": g,                      # (B, N_md) — zeroed for invalid elems
            "support_score": s,             # (B, N_sup)
            "z_cand": z_cand,               # (B, N_md, K, D)
        }
