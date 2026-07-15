"""
SurgR2-MAE model.

Combines:
  - shared video ViT encoder
  - momentum encoder (EMA)
  - recoverability matcher (sparse unbalanced OT + reliability gate)
  - cross-view driver decoder + non-driver decoder
  - reliability-gated reconstruction ), semantic alignment,
    and temporal consensus losses; full objective.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit_encoder import VideoViTEncoder, build_encoder
from .recoverability_ot import RecoverabilityMatcher
from .masking import random_keep_mask, complementary_keep_mask
from .decoder import StandardDecoder, CrossViewDriverDecoder


def _patchify(
    x: torch.Tensor,
    tubelet_t: int,
    patch_h: int,
    patch_w: int,
) -> torch.Tensor:
    """(B, T, 3, H, W) -> (B, N, P), token order matches the encoder."""
    B, T, C, H, W = x.shape
    t = T // tubelet_t
    h = H // patch_h
    w = W // patch_w
    x = x.view(B, t, tubelet_t, C, h, patch_h, w, patch_w)
    x = x.permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()
    x = x.view(B, t * h * w, tubelet_t * patch_h * patch_w * C)
    return x


def _normalize_target_per_patch(target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-patch mean/std normalization (VideoMAE convention).
    """
    mean = target.mean(dim=-1, keepdim=True)
    var = target.var(dim=-1, keepdim=True, unbiased=False)
    return (target - mean) / (var + eps).sqrt()


def _pool_by_time(
    z: torch.Tensor,          # (B, N_vis, D) online features of visible tokens
    idx: torch.Tensor,        # (B, N_vis) flat token indices
    t_grid: int,
    hw: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    B, Nv, D = z.shape
    t_idx = idx // hw                                        # (B, N_vis)
    sums = z.new_zeros(B, t_grid, D)
    sums.scatter_add_(1, t_idx.unsqueeze(-1).expand(-1, -1, D), z)
    cnt = z.new_zeros(B, t_grid)
    cnt.scatter_add_(1, t_idx, torch.ones_like(t_idx, dtype=z.dtype))
    h = sums / cnt.clamp_min(1.0).unsqueeze(-1)
    return h, (cnt > 0)


def _gate_time_weights(
    gate: torch.Tensor,       # (B, N_md) reliability gate (already elem-masked)
    drv_mask_idx: torch.Tensor,  # (B, N_md)
    t_grid: int,
    hw: int,
) -> torch.Tensor:
    
    t_idx = drv_mask_idx // hw
    s = gate.new_zeros(gate.shape[0], t_grid)
    s.scatter_add_(1, t_idx, gate)
    c = gate.new_zeros(gate.shape[0], t_grid)
    c.scatter_add_(1, t_idx, torch.ones_like(gate))
    return s / c.clamp_min(1.0)


# ---------------------------------------------------------------------------
# SurgR2-MAE pretraining model
# ---------------------------------------------------------------------------
class SurgR2MAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Online encoder
        self.encoder = build_encoder(cfg)

        # Momentum encoder (no grad)
        self.momentum_encoder = build_encoder(cfg)
        for p_m, p_o in zip(self.momentum_encoder.parameters(),
                            self.encoder.parameters()):
            p_m.data.copy_(p_o.data)
            p_m.requires_grad = False

        self.matcher = RecoverabilityMatcher(cfg)

        grid = self.encoder.grid_size
        n_tokens = self.encoder.num_tokens
        patch_pixel_dim = (cfg.model.tubelet_t * cfg.model.tubelet_h
                           * cfg.model.tubelet_w * 3)

        self.driver_decoder = CrossViewDriverDecoder(
            encoder_dim=cfg.model.embed_dim,
            num_tokens=n_tokens,
            grid=grid,
            patch_pixel_dim=patch_pixel_dim,
            decoder_dim=cfg.model.decoder_dim,
            depth=cfg.model.decoder_depth,
            num_heads=cfg.model.decoder_heads,
        )
        self.support_decoder = StandardDecoder(
            encoder_dim=cfg.model.embed_dim,
            num_tokens=n_tokens,
            grid=grid,
            patch_pixel_dim=patch_pixel_dim,
            decoder_dim=cfg.model.decoder_dim,
            depth=cfg.model.decoder_depth,
            num_heads=cfg.model.decoder_heads,
        )

       
        d_dec = cfg.model.decoder_dim
        d_enc = cfg.model.embed_dim
        self.q_d = nn.Sequential(
            nn.Linear(d_dec, d_dec), nn.GELU(), nn.Linear(d_dec, d_dec),
        )
       
        self.q_s = nn.Sequential(
            nn.Linear(d_enc, d_dec), nn.GELU(), nn.Linear(d_dec, d_dec),
        )

        # Hyperparameters
        self.tubelet_t = cfg.model.tubelet_t
        self.patch_h = cfg.model.tubelet_h
        self.patch_w = cfg.model.tubelet_w
        self.driver_mask_ratio = cfg.pretrain.driver_mask_ratio
        self.support_mask_ratio = cfg.pretrain.support_mask_ratio
        self.protect_fraction = getattr(cfg.pretrain, "protect_fraction", 0.5)
        self.ablation = getattr(cfg.pretrain, "ablation", "full")
        self.lambda_g = cfg.pretrain.lambda_g
        self.lambda_s = cfg.pretrain.lambda_s
        self.lambda_align = cfg.pretrain.lambda_align
        self.lambda_temp = cfg.pretrain.lambda_temp_loss
        self.momentum_coef = cfg.pretrain.momentum

    @torch.no_grad()
    def update_momentum(self) -> None:
        m = self.momentum_coef
        for p_m, p_o in zip(self.momentum_encoder.parameters(),
                            self.encoder.parameters()):
            p_m.data.mul_(m).add_(p_o.data, alpha=1.0 - m)

    # -----------------------------------------------------------------------
    def _fallback_single_view(
        self,
        views: torch.Tensor,
        driver_view: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Degenerate-batch fallback: standard MAE on the driver view only."""
        B = views.shape[0]
        N = self.encoder.num_tokens
        drv_keep_idx, drv_mask_idx = random_keep_mask(
            B, N, self.driver_mask_ratio, device,
        )
        z_drv_online = self.encoder.forward_masked(views[:, driver_view], drv_keep_idx)
        recon = self.support_decoder(
            z_visible=z_drv_online,
            keep_idx=drv_keep_idx,
            mask_idx=drv_mask_idx,
        )
        target = _patchify(views[:, driver_view], self.tubelet_t,
                           self.patch_h, self.patch_w)
        target = _normalize_target_per_patch(target)
        target = torch.gather(
            target, 1,
            drv_mask_idx.unsqueeze(-1).expand(-1, -1, target.shape[-1]),
        )
        L = F.mse_loss(recon, target)
        zero = torch.tensor(0.0, device=device)
        return {
            "loss": L,
            "loss_rec": L.detach(),
            "loss_drec": L.detach(),
            "loss_srec": zero, "loss_align": zero, "loss_temp": zero,
            "mean_recoverability": zero,
            "driver_view": driver_view,
        }

    # -----------------------------------------------------------------------
    # Pretraining forward
    # -----------------------------------------------------------------------
    def forward(
        self,
        views: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """views: (B, V, T, 3, H, W).
        view_mask: (B, V) bool — True for views actually present. Padded
        views are excluded from the reliability EMA, gated losses, the
        supporting reconstruction loss, and the temporal-consensus weights.
        """
        B, V, T, C, H, W = views.shape
        device = views.device
        N = self.encoder.num_tokens
        grid = self.encoder.grid_size
        T_grid, H_grid, W_grid = grid
        HW = H_grid * W_grid

        #Pick driver view — prefer one present in every batch element.
        if view_mask is not None:
            all_present = view_mask.all(dim=0)
            present_idxs = torch.nonzero(all_present, as_tuple=False).flatten().tolist()
            if not present_idxs:
                any_present = view_mask.any(dim=0)
                present_idxs = torch.nonzero(any_present, as_tuple=False).flatten().tolist()
            if not present_idxs:
                present_idxs = list(range(V))
            driver_view = random.choice(present_idxs)
            support_views = [v for v in range(V)
                             if v != driver_view and bool(view_mask[:, v].any())]
        else:
            driver_view = random.randrange(V)
            support_views = [v for v in range(V) if v != driver_view]

        if not support_views:
            return self._fallback_single_view(views, driver_view, device)

        # Per-element validity of the driver and of each supporting view.
        if view_mask is not None:
            drv_valid = view_mask[:, driver_view].to(views.dtype)        # (B,)
            sup_valid = {v: view_mask[:, v] for v in support_views}      # bool (B,)
        else:
            drv_valid = torch.ones(B, device=device, dtype=views.dtype)
            sup_valid = {v: None for v in support_views}

        #Heavy random masking on the driver view
        drv_keep_idx, drv_mask_idx = random_keep_mask(
            B, N, self.driver_mask_ratio, device,
        )

        #Momentum features for all views (matching only; no grad)
        with torch.no_grad():
            mom_feats: List[torch.Tensor] = [
                self.momentum_encoder(views[:, v]) for v in range(V)
            ]

        D_enc = mom_feats[0].shape[-1]
        z_drv_mom_masked = torch.gather(
            mom_feats[driver_view], 1,
            drv_mask_idx.unsqueeze(-1).expand(-1, -1, D_enc),
        )                                                  # (B, N_md, D)

        #OT matching per (driver, support) pair — elem_valid honored [C4]
        match_results: Dict[int, Dict[str, torch.Tensor]] = {}
        for v in support_views:
            match_results[v] = self.matcher.match(
                z_drv_masked=z_drv_mom_masked,
                drv_idx=drv_mask_idx,
                z_sup_all=mom_feats[v],
                grid=grid,
                driver_view=driver_view,
                support_view=v,
                elem_valid=sup_valid[v],
            )

        #Complementary masking on supporting views (protected first) 
        sup_keep_idx: Dict[int, torch.Tensor] = {}
        sup_mask_idx: Dict[int, torch.Tensor] = {}
        sup_n_protect: Dict[int, int] = {}
        for v in support_views:
            if self.ablation == "random_masking":
             
                keep, msk = random_keep_mask(
                    B, N, self.support_mask_ratio, device,
                )
                n_prot = keep.shape[1]
            else:
                keep, msk, n_prot = complementary_keep_mask(
                    support_score=match_results[v]["support_score"],
                    mask_ratio=self.support_mask_ratio,
                    protect_fraction=self.protect_fraction,
                )
            sup_keep_idx[v] = keep
            sup_mask_idx[v] = msk
            sup_n_protect[v] = n_prot

        #Online encoder on visible tokens
        z_drv_online = self.encoder.forward_masked(
            views[:, driver_view], drv_keep_idx,
        )                                                  # (B, N_keep_d, D)
        z_sup_online: Dict[int, torch.Tensor] = {}
        for v in support_views:
            z_sup_online[v] = self.encoder.forward_masked(
                views[:, v], sup_keep_idx[v],
            )

        #Cross-attention KV: PROTECTED supporting tokens only
        sup_kv = torch.cat(
            [z_sup_online[v][:, :sup_n_protect[v]] for v in support_views],
            dim=1,
        )

        # 8. Driver reconstruction + decoder hidden states h^d_i [C1]
        recon_drv, h_dec = self.driver_decoder(
            z_drv_visible=z_drv_online,
            drv_keep_idx=drv_keep_idx,
            drv_mask_idx=drv_mask_idx,
            sup_protected_tokens=sup_kv,
            return_hidden=True,
        )                                                  # (B,N_md,P), (B,N_md,D_dec)

        all_patches = _patchify(
            views[:, driver_view], self.tubelet_t, self.patch_h, self.patch_w,
        )
        all_patches = _normalize_target_per_patch(all_patches)
        target_drv = torch.gather(
            all_patches, 1,
            drv_mask_idx.unsqueeze(-1).expand(-1, -1, all_patches.shape[-1]),
        )

        # Reliability-weighted driver loss 
        gates = torch.stack(
            [match_results[v]["gate"] for v in support_views], dim=0,
        )                                                  # (Vs, B, N_md)
        if view_mask is not None:
            n_present = torch.stack(
                [sup_valid[v].to(gates.dtype) for v in support_views], dim=0,
            ).sum(dim=0).clamp_min(1.0)                    # (B,)
            g_bar = gates.sum(dim=0) / n_present.unsqueeze(1)
        else:
            g_bar = gates.mean(dim=0)                      # (B, N_md)

        per_token_se = ((recon_drv - target_drv) ** 2).mean(dim=-1)  # (B, N_md)
        weight = 1.0 + self.lambda_g * g_bar
        per_elem_drec = (weight * per_token_se).mean(dim=1)          # (B,)
        L_drec = (per_elem_drec * drv_valid).sum() / drv_valid.sum().clamp_min(1.0)

        #Standard MAE loss for supporting views, padded elems excluded [C4]
        L_srec = recon_drv.new_zeros(())
        n_sup_terms = 0
        for v in support_views:
            recon_v = self.support_decoder(
                z_visible=z_sup_online[v],
                keep_idx=sup_keep_idx[v],
                mask_idx=sup_mask_idx[v],
            )
            target_v = _patchify(views[:, v], self.tubelet_t,
                                 self.patch_h, self.patch_w)
            target_v = _normalize_target_per_patch(target_v)
            target_v = torch.gather(
                target_v, 1,
                sup_mask_idx[v].unsqueeze(-1).expand(-1, -1, target_v.shape[-1]),
            )
            per_elem = ((recon_v - target_v) ** 2).mean(dim=(1, 2))  # (B,)
            if sup_valid[v] is not None:
                w = sup_valid[v].to(per_elem.dtype)
                L_srec = L_srec + (per_elem * w).sum() / w.sum().clamp_min(1.0)
            else:
                L_srec = L_srec + per_elem.mean()
            n_sup_terms += 1
        L_srec = L_srec / max(n_sup_terms, 1)

        L_rec = L_drec + self.lambda_s * L_srec

        #Reliability-gated semantic alignment 
        u_d = self.q_d(h_dec)                              # (B, N_md, D_dec)
        L_align = h_dec.new_zeros(())
        n_pairs = 0
        for v in support_views:
            r = match_results[v]
            pi_cand = r["pi"][..., :-1]                    # (B, N_md, K)
            row_sum = pi_cand.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            w = pi_cand / row_sum
            with torch.no_grad():
                z_tilde = (w.unsqueeze(-1) * r["z_cand"]).sum(dim=2)  # (B,N_md,D_enc)
                u_tilde = self.q_s(z_tilde)                # stop-grad target
            cos = F.cosine_similarity(u_d, u_tilde, dim=-1)  # (B, N_md)
            gate = r["gate"]                               # zeroed for padded elems
            L_align = L_align + (gate * (1.0 - cos)).mean()
            n_pairs += 1
        L_align = L_align / max(n_pairs, 1)

        # Reliability-gated temporal consensus
        L_temp = self._temporal_consensus_loss(
            z_drv_online, drv_keep_idx,
            z_sup_online, sup_keep_idx,
            driver_view, support_views,
            match_results, drv_mask_idx,
            T_grid, HW, drv_valid, sup_valid,
        )

        L_total = L_rec + self.lambda_align * L_align + self.lambda_temp * L_temp

        return {
            "loss": L_total,
            "loss_rec": L_rec.detach(),
            "loss_drec": L_drec.detach(),
            "loss_srec": L_srec.detach(),
            "loss_align": L_align.detach(),
            "loss_temp": L_temp.detach(),
            "mean_recoverability": torch.stack(
                [match_results[v]["recoverability"].mean() for v in support_views],
            ).mean().detach(),
            "driver_view": driver_view,
        }

    # -----------------------------------------------------------------------
    def _temporal_consensus_loss(
        self,
        z_drv_online: torch.Tensor,
        drv_keep_idx: torch.Tensor,
        z_sup_online: Dict[int, torch.Tensor],
        sup_keep_idx: Dict[int, torch.Tensor],
        driver_view: int,
        support_views: List[int],
        match_results: Dict[int, Dict[str, torch.Tensor]],
        drv_mask_idx: torch.Tensor,
        t_grid: int,
        hw: int,
        drv_valid: torch.Tensor,                       # (B,) float
        sup_valid: Dict[int, Optional[torch.Tensor]],  # (B,) bool or None
    ) -> torch.Tensor:
        """computed from ONLINE features so it trains the encoder.

        
        """
        # Per-view time-pooled online features and slice-validity
        h_views: List[torch.Tensor] = []
        slice_valid: List[torch.Tensor] = []
        r_views: List[torch.Tensor] = []

        # Supporting views first (their r^v_t define r^d_t).
        
        r_sup_raw = []
        for v in support_views:
            h_v, sv = _pool_by_time(z_sup_online[v], sup_keep_idx[v], t_grid, hw)
            # gate already zeroed for padded elements
            r_v = _gate_time_weights(
                match_results[v]["gate"].detach(), drv_mask_idx, t_grid, hw,
            )                                              # (B, T)
            if sup_valid[v] is not None:
                r_v = r_v * sup_valid[v].to(r_v.dtype).unsqueeze(1)
            r_sup_raw.append(r_v)
            r_v = r_v * sv.to(r_v.dtype)                   # zero empty slices
            h_views.append(h_v)
            slice_valid.append(sv)
            r_views.append(r_v)

        # Driver
        h_d, sd = _pool_by_time(z_drv_online, drv_keep_idx, t_grid, hw)
        r_d = torch.stack(r_sup_raw, dim=0).mean(dim=0)    # (B, T), Eq. 9
        r_d = r_d * drv_valid.unsqueeze(1) * sd.to(r_d.dtype)
        h_views.insert(0, h_d)
        slice_valid.insert(0, sd)
        r_views.insert(0, r_d)

        H_stack = torch.stack(h_views, dim=0)              # (U, B, T, D)
        R_stack = torch.stack(r_views, dim=0)              # (U, B, T)

        # Reliability-weighted consensus, stop-grad (prevents collapse)
        denom = R_stack.sum(dim=0).clamp_min(1e-8)         # (B, T)
        c = (R_stack.unsqueeze(-1) * H_stack).sum(dim=0) / denom.unsqueeze(-1)
        c = c.detach()                                     # (B, T, D)

        cos = F.cosine_similarity(H_stack, c.unsqueeze(0), dim=-1)  # (U, B, T)
        L = (R_stack * (1.0 - cos)).sum() / (
            R_stack.shape[0] * R_stack.shape[1] * R_stack.shape[2]
        )
        return L


# ---------------------------------------------------------------------------
# Fine-tuning model: encoder + gesture head + gesture-conditioned error head
# ---------------------------------------------------------------------------
class SurgR2Classifier(nn.Module):
    def __init__(self, cfg, encoder: Optional[VideoViTEncoder] = None):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder if encoder is not None else build_encoder(cfg)

        D = cfg.model.embed_dim
        self.norm = nn.LayerNorm(D)
        self.gesture_head = nn.Linear(D, cfg.model.num_gesture_classes)

        Cg = cfg.model.num_gesture_classes
        self.gesture_embed = nn.Linear(Cg, D, bias=False)
        
        cond = getattr(cfg.finetune, "gesture_conditioning", None)
        if cond is None:  # legacy flag
            cond = "soft" if cfg.finetune.use_gesture_conditioning else "none"
        if not getattr(cfg.finetune, "use_gesture_conditioning", True):
            cond = "none"
        assert cond in ("soft", "hard", "none")
        self.gesture_cond = cond
        if cond in ("soft", "hard"):
            self.error_head = nn.Linear(D + D, cfg.model.num_error_classes)
        else:
            self.error_head = nn.Linear(D, cfg.model.num_error_classes)

    def forward_single_view(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = self.norm(z)
        return z.mean(dim=1)

    def forward(
        self,
        views: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, V, T, C, H, W = views.shape
        reps = [self.forward_single_view(views[:, v]) for v in range(V)]
        reps_t = torch.stack(reps, dim=1)                        # (B, V, D)

        if view_mask is None:
            z = reps_t.mean(dim=1)
        else:
            mask = view_mask.to(reps_t.dtype).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            z = (reps_t * mask).sum(dim=1) / denom

        gesture_logits = self.gesture_head(z)
        if self.gesture_cond == "soft":
            pg = F.softmax(gesture_logits, dim=-1)
            err_input = torch.cat([z, self.gesture_embed(pg)], dim=-1)
        elif self.gesture_cond == "hard":
            # One-hot argmax; no gradient flows through the conditioning path.
            idx = gesture_logits.argmax(dim=-1)
            pg = F.one_hot(idx, gesture_logits.shape[-1]).to(z.dtype).detach()
            err_input = torch.cat([z, self.gesture_embed(pg)], dim=-1)
        else:
            err_input = z
        error_logits = self.error_head(err_input)

        return {
            "gesture_logits": gesture_logits,
            "error_logits": error_logits,
            "z": z,
        }
