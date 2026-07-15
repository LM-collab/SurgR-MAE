"""
Decoders for SurgR2-MAE pretraining.

"""
from __future__ import annotations

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn

from .vit_encoder import Block, _get_3d_sincos_pos_embed


class StandardDecoder(nn.Module):
    """Standard MAE decoder: visible tokens + mask tokens -> reconstruct patches.

    Used for non-driver (supporting) views.
    """
    def __init__(
        self,
        encoder_dim: int,
        num_tokens: int,
        grid: tuple,
        patch_pixel_dim: int,
        decoder_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.grid = grid
        self.proj = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        pe = _get_3d_sincos_pos_embed(decoder_dim, *grid)
        self.register_buffer("pos_embed", pe.unsqueeze(0), persistent=False)

        self.blocks = nn.ModuleList([
            Block(decoder_dim, num_heads, mlp_ratio=4.0) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, patch_pixel_dim)

    def forward(
        self,
        z_visible: torch.Tensor,        # (B, N_keep, D_enc)
        keep_idx: torch.Tensor,         # (B, N_keep)
        mask_idx: torch.Tensor,         # (B, N_mask)
    ) -> torch.Tensor:
        """Returns reconstructed patches at masked positions: (B, N_mask, P)."""
        B, N_keep, _ = z_visible.shape
        D = self.proj.out_features

        z_visible = self.proj(z_visible)
        # Build full token sequence with mask tokens.
        # Cast mask_token to z_visible's dtype — autocast keeps the parameter
        # at fp32 but z_visible is bf16/fp16, and scatter_ requires matching
        # dtypes.
        x = self.mask_token.to(z_visible.dtype).expand(B, self.num_tokens, -1).clone()
        x.scatter_(1, keep_idx.unsqueeze(-1).expand(-1, -1, D), z_visible)

        x = x + self.pos_embed.to(x.dtype)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        out = torch.gather(x, dim=1, index=mask_idx.unsqueeze(-1).expand(-1, -1, D))
        out = self.head(out)
        return out


class CrossViewDriverDecoder(nn.Module):
   
    def __init__(
        self,
        encoder_dim: int,
        num_tokens: int,
        grid: tuple,
        patch_pixel_dim: int,
        decoder_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.decoder_dim = decoder_dim
        self.proj = nn.Linear(encoder_dim, decoder_dim)
        self.proj_sup = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        pe = _get_3d_sincos_pos_embed(decoder_dim, *grid)
        self.register_buffer("pos_embed", pe.unsqueeze(0), persistent=False)

        self.self_blocks = nn.ModuleList([
            Block(decoder_dim, num_heads, mlp_ratio=4.0) for _ in range(depth - 1)
        ])
        self.cross_attn = nn.MultiheadAttention(
            decoder_dim, num_heads, batch_first=True,
        )
        self.cross_norm_q = nn.LayerNorm(decoder_dim)
        self.cross_norm_kv = nn.LayerNorm(decoder_dim)
        self.cross_ff = nn.Sequential(
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, decoder_dim * 4),
            nn.GELU(),
            nn.Linear(decoder_dim * 4, decoder_dim),
        )
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, patch_pixel_dim)

    def forward(
        self,
        z_drv_visible: torch.Tensor,
        drv_keep_idx: torch.Tensor,
        drv_mask_idx: torch.Tensor,
        sup_protected_tokens: torch.Tensor,   # (B, N_prot_total, D_enc)
        return_hidden: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        B, N_keep, _ = z_drv_visible.shape
        D = self.decoder_dim

        z_vis = self.proj(z_drv_visible)
        x = self.mask_token.to(z_vis.dtype).expand(B, self.num_tokens, -1).clone()
        x.scatter_(1, drv_keep_idx.unsqueeze(-1).expand(-1, -1, D), z_vis)
        x = x + self.pos_embed.to(x.dtype)

        for blk in self.self_blocks:
            x = blk(x)

        kv = self.proj_sup(sup_protected_tokens)
        q = self.cross_norm_q(x)
        k = v = self.cross_norm_kv(kv)
        attn_out, _ = self.cross_attn(q, k, v, need_weights=False)
        x = x + attn_out
        x = x + self.cross_ff(x)

        x = self.norm(x)
        hidden = torch.gather(
            x, dim=1, index=drv_mask_idx.unsqueeze(-1).expand(-1, -1, D),
        )                                                 # (B, N_md, D_dec)
        out = self.head(hidden)
        if return_hidden:
            return out, hidden
        return out
