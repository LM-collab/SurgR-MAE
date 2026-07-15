"""
Shared video ViT encoder for SurgR2-MAE.

"""
from __future__ import annotations

import math
from typing import Optional, Tuple
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional embedding (sin-cos, separable t and (h, w))
# ---------------------------------------------------------------------------
def _get_1d_sincos_pos_embed(embed_dim: int, length: int) -> torch.Tensor:
    pos = torch.arange(length, dtype=torch.float32)
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2.0)))
    out = torch.einsum("m,d->md", pos, omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # (length, dim)


def _get_3d_sincos_pos_embed(
    embed_dim: int, t_len: int, h_len: int, w_len: int,
) -> torch.Tensor:
  
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 3D sincos"
    dim_t = embed_dim // 2
    dim_s = embed_dim // 2
    pe_t = _get_1d_sincos_pos_embed(dim_t, t_len)             # (T, dim_t)
    pe_h = _get_1d_sincos_pos_embed(dim_s // 2, h_len)        # (H, dim_s/2)
    pe_w = _get_1d_sincos_pos_embed(dim_s // 2, w_len)        # (W, dim_s/2)

    pe_hw = torch.cat([
        pe_h.unsqueeze(1).expand(-1, w_len, -1),
        pe_w.unsqueeze(0).expand(h_len, -1, -1),
    ], dim=-1)                                                # (H, W, dim_s)
    pe_hw = pe_hw.reshape(h_len * w_len, dim_s)

    pe_t = pe_t.unsqueeze(1).expand(-1, h_len * w_len, -1)
    pe_hw = pe_hw.unsqueeze(0).expand(t_len, -1, -1)

    pe = torch.cat([pe_t, pe_hw], dim=-1)                     # (T, HW, dim)
    return pe.reshape(t_len * h_len * w_len, embed_dim)


# ---------------------------------------------------------------------------
# Transformer block (standard pre-norm)
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # PyTorch SDPA; falls back to flash attention when available
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class Mlp(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep) / keep
        return x * mask


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float,
                 drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Tubelet embedding
# ---------------------------------------------------------------------------
class TubeletEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int,
                 tubelet_t: int, patch_h: int, patch_w: int):
        super().__init__()
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet_t, patch_h, patch_w),
            stride=(tubelet_t, patch_h, patch_w),
        )
        self.tubelet_t = tubelet_t
        self.patch_h = patch_h
        self.patch_w = patch_w

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        # x: (B, 3, T, H, W)
        x = self.proj(x)
        B, D, t, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)   # (B, t*h*w, D)
        return x, (t, h, w)


# ---------------------------------------------------------------------------
# Video ViT encoder
# ---------------------------------------------------------------------------
class VideoViTEncoder(nn.Module):
    """Shared video ViT encoder.
    """

    def __init__(self,
                 img_size: int = 224,
                 num_frames: int = 16,
                 tubelet_t: int = 2,
                 patch_h: int = 16,
                 patch_w: int = 16,
                 embed_dim: int = 384,
                 depth: int = 12,
                 num_heads: int = 6,
                 mlp_ratio: float = 4.0,
                 drop_path: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.t_len = num_frames // tubelet_t
        self.h_len = img_size // patch_h
        self.w_len = img_size // patch_w
        self.num_tokens = self.t_len * self.h_len * self.w_len

        self.patch_embed = TubeletEmbed(3, embed_dim, tubelet_t, patch_h, patch_w)

        # Fixed sin-cos pos embed
        pe = _get_3d_sincos_pos_embed(embed_dim, self.t_len, self.h_len, self.w_len)
        self.register_buffer("pos_embed", pe.unsqueeze(0), persistent=False)

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @property
    def grid_size(self) -> Tuple[int, int, int]:
        return (self.t_len, self.h_len, self.w_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # x: (B, T, 3, H, W) -> (B, 3, T, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x, _ = self.patch_embed(x)
        x = x + self.pos_embed.to(x.dtype)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x  # (B, N, D)

    def forward_masked(
        self,
        x: torch.Tensor,
        keep_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward only visible tokens.

        """
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x, _ = self.patch_embed(x)
        x = x + self.pos_embed.to(x.dtype)        # (B, N, D)

        B, N, D = x.shape
        idx = keep_indices.unsqueeze(-1).expand(-1, -1, D)
        x = torch.gather(x, dim=1, index=idx)     # (B, N_keep, D)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


def build_encoder(cfg) -> VideoViTEncoder:
    return VideoViTEncoder(
        img_size=cfg.data.image_size,
        num_frames=cfg.data.num_frames,
        tubelet_t=cfg.model.tubelet_t,
        patch_h=cfg.model.tubelet_h,
        patch_w=cfg.model.tubelet_w,
        embed_dim=cfg.model.embed_dim,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
    )
