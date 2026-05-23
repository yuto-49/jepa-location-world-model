"""Predictor: smaller Transformer that predicts target latents from context."""

import torch
import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self, d_model: int = 128, n_layers: int = 2, n_heads: int = 4,
                 d_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, ctx_repr: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """ctx_repr: (B, T, D), tgt_mask: (B, T) bool True at target positions"""
        B, T, D = ctx_repr.shape
        x = ctx_repr.clone()
        n_tgt = int(tgt_mask.sum().item())
        x[tgt_mask] = self.mask_token.squeeze(0).expand(n_tgt, -1)
        out = self.transformer(x)
        return self.norm(out)
