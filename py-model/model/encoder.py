"""Transformer encoder used for both context and target encoders."""

import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int = 128, n_layers: int = 4, n_heads: int = 4,
                 d_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, D), key_padding_mask: (B, T) True where PADDED"""
        out = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.norm(out)
