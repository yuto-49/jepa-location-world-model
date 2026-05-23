"""Trajectory token embeddings: geocell + hour + dow + positional encoding."""

import math
import torch
import torch.nn as nn


class TrajectoryTokenizer(nn.Module):
    """Converts (cell_ids, hours, dows) -> (B, T, D) token embeddings.
    Embedding = cell_emb + hour_emb + dow_emb + pos_encoding
    """

    def __init__(self, vocab_size: int, d_model: int = 128, max_len: int = 64):
        super().__init__()
        self.d_model = d_model
        self.cell_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.hour_emb = nn.Embedding(24, d_model)
        self.dow_emb = nn.Embedding(7, d_model)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, T, D)

    def forward(self, cell_ids: torch.Tensor, hours: torch.Tensor, dows: torch.Tensor) -> torch.Tensor:
        T = cell_ids.size(1)
        emb = self.cell_emb(cell_ids) + self.hour_emb(hours) + self.dow_emb(dows)
        emb = emb + self.pe[:, :T, :]
        return emb
