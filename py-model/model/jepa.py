"""JEPA model: context encoder + EMA target encoder + predictor."""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.tokenizer import TrajectoryTokenizer
from model.encoder import TransformerEncoder
from model.predictor import Predictor


class JEPAModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, max_len: int = 64,
                 enc_layers: int = 4, pred_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.tokenizer = TrajectoryTokenizer(vocab_size, d_model, max_len)
        self.context_encoder = TransformerEncoder(d_model, enc_layers, n_heads)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.predictor = Predictor(d_model, pred_layers, n_heads)
        self.target_norm = nn.LayerNorm(d_model)

        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        tok_emb = self.tokenizer(batch["cell_ids"], batch["hours"], batch["dows"])
        ctx_repr = self.context_encoder(tok_emb, key_padding_mask=~batch["ctx_mask"])

        with torch.no_grad():
            tgt_repr = self.target_encoder(tok_emb, key_padding_mask=~batch["pad_mask"])
            tgt_repr = self.target_norm(tgt_repr)

        pred = self.predictor(ctx_repr, batch["tgt_mask"])
        pred_tgt = pred[batch["tgt_mask"]]
        repr_tgt = tgt_repr[batch["tgt_mask"]]
        loss = F.smooth_l1_loss(pred_tgt, repr_tgt)

        with torch.no_grad():
            real_repr = tgt_repr[batch["pad_mask"]]
            emb_std = real_repr.std(dim=0).mean().item()
            cos = F.cosine_similarity(pred_tgt, repr_tgt, dim=-1).mean().item()

        return loss, {"emb_std": emb_std, "pred_target_cos": cos}

    @torch.no_grad()
    def ema_update(self, momentum: float = 0.996) -> None:
        for p_tgt, p_ctx in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            p_tgt.data.mul_(momentum).add_(p_ctx.data, alpha=1.0 - momentum)
