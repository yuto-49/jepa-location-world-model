"""PyTorch Dataset and DataLoader for geocell trajectory sequences."""

import json
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """Loads JSONL trajectories and produces padded/masked tensors.

    Tensor contract (per item):
      cell_ids  (T,) int64
      hours     (T,) int64
      dows      (T,) int64
      pad_mask  (T,) bool   — True where real token
      ctx_mask  (T,) bool   — True where context (visible)
      tgt_mask  (T,) bool   — True where target (predict)
    """

    def __init__(self, jsonl_path: str, vocab: dict[str, int], max_len: int = 64, ctx_frac: float = 0.75):
        self.vocab = vocab
        self.max_len = max_len
        self.ctx_frac = ctx_frac
        self.pad_idx = vocab["<PAD>"]
        self.unk_idx = vocab["<UNK>"]
        self.trajectories = []
        with open(jsonl_path) as f:
            for line in f:
                self.trajectories.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        traj = self.trajectories[idx]
        tokens = traj["tokens"][: self.max_len]
        seq_len = len(tokens)

        cell_ids = torch.full((self.max_len,), self.pad_idx, dtype=torch.long)
        hours = torch.zeros(self.max_len, dtype=torch.long)
        dows = torch.zeros(self.max_len, dtype=torch.long)
        pad_mask = torch.zeros(self.max_len, dtype=torch.bool)
        ctx_mask = torch.zeros(self.max_len, dtype=torch.bool)
        tgt_mask = torch.zeros(self.max_len, dtype=torch.bool)

        for i, token in enumerate(tokens):
            cell_ids[i] = self.vocab.get(token["geocell"], self.unk_idx)
            hours[i] = token["hour"]
            dows[i] = token["dow"]
            pad_mask[i] = True

        n_ctx = max(1, int(seq_len * self.ctx_frac))
        ctx_mask[:n_ctx] = True
        ctx_mask[seq_len:] = False
        tgt_mask[n_ctx:seq_len] = True

        return {
            "cell_ids": cell_ids,
            "hours": hours,
            "dows": dows,
            "pad_mask": pad_mask,
            "ctx_mask": ctx_mask,
            "tgt_mask": tgt_mask,
        }


def collate_trajectories(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}
