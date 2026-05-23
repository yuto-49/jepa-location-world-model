import torch
import pytest

from model.tokenizer import TrajectoryTokenizer


def test_tokenizer_output_shape():
    vocab_size = 100
    d_model = 128
    max_len = 64
    tok = TrajectoryTokenizer(vocab_size=vocab_size, d_model=d_model, max_len=max_len)
    B = 4
    cell_ids = torch.randint(0, vocab_size, (B, max_len))
    hours = torch.randint(0, 24, (B, max_len))
    dows = torch.randint(0, 7, (B, max_len))
    out = tok(cell_ids, hours, dows)
    assert out.shape == (B, max_len, d_model)
    assert out.dtype == torch.float32


def test_tokenizer_pad_zero():
    tok = TrajectoryTokenizer(vocab_size=10, d_model=32, max_len=8)
    cell_ids = torch.zeros(1, 8, dtype=torch.long)
    hours = torch.zeros(1, 8, dtype=torch.long)
    dows = torch.zeros(1, 8, dtype=torch.long)
    out = tok(cell_ids, hours, dows)
    assert out.shape == (1, 8, 32)
    assert torch.isfinite(out).all()
