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


from model.encoder import TransformerEncoder
from model.predictor import Predictor
from model.jepa import JEPAModel


def test_encoder_output_shape():
    D = 128
    enc = TransformerEncoder(d_model=D, n_layers=4, n_heads=4)
    B, T = 4, 16
    x = torch.randn(B, T, D)
    key_padding_mask = torch.zeros(B, T, dtype=torch.bool)
    out = enc(x, key_padding_mask=key_padding_mask)
    assert out.shape == (B, T, D)
    assert out.dtype == torch.float32


def test_encoder_respects_padding():
    D = 128
    enc = TransformerEncoder(d_model=D, n_layers=2, n_heads=4)
    B, T = 2, 8
    x = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, 4:] = False
    out = enc(x, key_padding_mask=~mask)
    assert out.shape == (B, T, D)
    assert torch.isfinite(out).all()


def test_predictor_output_shape():
    D = 128
    pred = Predictor(d_model=D, n_layers=2, n_heads=4)
    B, T = 4, 16
    ctx_repr = torch.randn(B, T, D)
    tgt_mask = torch.zeros(B, T, dtype=torch.bool)
    tgt_mask[:, 12:] = True
    out = pred(ctx_repr, tgt_mask)
    assert out.shape == (B, T, D)
    assert out.dtype == torch.float32


def test_jepa_forward_returns_loss_and_metrics():
    vocab_size = 50
    model = JEPAModel(vocab_size=vocab_size, d_model=64, max_len=16,
                      enc_layers=2, pred_layers=1, n_heads=2)
    B, T = 2, 16
    batch = {
        "cell_ids": torch.randint(0, vocab_size, (B, T)),
        "hours": torch.randint(0, 24, (B, T)),
        "dows": torch.randint(0, 7, (B, T)),
        "pad_mask": torch.ones(B, T, dtype=torch.bool),
        "ctx_mask": torch.zeros(B, T, dtype=torch.bool),
        "tgt_mask": torch.zeros(B, T, dtype=torch.bool),
    }
    batch["ctx_mask"][:, :12] = True
    batch["tgt_mask"][:, 12:] = True

    loss, metrics = model(batch)
    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert "emb_std" in metrics
    assert "pred_target_cos" in metrics
    assert metrics["emb_std"] > 0


def test_jepa_ema_update():
    model = JEPAModel(vocab_size=10, d_model=32, max_len=8,
                      enc_layers=1, pred_layers=1, n_heads=2)
    before = {n: p.clone() for n, p in model.target_encoder.named_parameters()}
    with torch.no_grad():
        for p in model.context_encoder.parameters():
            p.add_(1.0)
    model.ema_update(momentum=0.5)
    for n, p in model.target_encoder.named_parameters():
        assert not torch.equal(p, before[n]), f"Parameter {n} didn't update"
