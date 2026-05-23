"""Training loop for JEPA location world model.
Logs collapse metrics (emb_std, pred_target_cos) from step 1.
"""

import json
import os
import math
import torch
from torch.utils.data import DataLoader

from data.dataset import TrajectoryDataset, collate_trajectories
from model.jepa import JEPAModel

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def get_momentum(step: int, total_steps: int, m_start: float = 0.996, m_end: float = 0.999) -> float:
    frac = min(step / max(total_steps, 1), 1.0)
    return m_start + (m_end - m_start) * frac


def train(
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    d_model: int = 128,
    enc_layers: int = 4,
    pred_layers: int = 2,
    n_heads: int = 4,
    max_len: int = 64,
    log_every: int = 10,
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    with open(os.path.join(DATA_DIR, "vocab.json")) as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")

    train_ds = TrajectoryDataset(os.path.join(DATA_DIR, "train.jsonl"), vocab, max_len=max_len)
    val_ds = TrajectoryDataset(os.path.join(DATA_DIR, "val.jsonl"), vocab, max_len=max_len)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_trajectories, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_trajectories,
    )

    model = JEPAModel(
        vocab_size=vocab_size, d_model=d_model, max_len=max_len,
        enc_layers=enc_layers, pred_layers=pred_layers, n_heads=n_heads,
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    total_steps = epochs * len(train_loader)
    warmup_steps = min(500, total_steps // 10)

    def lr_schedule(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, metrics = model(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            momentum = get_momentum(global_step, total_steps)
            model.ema_update(momentum)

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            if global_step % log_every == 0:
                print(
                    f"  step {global_step:>5d} | loss {loss.item():.4f} | "
                    f"emb_std {metrics['emb_std']:.4f} | "
                    f"cos {metrics['pred_target_cos']:.4f} | "
                    f"mom {momentum:.4f} | lr {scheduler.get_last_lr()[0]:.6f}"
                )

        avg_train_loss = epoch_loss / max(epoch_steps, 1)

        model.eval()
        val_loss = 0.0
        val_steps = 0
        val_emb_std = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss, metrics = model(batch)
                val_loss += loss.item()
                val_emb_std += metrics["emb_std"]
                val_steps += 1

        avg_val_loss = val_loss / max(val_steps, 1)
        avg_val_std = val_emb_std / max(val_steps, 1)

        print(
            f"Epoch {epoch + 1:>3d}/{epochs} | "
            f"train_loss {avg_train_loss:.4f} | "
            f"val_loss {avg_val_loss:.4f} | "
            f"val_emb_std {avg_val_std:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "vocab_size": vocab_size,
                    "d_model": d_model,
                    "enc_layers": enc_layers,
                    "pred_layers": pred_layers,
                    "n_heads": n_heads,
                    "max_len": max_len,
                },
                os.path.join(CHECKPOINT_DIR, "best.pt"),
            )
            print(f"  -> Saved best checkpoint (val_loss={avg_val_loss:.4f})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
