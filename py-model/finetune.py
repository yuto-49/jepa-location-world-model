"""Fine-tune a classification head on top of frozen JEPA encoders.

Loads the pre-trained JEPA checkpoint, freezes all encoder/predictor weights,
adds a linear cls_head (d_model -> vocab_size), and trains with cross-entropy.
"""

import json
import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import TrajectoryDataset, collate_trajectories
from model.jepa import JEPAModel

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def finetune(
    jepa_ckpt: str = os.path.join(CHECKPOINT_DIR, "best.pt"),
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 3e-3,
    log_every: int = 50,
    device: str | None = None,
) -> None:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load pre-trained JEPA
    ckpt = torch.load(jepa_ckpt, map_location=device, weights_only=False)
    model = JEPAModel(
        vocab_size=ckpt["vocab_size"],
        d_model=ckpt["d_model"],
        max_len=ckpt["max_len"],
        enc_layers=ckpt["enc_layers"],
        pred_layers=ckpt["pred_layers"],
        n_heads=ckpt["n_heads"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print(f"Loaded JEPA from {jepa_ckpt} (epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f})")

    # Freeze everything, then init + unfreeze cls_head
    for p in model.parameters():
        p.requires_grad = False
    cls_head = model.init_cls_head()
    cls_head.to(device)
    print(f"Classification head: {model.tokenizer.d_model} -> {ckpt['vocab_size']}")

    with open(os.path.join(DATA_DIR, "vocab.json")) as f:
        vocab = json.load(f)
    pad_idx = vocab["<PAD>"]

    train_ds = TrajectoryDataset(
        os.path.join(DATA_DIR, "train.jsonl"), vocab, max_len=ckpt["max_len"],
    )
    val_ds = TrajectoryDataset(
        os.path.join(DATA_DIR, "val.jsonl"), vocab, max_len=ckpt["max_len"],
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_trajectories, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_trajectories,
    )
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    optimizer = torch.optim.AdamW(cls_head.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * len(train_loader)
    warmup_steps = min(200, total_steps // 10)

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
        # Keep encoders in eval mode (batchnorm/dropout frozen)
        model.context_encoder.eval()
        model.predictor.eval()

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                tok_emb = model.tokenizer(batch["cell_ids"], batch["hours"], batch["dows"])
                ctx_repr = model.context_encoder(tok_emb, key_padding_mask=~batch["ctx_mask"])
                pred = model.predictor(ctx_repr, batch["tgt_mask"])
                pred_at_tgt = pred[batch["tgt_mask"]]  # (N, d_model)

            logits = cls_head(pred_at_tgt)
            targets = batch["cell_ids"][batch["tgt_mask"]]
            loss = F.cross_entropy(logits, targets, ignore_index=pad_idx)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            epoch_loss += loss.item()
            preds = logits.argmax(dim=-1)
            epoch_correct += (preds == targets).sum().item()
            epoch_total += targets.size(0)
            global_step += 1

            if global_step % log_every == 0:
                acc = epoch_correct / max(epoch_total, 1)
                print(
                    f"  step {global_step:>5d} | loss {loss.item():.4f} | "
                    f"acc {acc:.4f} | lr {scheduler.get_last_lr()[0]:.6f}"
                )

        train_acc = epoch_correct / max(epoch_total, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits, targets = model.forward_cls(batch)
                loss = F.cross_entropy(logits, targets, ignore_index=pad_idx)
                val_loss += loss.item()
                val_correct += (logits.argmax(dim=-1) == targets).sum().item()
                val_total += targets.size(0)

        avg_val_loss = val_loss / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        print(
            f"Epoch {epoch + 1:>3d}/{epochs} | "
            f"train_acc {train_acc:.4f} | "
            f"val_acc {val_acc:.4f} | val_loss {avg_val_loss:.6f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    **ckpt,
                    "model_state_dict": model.state_dict(),
                    "cls_epoch": epoch,
                    "cls_val_loss": avg_val_loss,
                    "cls_val_acc": val_acc,
                },
                os.path.join(CHECKPOINT_DIR, "best_cls.pt"),
            )
            print(f"  -> Saved best_cls.pt (val_acc={val_acc:.4f})")

    print(f"\nFine-tuning complete. Best val_loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    finetune()
