"""UMAP visualization of learned geocell embeddings."""

import json
import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def load_jepa_model(checkpoint_path, device="cpu"):
    from model.jepa import JEPAModel
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = JEPAModel(
        vocab_size=ckpt["vocab_size"], d_model=ckpt["d_model"],
        max_len=ckpt["max_len"], enc_layers=ckpt["enc_layers"],
        pred_layers=ckpt["pred_layers"], n_heads=ckpt["n_heads"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def get_cell_embeddings(model, vocab, device="cpu"):
    real_cells = {k: v for k, v in vocab.items() if k not in ("<PAD>", "<UNK>")}
    cell_names = list(real_cells.keys())
    cell_indices = torch.tensor(list(real_cells.values()), dtype=torch.long, device=device)

    max_len = model.tokenizer.pe.size(1)
    all_reprs = []

    with torch.no_grad():
        # Process in chunks of max_len to respect positional encoding size
        for start in range(0, len(cell_indices), max_len):
            chunk = cell_indices[start:start + max_len]
            cell_ids = chunk.unsqueeze(0)
            hours = torch.zeros_like(cell_ids)
            dows = torch.zeros_like(cell_ids)
            emb = model.tokenizer(cell_ids, hours, dows)
            repr_ = model.target_encoder(emb)
            repr_ = model.target_norm(repr_).squeeze(0)
            all_reprs.append(repr_.cpu().numpy())

    embeddings = np.concatenate(all_reprs, axis=0)
    return cell_names, embeddings


def main():
    with open(os.path.join(DATA_DIR, "vocab.json")) as f:
        vocab = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_jepa_model(os.path.join(CHECKPOINT_DIR, "best.pt"), device)

    cell_names, embeddings = get_cell_embeddings(model, vocab, device)
    print(f"Embedding matrix: {embeddings.shape}")

    if embeddings.shape[0] < 5:
        print("Too few cells for UMAP. Skipping.")
        return

    n_neighbors = min(15, embeddings.shape[0] - 1)
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.1)
    coords = reducer.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))
    plt.scatter(coords[:, 0], coords[:, 1], s=10, alpha=0.6)
    plt.title("UMAP of Learned Geocell Embeddings")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")

    out_path = os.path.join(CHECKPOINT_DIR, "umap_embeddings.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved UMAP plot to {out_path}")


if __name__ == "__main__":
    main()
