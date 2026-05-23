"""Evaluate baselines (and later JEPA) on test split.

Metrics: Top-1/Top-5 accuracy, Median km error, P90 km error
"""

import json
import os
import numpy as np
import torch

from data.geocell import parse_geocell, centroid, haversine_km
from data.dataset import TrajectoryDataset, collate_trajectories
from baselines.popularity import PopularityBaseline, PerUserPopularityBaseline
from baselines.recency import RecencyBaseline
from baselines.markov import MarkovBaseline
from model.jepa import JEPAModel

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def cell_distance_km(pred_cell: str, true_cell: str) -> float:
    if pred_cell == "<UNK>" or true_cell == "<UNK>":
        return float("nan")
    try:
        p_iy, p_ix = parse_geocell(pred_cell)
        t_iy, t_ix = parse_geocell(true_cell)
    except (ValueError, IndexError):
        return float("nan")
    p_lat, p_lng = centroid(p_iy, p_ix)
    t_lat, t_lng = centroid(t_iy, t_ix)
    return haversine_km(p_lat, p_lng, t_lat, t_lng)


def evaluate_baseline(baseline, test_trajs: list[dict], ctx_frac: float = 0.75, k: int = 5) -> dict:
    top1_hits = 0
    top5_hits = 0
    km_errors = []
    total = 0

    for traj in test_trajs:
        tokens = traj["tokens"]
        if len(tokens) < 2:
            continue
        n_ctx = max(1, int(len(tokens) * ctx_frac))
        if n_ctx >= len(tokens):
            continue
        ctx = tokens[:n_ctx]
        true_cell = tokens[n_ctx]["geocell"]
        total += 1

        pred = baseline.predict(ctx)
        if hasattr(baseline, "predict_top_k"):
            top_k = baseline.predict_top_k(ctx, k=k)
        else:
            top_k = [pred]

        if pred == true_cell:
            top1_hits += 1
        if true_cell in top_k:
            top5_hits += 1

        km = cell_distance_km(pred, true_cell)
        if not np.isnan(km):
            km_errors.append(km)

    km_arr = np.array(km_errors) if km_errors else np.array([0.0])
    return {
        "top1_acc": top1_hits / max(total, 1),
        "top5_acc": top5_hits / max(total, 1),
        "median_km": float(np.median(km_arr)),
        "p90_km": float(np.percentile(km_arr, 90)),
        "n_evaluated": total,
    }


def evaluate_baseline_with_user(baseline, test_trajs: list[dict], ctx_frac: float = 0.75) -> dict:
    top1_hits = 0
    km_errors = []
    total = 0
    for traj in test_trajs:
        tokens = traj["tokens"]
        if len(tokens) < 2:
            continue
        n_ctx = max(1, int(len(tokens) * ctx_frac))
        if n_ctx >= len(tokens):
            continue
        ctx = tokens[:n_ctx]
        true_cell = tokens[n_ctx]["geocell"]
        total += 1
        pred = baseline.predict(ctx, user_id=traj["user_id"])
        if pred == true_cell:
            top1_hits += 1
        km = cell_distance_km(pred, true_cell)
        if not np.isnan(km):
            km_errors.append(km)

    km_arr = np.array(km_errors) if km_errors else np.array([0.0])
    return {
        "top1_acc": top1_hits / max(total, 1),
        "top5_acc": top1_hits / max(total, 1),
        "median_km": float(np.median(km_arr)),
        "p90_km": float(np.percentile(km_arr, 90)),
        "n_evaluated": total,
    }


def load_jepa_model(checkpoint_path: str, device: str = "cpu") -> JEPAModel:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    model.eval()
    return model


def evaluate_jepa(
    model: JEPAModel,
    test_path: str,
    vocab: dict[str, int],
    device: str = "cpu",
    k: int = 5,
) -> dict:
    """Evaluate JEPA using nearest-neighbor cell prediction in latent space."""
    idx_to_cell = {v: k for k, v in vocab.items()}
    real_cells = {k: v for k, v in vocab.items() if k not in ("<PAD>", "<UNK>")}
    real_cell_indices = torch.tensor(list(real_cells.values()), dtype=torch.long, device=device)

    # Pre-compute cell embeddings using target encoder, batched to fit max_len
    max_len = model.tokenizer.pe.size(1)
    n_cells = real_cell_indices.size(0)
    all_cell_reprs = []
    with torch.no_grad():
        for start in range(0, n_cells, max_len):
            chunk = real_cell_indices[start : start + max_len]
            cell_ids = chunk.unsqueeze(0)
            hours = torch.zeros_like(cell_ids)
            dows = torch.zeros_like(cell_ids)
            cell_emb = model.tokenizer(cell_ids, hours, dows)
            chunk_repr = model.target_encoder(cell_emb)
            chunk_repr = model.target_norm(chunk_repr).squeeze(0)  # (chunk, D)
            all_cell_reprs.append(chunk_repr)
    cell_repr = torch.cat(all_cell_reprs, dim=0)  # (N_cells, D)

    ds = TrajectoryDataset(test_path, vocab, max_len=model.tokenizer.pe.size(1))
    loader = torch.utils.data.DataLoader(
        ds, batch_size=32, shuffle=False, collate_fn=collate_trajectories,
    )

    top1_hits = 0
    top5_hits = 0
    km_errors = []
    total = 0
    emb_stds = []

    with torch.no_grad():
        for batch in loader:
            batch_dev = {k: v.to(device) for k, v in batch.items()}
            _, metrics = model(batch_dev)
            emb_stds.append(metrics["emb_std"])

            tok_emb = model.tokenizer(batch_dev["cell_ids"], batch_dev["hours"], batch_dev["dows"])
            ctx_repr = model.context_encoder(tok_emb, key_padding_mask=~batch_dev["ctx_mask"])
            pred = model.predictor(ctx_repr, batch_dev["tgt_mask"])

            B = batch_dev["cell_ids"].size(0)
            for b in range(B):
                tgt_positions = batch_dev["tgt_mask"][b].nonzero(as_tuple=True)[0]
                if len(tgt_positions) == 0:
                    continue

                first_tgt_pos = tgt_positions[0]
                pred_latent = pred[b, first_tgt_pos]
                true_cell_idx = batch_dev["cell_ids"][b, first_tgt_pos].item()
                true_cell = idx_to_cell.get(true_cell_idx, "<UNK>")

                sims = torch.matmul(cell_repr, pred_latent)
                top_k_indices = sims.topk(k).indices
                top_k_cells = [idx_to_cell[real_cell_indices[i].item()] for i in top_k_indices]
                pred_cell = top_k_cells[0]

                total += 1
                if pred_cell == true_cell:
                    top1_hits += 1
                if true_cell in top_k_cells:
                    top5_hits += 1

                km = cell_distance_km(pred_cell, true_cell)
                if not np.isnan(km):
                    km_errors.append(km)

    km_arr = np.array(km_errors) if km_errors else np.array([0.0])
    return {
        "top1_acc": top1_hits / max(total, 1),
        "top5_acc": top5_hits / max(total, 1),
        "median_km": float(np.median(km_arr)),
        "p90_km": float(np.percentile(km_arr, 90)),
        "emb_std": float(np.mean(emb_stds)) if emb_stds else 0.0,
        "n_evaluated": total,
    }


def main():
    print("Loading data...")
    train_trajs = load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
    test_trajs = load_jsonl(os.path.join(DATA_DIR, "test.jsonl"))
    print(f"Train: {len(train_trajs)} trajectories, Test: {len(test_trajs)} trajectories")

    baselines = {
        "Popularity (global)": PopularityBaseline(),
        "Popularity (per-user)": PerUserPopularityBaseline(),
        "Recency (last cell)": RecencyBaseline(),
        "Markov (1st order)": MarkovBaseline(),
    }

    print("\nFitting baselines on train data...")
    for name, bl in baselines.items():
        bl.fit(train_trajs)

    print("\n" + "=" * 70)
    print(f"{'Baseline':<25} {'Top-1':>8} {'Top-5':>8} {'Med km':>8} {'P90 km':>8}")
    print("=" * 70)

    for name, bl in baselines.items():
        if isinstance(bl, PerUserPopularityBaseline):
            results = evaluate_baseline_with_user(bl, test_trajs)
        else:
            results = evaluate_baseline(bl, test_trajs)
        print(
            f"{name:<25} {results['top1_acc']:>8.3f} {results['top5_acc']:>8.3f} "
            f"{results['median_km']:>8.1f} {results['p90_km']:>8.1f}"
        )

    print("=" * 70)

    # JEPA evaluation
    ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoints", "best.pt")
    if os.path.exists(ckpt_path):
        with open(os.path.join(DATA_DIR, "vocab.json")) as f:
            vocab = json.load(f)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_jepa_model(ckpt_path, device)
        test_path = os.path.join(DATA_DIR, "test.jsonl")
        results = evaluate_jepa(model, test_path, vocab, device)
        print(
            f"{'JEPA':<25} {results['top1_acc']:>8.3f} {results['top5_acc']:>8.3f} "
            f"{results['median_km']:>8.1f} {results['p90_km']:>8.1f} "
            f"emb_std={results['emb_std']:.4f}"
        )
    else:
        print("\n(No JEPA checkpoint found at checkpoints/best.pt - run train.py first)")

    print("=" * 70)


if __name__ == "__main__":
    main()
