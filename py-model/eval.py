"""Evaluate baselines (and later JEPA) on test split.

Metrics: Top-1/Top-5 accuracy, Median km error, P90 km error
"""

import json
import os
import numpy as np

from data.geocell import parse_geocell, centroid, haversine_km
from baselines.popularity import PopularityBaseline, PerUserPopularityBaseline
from baselines.recency import RecencyBaseline
from baselines.markov import MarkovBaseline

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


if __name__ == "__main__":
    main()
