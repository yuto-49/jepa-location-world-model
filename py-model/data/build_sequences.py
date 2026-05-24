"""Build trajectory token sequences from raw check-ins.

Pipeline:
  1. Load raw check-ins
  2. Filter inactive users (< MIN_CHECKINS total)
  3. Quantize lat/lng -> geocell (0.01 deg, ~1km cells)
  4. Build MULTI-DAY trajectories (up to WINDOW_DAYS consecutive days per user)
  5. Filter short trajectories (< 2 tokens)
  6. Split users (train/val/test, no overlap)
  7. Apply k-anonymity on cells
  8. Build vocab from train only
  9. Export JSONL
"""

import json
import os
from collections import Counter
from datetime import timedelta

import pandas as pd

from data.geocell import quantize
from data.load_gowalla import load_gowalla, print_stats
from data.splits import split_users

K_ANONYMITY = 5
MIN_CHECKINS = 10  # drop users with fewer total check-ins
WINDOW_DAYS = 7    # concatenate up to 7 consecutive days into one trajectory
MAX_SEQ_LEN = 64   # cap sequence length to match model max_len
UNK = "<UNK>"
PAD = "<PAD>"


def filter_inactive_users(df: pd.DataFrame, min_checkins: int = MIN_CHECKINS) -> pd.DataFrame:
    """Drop users with fewer than min_checkins total check-ins."""
    user_counts = df.groupby("user_id").size()
    active_users = user_counts[user_counts >= min_checkins].index
    filtered = df[df["user_id"].isin(active_users)].copy()
    n_dropped = df["user_id"].nunique() - filtered["user_id"].nunique()
    print(f"  Dropped {n_dropped:,} inactive users (< {min_checkins} check-ins)")
    print(f"  Remaining: {filtered['user_id'].nunique():,} users, {len(filtered):,} check-ins")
    return filtered


def quantize_checkins(df: pd.DataFrame) -> pd.DataFrame:
    """Add geocell, hour, dow, date columns."""
    out = df.copy()
    out["geocell"] = [quantize(lat, lng) for lat, lng in zip(df["lat"], df["lng"])]
    out["hour"] = df["timestamp"].dt.hour
    out["dow"] = df["timestamp"].dt.weekday
    out["date"] = df["timestamp"].dt.date
    return out


def build_multiday_trajectories(
    df: pd.DataFrame,
    window_days: int = WINDOW_DAYS,
    max_seq_len: int = MAX_SEQ_LEN,
) -> list[dict]:
    """Build multi-day trajectory windows per user.

    Instead of one trajectory per (user, day), we slide a window of
    `window_days` consecutive days and concatenate all check-ins within
    that window into a single trajectory. This produces longer sequences
    (avg 15-20 tokens instead of 4) giving the model more context.

    Windows slide by `window_days` (non-overlapping) to avoid data leakage.
    """
    trajectories = []

    for uid, user_df in df.groupby("user_id"):
        user_df = user_df.sort_values("timestamp")
        dates = sorted(user_df["date"].unique())

        if len(dates) == 0:
            continue

        # Non-overlapping windows of window_days
        window_start_idx = 0
        while window_start_idx < len(dates):
            start_date = dates[window_start_idx]
            end_date = start_date + timedelta(days=window_days - 1)

            # Get all check-ins in this window
            window_df = user_df[
                (user_df["date"] >= start_date) & (user_df["date"] <= end_date)
            ].sort_values("timestamp")

            if len(window_df) >= 2:
                tokens = [
                    {
                        "geocell": row["geocell"],
                        "hour": int(row["hour"]),
                        "dow": int(row["dow"]),
                    }
                    for _, row in window_df.head(max_seq_len).iterrows()
                ]
                trajectories.append({
                    "user_id": uid,
                    "date": str(start_date),
                    "tokens": tokens,
                })

            # Advance to next window
            window_start_idx += 1
            while window_start_idx < len(dates) and dates[window_start_idx] <= end_date:
                window_start_idx += 1

    return trajectories


def compute_cell_user_counts(trajectories: list[dict]) -> Counter:
    """Count distinct users per cell."""
    cell_users: dict[str, set[str]] = {}
    for traj in trajectories:
        for token in traj["tokens"]:
            cell = token["geocell"]
            if cell not in cell_users:
                cell_users[cell] = set()
            cell_users[cell].add(traj["user_id"])
    return Counter({cell: len(users) for cell, users in cell_users.items()})


def apply_k_anonymity(
    trajectories: list[dict], cell_user_counts: Counter, k: int = K_ANONYMITY
) -> list[dict]:
    """Replace rare cells (< k users) with <UNK>."""
    rare_cells = {cell for cell, count in cell_user_counts.items() if count < k}
    result = []
    for traj in trajectories:
        tokens = []
        for token in traj["tokens"]:
            t = dict(token)
            if t["geocell"] in rare_cells:
                t["geocell"] = UNK
            tokens.append(t)
        result.append({**traj, "tokens": tokens})
    return result


def build_vocab(trajectories: list[dict]) -> dict[str, int]:
    """Build cell -> index mapping from train trajectories only."""
    cells = set()
    for traj in trajectories:
        for token in traj["tokens"]:
            cells.add(token["geocell"])

    vocab = {PAD: 0, UNK: 1}
    for i, cell in enumerate(sorted(cells - {UNK}), start=2):
        vocab[cell] = i
    return vocab


def export_jsonl(trajectories: list[dict], path: str) -> None:
    """Write trajectories to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for traj in trajectories:
            f.write(json.dumps(traj) + "\n")


def main():
    print("Loading Gowalla data...")
    df = load_gowalla()
    print_stats(df)

    print("\nFiltering inactive users...")
    df = filter_inactive_users(df, MIN_CHECKINS)

    print("\nQuantizing to geocells (CELL_SIZE_DEG=0.01, ~1km)...")
    df = quantize_checkins(df)

    print(f"\nBuilding multi-day trajectories (window={WINDOW_DAYS} days)...")
    all_trajs = build_multiday_trajectories(df)
    print(f"Total trajectories (>=2 tokens): {len(all_trajs):,}")

    # Trajectory length stats
    lengths = [len(t["tokens"]) for t in all_trajs]
    import numpy as np
    lengths_arr = np.array(lengths)
    print(f"Trajectory lengths: min={lengths_arr.min()}, median={np.median(lengths_arr):.0f}, "
          f"mean={lengths_arr.mean():.1f}, max={lengths_arr.max()}")

    print("\nSplitting users...")
    all_users = list({t["user_id"] for t in all_trajs})
    train_users, val_users, test_users = split_users(all_users)
    print(f"Train users: {len(train_users):,}, Val: {len(val_users):,}, Test: {len(test_users):,}")

    train_set = set(train_users)
    val_set = set(val_users)
    test_set = set(test_users)
    train_trajs = [t for t in all_trajs if t["user_id"] in train_set]
    val_trajs = [t for t in all_trajs if t["user_id"] in val_set]
    test_trajs = [t for t in all_trajs if t["user_id"] in test_set]

    print(f"\nApplying k-anonymity (k={K_ANONYMITY})...")
    cell_counts = compute_cell_user_counts(train_trajs)
    train_trajs = apply_k_anonymity(train_trajs, cell_counts)
    val_trajs = apply_k_anonymity(val_trajs, cell_counts)
    test_trajs = apply_k_anonymity(test_trajs, cell_counts)

    print("Building vocab from train split...")
    vocab = build_vocab(train_trajs)
    print(f"Vocab size: {len(vocab):,}")

    avg_len = sum(len(t["tokens"]) for t in train_trajs) / max(len(train_trajs), 1)
    print(f"Avg trajectory length (train): {avg_len:.1f}")

    out_dir = os.path.join(os.path.dirname(__file__), "processed")
    export_jsonl(train_trajs, os.path.join(out_dir, "train.jsonl"))
    export_jsonl(val_trajs, os.path.join(out_dir, "val.jsonl"))
    export_jsonl(test_trajs, os.path.join(out_dir, "test.jsonl"))

    vocab_path = os.path.join(out_dir, "vocab.json")
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    with open(vocab_path, "w") as f:
        json.dump(vocab, f)

    print(f"\nExported to {out_dir}/")
    print(f"  train.jsonl: {len(train_trajs):,} trajectories")
    print(f"  val.jsonl:   {len(val_trajs):,} trajectories")
    print(f"  test.jsonl:  {len(test_trajs):,} trajectories")
    print(f"  vocab.json:  {len(vocab):,} cells")


if __name__ == "__main__":
    main()
