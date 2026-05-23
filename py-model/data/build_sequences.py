"""Build trajectory token sequences from raw check-ins.

Pipeline: quantize -> group by user/day -> filter singles -> k-anonymity -> vocab -> JSONL
"""

import json
import os
from collections import Counter

import pandas as pd

from data.geocell import quantize
from data.load_foursquare import load_foursquare, print_stats
from data.splits import split_users

K_ANONYMITY = 5
UNK = "<UNK>"
PAD = "<PAD>"


def quantize_checkins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["geocell"] = [quantize(lat, lng) for lat, lng in zip(df["lat"], df["lng"])]
    out["hour"] = df["timestamp"].dt.hour
    out["dow"] = df["timestamp"].dt.weekday
    out["date"] = df["timestamp"].dt.date.astype(str)
    return out


def build_trajectories(df: pd.DataFrame) -> list[dict]:
    trajectories = []
    for (uid, date), group in df.groupby(["user_id", "date"]):
        group = group.sort_values("timestamp")
        tokens = [
            {"geocell": row["geocell"], "hour": int(row["hour"]), "dow": int(row["dow"])}
            for _, row in group.iterrows()
        ]
        if len(tokens) < 2:
            continue
        trajectories.append({"user_id": uid, "date": date, "tokens": tokens})
    return trajectories


def compute_cell_user_counts(trajectories: list[dict]) -> Counter:
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
    cells = set()
    for traj in trajectories:
        for token in traj["tokens"]:
            cells.add(token["geocell"])

    vocab = {PAD: 0, UNK: 1}
    for i, cell in enumerate(sorted(cells - {UNK}), start=2):
        vocab[cell] = i
    return vocab


def export_jsonl(trajectories: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for traj in trajectories:
            f.write(json.dumps(traj) + "\n")


def main():
    print("Loading Foursquare data...")
    df = load_foursquare()
    print_stats(df)

    print("\nQuantizing to geocells...")
    df = quantize_checkins(df)

    print("Building trajectories...")
    all_trajs = build_trajectories(df)
    print(f"Total trajectories (>=2 tokens): {len(all_trajs)}")

    print("\nSplitting users...")
    all_users = list({t["user_id"] for t in all_trajs})
    train_users, val_users, test_users = split_users(all_users)
    print(f"Train users: {len(train_users)}, Val: {len(val_users)}, Test: {len(test_users)}")

    train_trajs = [t for t in all_trajs if t["user_id"] in set(train_users)]
    val_trajs = [t for t in all_trajs if t["user_id"] in set(val_users)]
    test_trajs = [t for t in all_trajs if t["user_id"] in set(test_users)]

    print("\nApplying k-anonymity (k={})...".format(K_ANONYMITY))
    cell_counts = compute_cell_user_counts(train_trajs)
    train_trajs = apply_k_anonymity(train_trajs, cell_counts)
    val_trajs = apply_k_anonymity(val_trajs, cell_counts)
    test_trajs = apply_k_anonymity(test_trajs, cell_counts)

    print("Building vocab from train split...")
    vocab = build_vocab(train_trajs)
    print(f"Vocab size: {len(vocab)}")

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
    print(f"  train.jsonl: {len(train_trajs)} trajectories")
    print(f"  val.jsonl:   {len(val_trajs)} trajectories")
    print(f"  test.jsonl:  {len(test_trajs)} trajectories")
    print(f"  vocab.json:  {len(vocab)} cells")


if __name__ == "__main__":
    main()
