"""User-level and time-level data splits to prevent leakage."""

import random
from collections.abc import Sequence
import pandas as pd


def split_users(
    user_ids: Sequence[str],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    ids = sorted(set(user_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = ids[:n_train]
    val = ids[n_train : n_train + n_val]
    test = ids[n_train + n_val :]

    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0
    assert len(train) + len(val) + len(test) == n

    return train, val, test


def time_split_train(
    df: pd.DataFrame,
    date_col: str = "date",
    temporal_val_frac: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(df[date_col].unique())
    n_val = max(1, int(len(dates) * temporal_val_frac))
    cutoff_dates = set(dates[-n_val:])

    temporal_val = df[df[date_col].isin(cutoff_dates)]
    train_time = df[~df[date_col].isin(cutoff_dates)]

    return train_time, temporal_val
