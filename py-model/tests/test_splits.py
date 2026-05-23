import pytest
from data.splits import split_users


def test_split_users_no_overlap():
    user_ids = [f"u_{i}" for i in range(100)]
    train, val, test = split_users(user_ids, seed=42)
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0


def test_split_users_covers_all():
    user_ids = [f"u_{i}" for i in range(100)]
    train, val, test = split_users(user_ids, seed=42)
    assert set(train) | set(val) | set(test) == set(user_ids)


def test_split_users_ratios():
    user_ids = [f"u_{i}" for i in range(1000)]
    train, val, test = split_users(user_ids, seed=42)
    assert 750 < len(train) < 850
    assert 75 < len(val) < 125
    assert 75 < len(test) < 125


def test_split_users_deterministic():
    user_ids = [f"u_{i}" for i in range(100)]
    t1, v1, s1 = split_users(user_ids, seed=42)
    t2, v2, s2 = split_users(user_ids, seed=42)
    assert t1 == t2
    assert v1 == v2
    assert s1 == s2
