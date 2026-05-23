"""Global and per-user popularity baselines."""
from collections import Counter


class PopularityBaseline:
    """Predicts the globally most-frequent cell."""
    def __init__(self):
        self.most_common: str | None = None

    def fit(self, trajectories: list[dict]) -> None:
        counts: Counter[str] = Counter()
        for traj in trajectories:
            for token in traj["tokens"]:
                counts[token["geocell"]] += 1
        self.most_common = counts.most_common(1)[0][0]

    def predict(self, context_tokens: list[dict]) -> str:
        return self.most_common


class PerUserPopularityBaseline:
    """Predicts a user's most-frequent cell, falls back to global."""
    def __init__(self):
        self.user_top: dict[str, str] = {}
        self.global_top: str | None = None

    def fit(self, trajectories: list[dict]) -> None:
        global_counts: Counter[str] = Counter()
        user_counts: dict[str, Counter[str]] = {}
        for traj in trajectories:
            uid = traj["user_id"]
            if uid not in user_counts:
                user_counts[uid] = Counter()
            for token in traj["tokens"]:
                cell = token["geocell"]
                global_counts[cell] += 1
                user_counts[uid][cell] += 1
        self.global_top = global_counts.most_common(1)[0][0]
        self.user_top = {uid: counts.most_common(1)[0][0] for uid, counts in user_counts.items()}

    def predict(self, context_tokens: list[dict], user_id: str | None = None) -> str:
        if user_id and user_id in self.user_top:
            return self.user_top[user_id]
        return self.global_top
