"""Last-cell (recency) baseline."""
from collections import Counter


class RecencyBaseline:
    """Predicts the last cell in the context as the next cell."""
    def __init__(self):
        self.fallback: str | None = None

    def fit(self, trajectories: list[dict]) -> None:
        counts: Counter[str] = Counter()
        for traj in trajectories:
            for token in traj["tokens"]:
                counts[token["geocell"]] += 1
        self.fallback = counts.most_common(1)[0][0]

    def predict(self, context_tokens: list[dict]) -> str:
        if context_tokens:
            return context_tokens[-1]["geocell"]
        return self.fallback
