"""First-order Markov transition baseline."""
from collections import Counter, defaultdict


class MarkovBaseline:
    """Predicts next cell as argmax P(next | current) from train transitions."""
    def __init__(self):
        self.transitions: dict[str, Counter[str]] = defaultdict(Counter)
        self.fallback: str | None = None

    def fit(self, trajectories: list[dict]) -> None:
        global_counts: Counter[str] = Counter()
        for traj in trajectories:
            tokens = traj["tokens"]
            for i in range(len(tokens) - 1):
                curr = tokens[i]["geocell"]
                nxt = tokens[i + 1]["geocell"]
                self.transitions[curr][nxt] += 1
                global_counts[curr] += 1
                global_counts[nxt] += 1
        self.fallback = global_counts.most_common(1)[0][0]

    def predict(self, context_tokens: list[dict]) -> str:
        if not context_tokens:
            return self.fallback
        curr = context_tokens[-1]["geocell"]
        if curr in self.transitions and self.transitions[curr]:
            return self.transitions[curr].most_common(1)[0][0]
        return self.fallback

    def predict_top_k(self, context_tokens: list[dict], k: int = 5) -> list[str]:
        if not context_tokens:
            return [self.fallback]
        curr = context_tokens[-1]["geocell"]
        if curr in self.transitions and self.transitions[curr]:
            return [cell for cell, _ in self.transitions[curr].most_common(k)]
        return [self.fallback]
