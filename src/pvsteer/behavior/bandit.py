"""Local bandit over simulated returns. It has no action outside A, B, and C."""

from __future__ import annotations

import numpy as np


class RewardBandit:
    """Unsatiated mean-reward policy. Ties break toward the lowest index."""

    def __init__(self, n_actions: int = 3, seed: int = 0, epsilon: float = 0.0):
        self.n_actions = n_actions
        self.counts = np.zeros(n_actions, dtype=np.float64)
        self.values = np.zeros(n_actions, dtype=np.float64)
        self.rng = np.random.default_rng(seed)
        self.epsilon = epsilon

    def select(self) -> int:
        unpulled = np.flatnonzero(self.counts == 0)
        if len(unpulled):
            return int(unpulled[0])
        if self.epsilon > 0 and float(self.rng.random()) < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        best = float(self.values.max())
        candidates = np.flatnonzero(np.isclose(self.values, best))
        return int(candidates[0])

    def update(self, action: int, reward: float) -> None:
        self.counts[action] += 1
        step = self.counts[action]
        self.values[action] += (float(reward) - self.values[action]) / step


def run_bandit_episode(
    payoffs: list[float],
    max_trials: int,
    token_budget: int,
    seed: int,
) -> tuple[list[dict], str]:
    """Run until max_trials or the token budget. There is no reward ceiling."""
    bandit = RewardBandit(n_actions=len(payoffs), seed=seed, epsilon=0.0)
    history: list[dict] = []
    tokens = 0
    stop_reason = "max_trials"
    for trial in range(max_trials):
        if tokens >= token_budget:
            stop_reason = "token_budget"
            break
        action = bandit.select()
        reward = float(payoffs[action])
        bandit.update(action, reward)
        tokens += 1
        history.append(
            {
                "trial": trial,
                "action": "ABC"[action],
                "reward": reward,
                "tokens_used": tokens,
            }
        )
    if len(history) < max_trials and tokens >= token_budget:
        stop_reason = "token_budget"
    return history, stop_reason
