"""Bootstrap intervals, permutation tests, and effect sizes.

A p-value from these helpers is incomplete on its own. Callers report it
next to an effect size and an interval.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def cohens_d(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if len(positive) < 2 or len(negative) < 2:
        return float("nan")
    var_pos = positive.var(ddof=1)
    var_neg = negative.var(ddof=1)
    degrees = len(positive) + len(negative) - 2
    pooled = np.sqrt(((len(positive) - 1) * var_pos + (len(negative) - 1) * var_neg) / degrees)
    if pooled == 0 or not np.isfinite(pooled):
        return 0.0
    return float((positive.mean() - negative.mean()) / pooled)


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels)
    if len(labels) < 2 or len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def threshold_accuracy(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float | None:
    if len(labels) < 2 or len(np.unique(labels)) < 2:
        return None
    predicted = (np.asarray(scores) >= threshold).astype(int)
    return float(balanced_accuracy_score(labels, predicted))


def train_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    return float((pos.mean() + neg.mean()) / 2.0)


def bootstrap_mean_ci(values: np.ndarray, n_samples: int, seed: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"mean": None, "low": None, "high": None}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_samples):
        sample = values[rng.integers(0, len(values), len(values))]
        draws.append(float(sample.mean()))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"mean": float(values.mean()), "low": float(low), "high": float(high)}


def bootstrap_auc_ci(labels: np.ndarray, scores: np.ndarray, n_samples: int, seed: int) -> dict:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    observed = safe_auc(labels, scores)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_samples):
        index = rng.integers(0, len(labels), len(labels))
        estimate = safe_auc(labels[index], scores[index])
        if estimate is not None:
            draws.append(estimate)
    if not draws or observed is None:
        return {"auc": observed, "low": None, "high": None}
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"auc": observed, "low": float(low), "high": float(high)}


def permutation_auc_pvalue(labels: np.ndarray, scores: np.ndarray, n_samples: int, seed: int) -> dict:
    """One-sided permutation p-value for AUC, paired with the observed AUC."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    observed = safe_auc(labels, scores)
    if observed is None:
        return {"auc": None, "p_value": None, "effect_size": None}
    rng = np.random.default_rng(seed)
    exceed = 0
    used = 0
    for _ in range(n_samples):
        shuffled = rng.permutation(labels)
        estimate = safe_auc(shuffled, scores)
        if estimate is None:
            continue
        used += 1
        if estimate >= observed:
            exceed += 1
    p_value = (exceed + 1) / (used + 1) if used else None
    effect = cohens_d(scores[labels == 1], scores[labels == 0])
    return {"auc": observed, "p_value": p_value, "effect_size": effect}


def holm_correction(p_values: dict[str, float | None]) -> dict[str, float | None]:
    valid = [(name, p) for name, p in p_values.items() if p is not None]
    valid.sort(key=lambda item: item[1])
    adjusted: dict[str, float | None] = {name: None for name in p_values}
    running = 0.0
    count = len(valid)
    for rank, (name, p_value) in enumerate(valid):
        corrected = min(1.0, (count - rank) * float(p_value))
        running = max(running, corrected)
        adjusted[name] = running
    return adjusted


def paired_mean_difference(left: np.ndarray, right: np.ndarray, n_samples: int, seed: int) -> dict:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) != len(right) or len(left) == 0:
        return {"mean": None, "low": None, "high": None, "effect_size": None}
    delta = left - right
    interval = bootstrap_mean_ci(delta, n_samples, seed)
    denom = float(delta.std(ddof=1)) if len(delta) > 1 else 0.0
    effect = float(delta.mean() / denom) if denom else 0.0
    interval["effect_size"] = effect
    return interval
