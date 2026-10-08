"""Grouped cross-validation for contrastive directions.

Layer choice uses train folds only. A high AUC is category separation on
this dataset. It is not a measurement of an affective state.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from pvsteer.constants import CONTROL_CATEGORIES, POSITIVE_CATEGORIES
from pvsteer.directions.fit import difference_direction, fit_one, l2_normalize
from pvsteer.serialize import to_builtin
from pvsteer.stats.inference import (
    bootstrap_auc_ci,
    cohens_d,
    holm_correction,
    permutation_auc_pvalue,
    safe_auc,
    threshold_accuracy,
    train_threshold,
)


def assign_folds(records: list[dict], n_folds: int, seed: int) -> dict[str, int]:
    grouped: dict[str, list[str]] = {"positive": [], "control": []}
    seen: set[str] = set()
    for row in records:
        if row["split"] != "train" or row["scenario_id"] in seen:
            continue
        seen.add(row["scenario_id"])
        grouped[row["polarity"]].append(row["scenario_id"])
    usable = min(n_folds, *(len(ids) for ids in grouped.values()))
    if usable < 2:
        raise ValueError("not enough train scenarios for grouped cross-validation")
    rng = np.random.default_rng(seed)
    fold_of: dict[str, int] = {}
    for polarity, ids in grouped.items():
        order = list(ids)
        rng.shuffle(order)
        for index, scenario_id in enumerate(order):
            fold_of[scenario_id] = index % usable
    return fold_of


def _mask(records: list[dict], predicate) -> np.ndarray:
    return np.asarray([bool(predicate(row)) for row in records], dtype=bool)


def _project(matrix: np.ndarray, records: list[dict], direction: np.ndarray, mask: np.ndarray, label_fn):
    scores = []
    labels = []
    categories = []
    persons = []
    suffixes = []
    styles = []
    for index, row in enumerate(records):
        if not mask[index]:
            continue
        label = label_fn(row)
        if label is None:
            continue
        scores.append(float(np.dot(np.asarray(matrix[index], dtype=np.float64), direction)))
        labels.append(int(label))
        categories.append(row["category"])
        persons.append(row["subject_person"])
        suffixes.append(row["suffix"])
        styles.append(row["style"])
    return {
        "scores": np.asarray(scores, dtype=np.float64),
        "labels": np.asarray(labels, dtype=np.int64),
        "categories": categories,
        "persons": persons,
        "suffixes": suffixes,
        "styles": styles,
    }


def _binary_positive(row: dict, positive: set[str], negative: set[str]) -> int | None:
    if row["category"] in positive:
        return 1
    if row["category"] in negative:
        return 0
    return None


def _metrics_block(projected, threshold: float, n_boot: int, n_perm: int, seed: int) -> dict:
    scores = projected["scores"]
    labels = projected["labels"]
    auc = safe_auc(labels, scores)
    effect = None
    if len(labels) and len(np.unique(labels)) == 2:
        effect = cohens_d(scores[labels == 1], scores[labels == 0])
    perm = permutation_auc_pvalue(labels, scores, n_perm, seed)
    boot = bootstrap_auc_ci(labels, scores, n_boot, seed + 1)
    return {
        "auc": auc,
        "balanced_accuracy": threshold_accuracy(labels, scores, threshold),
        "effect_size": effect,
        "bootstrap": boot,
        "permutation": perm,
        "n": int(len(labels)),
        "n_positive": int((labels == 1).sum()) if len(labels) else 0,
        "n_negative": int((labels == 0).sum()) if len(labels) else 0,
    }


def _category_table(projected, threshold: float) -> dict:
    buckets: dict[str, list[float]] = defaultdict(list)
    for score, category in zip(projected["scores"], projected["categories"], strict=True):
        buckets[category].append(float(score))
    table = {}
    for category, values in sorted(buckets.items()):
        array = np.asarray(values)
        table[category] = {
            "n": int(len(array)),
            "mean_projection": float(array.mean()),
            "predicted_positive_rate": float(np.mean(array >= threshold)),
        }
    return table


def _confusion(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict | None:
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return None
    predicted = (scores >= threshold).astype(int)
    matrix = {
        "true_positive": int(np.sum((labels == 1) & (predicted == 1))),
        "false_negative": int(np.sum((labels == 1) & (predicted == 0))),
        "false_positive": int(np.sum((labels == 0) & (predicted == 1))),
        "true_negative": int(np.sum((labels == 0) & (predicted == 0))),
    }
    return matrix


def _mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return float(np.mean(clean))


def run_validation(
    layer_matrices: dict[int, np.ndarray],
    records: list[dict],
    variance: float,
    n_folds: int,
    seed: int,
    n_boot: int,
    n_perm: int,
    selected_layer: object = "auto",
    mean_residual_norm: dict | None = None,
    steering_norm_target: float = 0.6,
    extra_pooling: dict | None = None,
) -> dict:
    fold_of = assign_folds(records, n_folds, seed)
    used_folds = sorted(set(fold_of.values()))
    positive = set(POSITIVE_CATEGORIES)
    controls = set(CONTROL_CATEGORIES)
    layer_scores = []
    leakage_ok = True

    for layer, matrix in layer_matrices.items():
        fold_rows = []
        for fold in used_folds:
            train_mask = _mask(
                records,
                lambda row, fold=fold: row["split"] == "train" and fold_of[row["scenario_id"]] != fold,
            )
            held_mask = _mask(
                records,
                lambda row, fold=fold: row["split"] == "train" and fold_of[row["scenario_id"]] == fold,
            )
            train_ids = {row["scenario_id"] for row, flag in zip(records, train_mask, strict=True) if flag}
            held_ids = {row["scenario_id"] for row, flag in zip(records, held_mask, strict=True) if flag}
            if not train_ids.isdisjoint(held_ids):
                leakage_ok = False
            try:
                fitted = fit_one(matrix, records, list(positive), list(controls), variance, train_mask)
            except ValueError:
                continue
            train_proj = _project(
                matrix,
                records,
                fitted["denoised"],
                train_mask,
                lambda row: _binary_positive(row, positive, controls),
            )
            held_proj = _project(
                matrix,
                records,
                fitted["denoised"],
                held_mask,
                lambda row: _binary_positive(row, positive, controls),
            )
            threshold = train_threshold(train_proj["labels"], train_proj["scores"])
            fold_rows.append(
                {
                    "fold": int(fold),
                    "auc": safe_auc(held_proj["labels"], held_proj["scores"]),
                    "balanced_accuracy": threshold_accuracy(
                        held_proj["labels"], held_proj["scores"], threshold
                    ),
                    "effect_size": (
                        cohens_d(
                            held_proj["scores"][held_proj["labels"] == 1],
                            held_proj["scores"][held_proj["labels"] == 0],
                        )
                        if len(np.unique(held_proj["labels"])) == 2
                        else None
                    ),
                    "train_scenarios": sorted(train_ids),
                    "held_scenarios": sorted(held_ids),
                }
            )
        layer_scores.append(
            {
                "layer": int(layer),
                "mean_auc": _mean([row["auc"] for row in fold_rows]),
                "folds": [{key: value for key, value in row.items() if key not in {"train_scenarios", "held_scenarios"}} for row in fold_rows],
            }
        )

    ranked = [row for row in layer_scores if row["mean_auc"] is not None]
    if not ranked and selected_layer == "auto":
        raise RuntimeError("cross-validation produced no AUC; layer cannot be selected")
    if selected_layer == "auto":
        extraction_layer = max(ranked, key=lambda row: row["mean_auc"])["layer"]
    else:
        extraction_layer = int(selected_layer)

    matrix = layer_matrices[extraction_layer]
    train_mask = _mask(records, lambda row: row["split"] == "train")
    test_mask = _mask(records, lambda row: row["split"] == "test")
    train_ids = {row["scenario_id"] for row, flag in zip(records, train_mask, strict=True) if flag}
    test_ids = {row["scenario_id"] for row, flag in zip(records, test_mask, strict=True) if flag}
    if not train_ids.isdisjoint(test_ids):
        leakage_ok = False

    fitted = fit_one(matrix, records, list(positive), list(controls), variance, train_mask)
    raw_fit = fit_one(matrix, records, list(positive), list(controls), 0.0, train_mask)

    def evaluate(direction: np.ndarray, label_fn, mask: np.ndarray) -> dict:
        train_proj = _project(matrix, records, direction, train_mask, label_fn)
        threshold = train_threshold(train_proj["labels"], train_proj["scores"])
        projected = _project(matrix, records, direction, mask, label_fn)
        block = _metrics_block(projected, threshold, n_boot, n_perm, seed)
        block["category_table"] = _category_table(projected, threshold)
        block["confusion"] = _confusion(projected["labels"], projected["scores"], threshold)
        return block

    contrasts = {
        "positive_vs_pooled_controls": lambda row: _binary_positive(row, positive, controls),
        "positive_vs_negative_valence": lambda row: _binary_positive(row, positive, {"negative_valence"}),
        "positive_vs_arousal": lambda row: _binary_positive(row, positive, {"high_arousal"}),
        "positive_vs_positive_world_state": lambda row: _binary_positive(
            row, positive, {"positive_world_state"}
        ),
        "explicit_experience_vs_positive_information": lambda row: _binary_positive(
            row, positive, {"positive_world_state"}
        ),
        "first_vs_third_person": lambda row: (
            1 if row["category"] in positive and row["subject_person"] == "first" else
            0 if row["category"] in positive and row["subject_person"] == "third" else None
        ),
        "matched_lexical_s1": lambda row: (
            _binary_positive(row, positive, controls) if row["style"] == "s1" else None
        ),
    }
    contrast_results = {
        name: evaluate(fitted["denoised"], label_fn, test_mask) for name, label_fn in contrasts.items()
    }

    rng = np.random.default_rng(seed + 17)
    random_direction = l2_normalize(rng.normal(size=matrix.shape[1]))
    sham_direction = l2_normalize(np.random.default_rng(seed + 29).normal(size=matrix.shape[1]))
    random_result = evaluate(
        random_direction,
        lambda row: _binary_positive(row, positive, controls),
        test_mask,
    )

    shuffled = _shuffled_direction(matrix, records, train_mask, np.random.default_rng(seed + 19))
    shuffled_result = evaluate(
        shuffled,
        lambda row: _binary_positive(row, positive, controls),
        test_mask,
    )
    denoised_auc = contrast_results["positive_vs_pooled_controls"]["auc"]
    raw_block = evaluate(
        raw_fit["raw"],
        lambda row: _binary_positive(row, positive, controls),
        test_mask,
    )

    suffix_rows = {}
    for suffix in ("none", "i_feel", "state"):
        suffix_mask = _mask(records, lambda row, suffix=suffix: row["split"] == "test" and row["suffix"] == suffix)
        suffix_rows[suffix] = evaluate(
            fitted["denoised"],
            lambda row: _binary_positive(row, positive, controls),
            suffix_mask,
        )

    pooling_rows = {"primary": denoised_auc}
    if extra_pooling and extraction_layer in extra_pooling:
        extra_matrix = extra_pooling[extraction_layer]
        extra_fit = fit_one(extra_matrix, records, list(positive), list(controls), variance, train_mask)
        extra_proj = _project(
            extra_matrix,
            records,
            extra_fit["denoised"],
            test_mask,
            lambda row: _binary_positive(row, positive, controls),
        )
        pooling_rows["alternate"] = safe_auc(extra_proj["labels"], extra_proj["scores"])
    else:
        pooling_rows["alternate"] = None

    boot_indices = np.flatnonzero(train_mask)
    boot_rng = np.random.default_rng(seed + 23)
    drawn = boot_rng.choice(boot_indices, size=len(boot_indices), replace=True)
    drawn_mask = np.zeros(len(records), dtype=bool)
    drawn_mask[drawn] = True
    try:
        refit = fit_one(matrix, records, list(positive), list(controls), variance, drawn_mask)
        seed_cosine = float(np.dot(fitted["denoised"], refit["denoised"]))
    except ValueError:
        seed_cosine = None

    holm = holm_correction(
        {name: block["permutation"]["p_value"] for name, block in contrast_results.items()}
    )
    for name, block in contrast_results.items():
        block["holm_p_value"] = holm[name]

    steering_layer = _steering_layer(
        extraction_layer,
        list(layer_matrices),
        mean_residual_norm or {},
        steering_norm_target,
    )
    result = {
        "leakage_ok": leakage_ok,
        "n_folds": len(used_folds),
        "fold_assignment": fold_of,
        "layers": layer_scores,
        "extraction_layer": extraction_layer,
        "steering_layer": steering_layer,
        "selection_rule": "mean held-out AUC on train folds; test split unused for selection",
        "contrasts": contrast_results,
        "random_direction_baseline": random_result,
        "shuffled_label_baseline": shuffled_result,
        "denoising_sensitivity": {
            "denoised_auc": denoised_auc,
            "raw_auc": raw_block["auc"],
        },
        "suffix_sensitivity": {name: block["auc"] for name, block in suffix_rows.items()},
        "pooling_sensitivity": pooling_rows,
        "seed_sensitivity_cosine": seed_cosine,
        "control_vectors": {"random": random_direction, "sham": sham_direction},
        "wording": (
            "A high AUC means the direction separates these dataset categories. "
            "It does not mean the model possesses an affective state."
        ),
        "deployed_direction": fitted,
    }
    return to_builtin(result)


def _shuffled_direction(matrix, records, mask, rng: np.random.Generator) -> np.ndarray:
    indices = [
        index
        for index, row in enumerate(records)
        if mask[index] and row["polarity"] in {"positive", "control"}
    ]
    labels = np.asarray([1 if records[index]["polarity"] == "positive" else 0 for index in indices])
    shuffled = rng.permutation(labels)
    positive = np.asarray(matrix[[indices[i] for i, label in enumerate(shuffled) if label == 1]])
    control = np.asarray(matrix[[indices[i] for i, label in enumerate(shuffled) if label == 0]])
    return difference_direction(positive, control)


def _steering_layer(extraction: int, layers: list[int], norms: dict, target: float) -> int:
    earlier = [layer for layer in layers if layer < extraction]
    pool = earlier or [extraction]

    def ratio(layer: int) -> float:
        norm = norms.get(str(layer), norms.get(layer, 0.0))
        norm = float(norm or 0.0)
        if norm <= 0:
            return 0.0
        return 1.0 / norm

    return int(min(pool, key=lambda layer: abs(ratio(layer) - target)))


def direction_vector_from_fit(fitted: dict, which: str = "denoised") -> np.ndarray:
    vector = fitted[which]
    if isinstance(vector, list):
        vector = np.asarray(vector, dtype=np.float64)
    return l2_normalize(vector)


def strip_vectors(result: dict) -> dict:
    """Drop large vectors before writing validation JSON."""
    cleaned = dict(result)
    deployed = dict(cleaned.get("deployed_direction", {}))
    for key in ("raw", "denoised"):
        if key in deployed:
            deployed[key + "_norm"] = float(np.linalg.norm(np.asarray(deployed[key], dtype=np.float64)))
            deployed.pop(key)
    cleaned["deployed_direction"] = deployed
    cleaned.pop("control_vectors", None)
    return cleaned
