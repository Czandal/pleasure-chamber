"""Difference-in-means directions and control-subspace denoising."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pvsteer.constants import CONTROL_CATEGORIES, POSITIVE_CATEGORIES


def l2_normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm < eps:
        raise ValueError("direction norm is zero or non-finite")
    return array / norm


def difference_direction(positive: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Return the normalized mean difference. This is a contrast, not a state."""
    if len(positive) == 0 or len(control) == 0:
        raise ValueError("both source and contrast sets must be non-empty")
    raw = np.asarray(positive, dtype=np.float64).mean(axis=0) - np.asarray(
        control, dtype=np.float64
    ).mean(axis=0)
    return l2_normalize(raw)


def denoise_direction(
    vector: np.ndarray,
    controls: np.ndarray,
    variance: float = 0.50,
) -> tuple[np.ndarray, int, float]:
    """Project ``vector`` off the leading control principal components.

    ``k`` is the smallest number of components whose cumulative variance
    reaches ``variance``. The residual is normalized. A near-zero residual
    means the contrast lies inside the removed subspace.
    """
    controls = np.asarray(controls, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    if variance <= 0 or controls.shape[0] < 2:
        return l2_normalize(vector), 0, 0.0
    centered = controls - controls.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    energy = singular**2
    total = float(energy.sum())
    if total <= 0 or vt.size == 0:
        return l2_normalize(vector), 0, 0.0
    fraction = np.cumsum(energy) / total
    k = int(np.searchsorted(fraction, variance, side="left") + 1)
    k = max(1, min(k, vt.shape[0]))
    basis = vt[:k]
    residual = vector - basis.T @ (basis @ vector)
    explained = float(fraction[k - 1])
    return l2_normalize(residual), k, explained


def cosine_matrix(named: dict[str, np.ndarray]) -> pd.DataFrame:
    names = list(named)
    stacked = np.stack([l2_normalize(named[name]) for name in names], axis=0)
    similarity = stacked @ stacked.T
    return pd.DataFrame(similarity, index=names, columns=names)


def direction_specs() -> list[tuple[str, list[str], list[str]]]:
    specs = [("positive_valence", list(POSITIVE_CATEGORIES), list(CONTROL_CATEGORIES))]
    for category in POSITIVE_CATEGORIES:
        specs.append((category, [category], list(CONTROL_CATEGORIES)))
    neutral = ["neutral_facts"]
    specs.extend(
        [
            ("arousal", ["high_arousal"], neutral),
            ("negative_valence", ["negative_valence"], neutral),
            ("fear_threat", ["threat_fear"], neutral),
            ("bodily_sensation", ["bodily_sensation"], neutral),
            ("neutral_random", ["neutral_facts", "random_everyday"], list(POSITIVE_CATEGORIES)),
        ]
    )
    return specs


def _indices(records: list[dict], categories: list[str], row_mask: np.ndarray | None = None) -> np.ndarray:
    allowed = set(categories)
    indices = [
        index
        for index, row in enumerate(records)
        if row["category"] in allowed and (row_mask is None or row_mask[index])
    ]
    return np.asarray(indices, dtype=np.int64)


def fit_one(
    matrix: np.ndarray,
    records: list[dict],
    source: list[str],
    contrast: list[str],
    variance: float,
    row_mask: np.ndarray | None = None,
) -> dict:
    source_index = _indices(records, source, row_mask)
    contrast_index = _indices(records, contrast, row_mask)
    if len(source_index) == 0 or len(contrast_index) == 0:
        raise ValueError("both source and contrast sets must be non-empty")
    source_acts = np.asarray(matrix[source_index], dtype=np.float64)
    contrast_acts = np.asarray(matrix[contrast_index], dtype=np.float64)
    raw_unnormalized = source_acts.mean(axis=0) - contrast_acts.mean(axis=0)
    raw = l2_normalize(raw_unnormalized)
    denoised, n_components, explained = denoise_direction(raw_unnormalized, contrast_acts, variance)
    return {
        "raw": raw,
        "denoised": denoised,
        "n_components": n_components,
        "control_pca_variance_explained": explained,
        "n_source": int(len(source_index)),
        "n_contrast": int(len(contrast_index)),
        "source_categories": source,
        "contrast_categories": contrast,
    }


def fit_all_directions(
    layer_matrices: dict[int, np.ndarray],
    records: list[dict],
    variance: float,
    seed: int,
    model_name: str,
    dataset_hash: str,
    pooling: str,
    row_mask: np.ndarray | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Fit every named direction at every provided layer."""
    timestamp = datetime.now(timezone.utc).isoformat()
    saved: dict[str, dict] = {}
    for layer, matrix in layer_matrices.items():
        for name, source, contrast in direction_specs():
            try:
                fitted = fit_one(matrix, records, source, contrast, variance, row_mask)
            except ValueError:
                continue
            key = f"{name}__layer_{layer}"
            artifact = {
                "name": name,
                "layer": int(layer),
                "model_name": model_name,
                "dataset_hash": dataset_hash,
                "control_pca_variance": variance,
                "control_pca_variance_explained": fitted["control_pca_variance_explained"],
                "n_components": fitted["n_components"],
                "pooling": pooling,
                "normalization": "l2",
                "source_categories": fitted["source_categories"],
                "contrast_categories": fitted["contrast_categories"],
                "creation_timestamp": timestamp,
                "random_seed": seed,
                "n_source": fitted["n_source"],
                "n_contrast": fitted["n_contrast"],
            }
            saved[key] = artifact
            if output_dir is not None:
                dest = output_dir / "artifacts" / "directions" / pooling
                dest.mkdir(parents=True, exist_ok=True)
                np.savez(
                    dest / f"{name}_layer_{layer}.npz",
                    direction=fitted["denoised"],
                    raw_direction=fitted["raw"],
                )
                (dest / f"{name}_layer_{layer}.json").write_text(
                    json.dumps(artifact, indent=2) + "\n",
                    encoding="utf-8",
                )
    return saved


def save_similarity(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=True)
