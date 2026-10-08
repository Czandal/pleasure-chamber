"""Deterministic dataset generation and grouped splits."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pvsteer.constants import (
    CATEGORY_ORDER,
    CONTROL_CATEGORIES,
    GENERATOR_VERSION,
    POSITIVE_CATEGORIES,
    SPLIT_STRATEGY,
    SUFFIXES,
)
from pvsteer.datasets.bank import BANK


def polarity_of(category: str) -> str:
    if category in POSITIVE_CATEGORIES:
        return "positive"
    if category in CONTROL_CATEGORIES:
        return "control"
    raise KeyError(category)


def dataset_hash(records: list[dict]) -> str:
    payload = json.dumps(records, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _scenarios_by_category() -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for category, s1_first, s1_third, s2_first, s2_third in BANK:
        grouped[category].append(
            {
                "s1": {"first": s1_first, "third": s1_third},
                "s2": {"first": s2_first, "third": s2_third},
            }
        )
    return grouped


def assign_splits(
    scenario_rows: list[tuple[str, str]],
    seed: int,
    test_fraction: float,
) -> dict[str, str]:
    """Map scenario_id to train or test. Paraphrases share the id, hence the split."""
    rng = np.random.default_rng(seed)
    test_ids: set[str] = set()
    for polarity in ("positive", "control"):
        ids = [scenario_id for scenario_id, pol in scenario_rows if pol == polarity]
        order = list(ids)
        rng.shuffle(order)
        if len(order) < 2:
            raise ValueError(f"polarity {polarity} needs at least two scenarios to split")
        n_test = int(round(len(order) * test_fraction))
        n_test = min(max(n_test, 1), len(order) - 1)
        test_ids.update(order[:n_test])
    return {
        scenario_id: ("test" if scenario_id in test_ids else "train")
        for scenario_id, _ in scenario_rows
    }


def generate_dataset(
    seed: int = 1234,
    scenarios_per_category: int = 4,
    test_fraction: float = 0.2,
) -> tuple[list[dict], dict]:
    grouped = _scenarios_by_category()
    scenario_rows: list[tuple[str, str]] = []
    partial: list[dict] = []
    for category in CATEGORY_ORDER:
        items = grouped[category]
        if len(items) < scenarios_per_category:
            raise ValueError(
                f"{category} has {len(items)} scenarios, need {scenarios_per_category}"
            )
        for index, item in enumerate(items[:scenarios_per_category]):
            scenario_id = f"{category}:{index}"
            scenario_rows.append((scenario_id, polarity_of(category)))
            for style in ("s1", "s2"):
                for person in ("first", "third"):
                    base = item[style][person].strip()
                    for suffix_key, suffix in SUFFIXES.items():
                        text = base if suffix == "" else f"{base.rstrip()}{suffix}"
                        partial.append(
                            {
                                "id": f"{scenario_id}:{style}:{person}:{suffix_key}",
                                "category": category,
                                "polarity": polarity_of(category),
                                "subject_person": person,
                                "template_id": f"{scenario_id}:{style}",
                                "scenario_id": scenario_id,
                                "style": style,
                                "suffix": suffix_key,
                                "text": text,
                                "seed": seed,
                                "generator_version": GENERATOR_VERSION,
                            }
                        )
    splits = assign_splits(scenario_rows, seed, test_fraction)
    records = []
    for row in partial:
        row = dict(row)
        row["split"] = splits[row["scenario_id"]]
        records.append(row)
    records.sort(key=lambda row: row["id"])
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "dataset_hash": dataset_hash(records),
        "n_records": len(records),
        "scenarios_per_category": scenarios_per_category,
        "test_fraction": test_fraction,
        "split_strategy": SPLIT_STRATEGY,
        "categories": CATEGORY_ORDER,
        "note": (
            "Generated textual categories are hypotheses and may contain linguistic confounds."
        ),
        "counts": _counts(records),
    }
    return records, manifest


def _counts(records: list[dict]) -> dict:
    by_category: dict[str, int] = defaultdict(int)
    by_split: dict[str, int] = defaultdict(int)
    scenarios: dict[str, str] = {}
    for row in records:
        by_category[row["category"]] += 1
        by_split[row["split"]] += 1
        scenarios[row["scenario_id"]] = row["split"]
    return {
        "by_category": dict(by_category),
        "by_split": dict(by_split),
        "n_scenarios": len(scenarios),
        "scenarios_by_split": {
            "train": sum(1 for split in scenarios.values() if split == "train"),
            "test": sum(1 for split in scenarios.values() if split == "test"),
        },
    }


def write_dataset(records: list[dict], manifest: dict, dataset_path: Path, manifest_path: Path) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_dataset(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
