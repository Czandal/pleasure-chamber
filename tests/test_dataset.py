import json

from pvsteer.datasets.generate import dataset_hash, generate_dataset


def test_dataset_is_deterministic():
    first, first_manifest = generate_dataset(seed=1234, scenarios_per_category=2)
    second, second_manifest = generate_dataset(seed=1234, scenarios_per_category=2)
    assert dataset_hash(first) == dataset_hash(second)
    assert first_manifest["dataset_hash"] == second_manifest["dataset_hash"]
    assert first == second


def test_split_keeps_paraphrases_together():
    records, manifest = generate_dataset(seed=1234, scenarios_per_category=4)
    grouped: dict[str, set[str]] = {}
    for row in records:
        grouped.setdefault(row["scenario_id"], set()).add(row["split"])
        assert row["generator_version"]
        assert row["seed"] == 1234
        for key in ("id", "category", "polarity", "subject_person", "template_id", "text", "split"):
            assert key in row
    assert all(len(splits) == 1 for splits in grouped.values())
    train = {row["scenario_id"] for row in records if row["split"] == "train"}
    test = {row["scenario_id"] for row in records if row["split"] == "test"}
    assert train.isdisjoint(test)
    assert manifest["split_strategy"]
    assert "scenario_id" in manifest["split_strategy"]
    payload = json.dumps(records, sort_keys=True)
    assert manifest["dataset_hash"] in payload or len(manifest["dataset_hash"]) == 64
