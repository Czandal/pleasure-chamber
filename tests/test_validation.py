import numpy as np

from pvsteer.validation.cv import assign_folds, run_validation


def _records():
    rows = []
    for polarity, categories in (("positive", ["achievement"]), ("control", ["neutral_facts"])):
        for scenario in range(8):
            scenario_id = f"{categories[0]}:{scenario}"
            for copy in range(4):
                rows.append(
                    {
                        "id": f"{scenario_id}:{copy}",
                        "category": categories[0],
                        "polarity": polarity,
                        "subject_person": "first" if copy % 2 == 0 else "third",
                        "template_id": f"{scenario_id}:s1",
                        "scenario_id": scenario_id,
                        "style": "s1",
                        "suffix": "none",
                        "text": scenario_id,
                        "split": "test" if scenario < 2 else "train",
                        "seed": 1,
                        "generator_version": "1.0.0",
                    }
                )
    return rows


def test_folds_do_not_share_scenarios():
    records = _records()
    fold_of = assign_folds(records, n_folds=3, seed=1)
    train_ids = {row["scenario_id"] for row in records if row["split"] == "train"}
    assert set(fold_of) == train_ids
    assert len(fold_of) == len(set(fold_of))


def test_validation_blocks_leakage_and_random_baseline():
    records = _records()
    rng = np.random.default_rng(0)
    matrix = rng.normal(scale=0.05, size=(len(records), 64))
    for index, row in enumerate(records):
        if row["polarity"] == "positive":
            matrix[index, 0] += 4.0
    result = run_validation(
        {0: matrix, 1: matrix * 0.5},
        records,
        variance=0.5,
        n_folds=3,
        seed=1,
        n_boot=20,
        n_perm=15,
        mean_residual_norm={"0": 2.0, "1": 10.0},
    )
    assert result["leakage_ok"] is True
    main = result["contrasts"]["positive_vs_pooled_controls"]["auc"]
    random_auc = result["random_direction_baseline"]["auc"]
    assert main is not None and main > 0.8
    assert random_auc is not None and random_auc < main
    assert result["shuffled_label_baseline"]["auc"] is not None
    assert "effect_size" in result["contrasts"]["positive_vs_pooled_controls"]
    assert result["contrasts"]["positive_vs_pooled_controls"]["bootstrap"]["low"] is not None
