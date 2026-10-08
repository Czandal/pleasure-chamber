from pvsteer.constants import BEHAVIOR_WARNING, EXPERIENCE_DISCLAIMER
from pvsteer.reporting.write import audit_claims, render_markdown


def test_report_contains_disclaimer_and_rejects_affirmative_claims():
    text = render_markdown(
        {
            "model_name": "mock",
            "model_revision": "mock-0",
            "dataset_hash": "abc",
            "seed": 1,
            "generator_version": "1.0.0",
            "split_strategy": "grouped by scenario_id",
            "pooling": "final",
            "layers": [0],
            "control_pca_variance": 0.5,
            "steering_alphas": [0.0, 1.0],
            "git_commit": None,
            "python": "3",
        },
        None,
        None,
    )
    assert BEHAVIOR_WARNING in text
    assert EXPERIENCE_DISCLAIMER in text
    assert audit_claims(text) == []
    assert audit_claims("the model is sentient") == ["the model is sentient"]
