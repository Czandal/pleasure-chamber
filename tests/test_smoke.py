from pvsteer.config import load_config
from pvsteer.pipeline import run_all
from pvsteer.reporting.write import verify_outputs


def test_mock_smoke_pipeline(tmp_path):
    config = load_config("configs/smoke_mock.yaml")
    config = config.model_copy(
        update={
            "output_dir": str(tmp_path / "out"),
            "dataset_path": str(tmp_path / "dataset.jsonl"),
            "manifest_path": str(tmp_path / "manifest.json"),
            "bootstrap_samples": 15,
            "permutation_samples": 8,
        }
    )
    run_all(config)
    verify_outputs(config.output_dir)
    summary = (tmp_path / "out" / "summary.json").read_text(encoding="utf-8")
    assert "Behavioral changes do not establish subjective experience." in summary
    assert "random_direction_present" in summary
