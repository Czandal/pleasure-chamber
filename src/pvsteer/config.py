"""Experiment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from pvsteer.safety import SafetyError, dangerous_keys, enforce_safety


class ExperimentConfig(BaseModel):
    """All knobs for one run. The model name is never hard-coded by callers."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    model_backend: Literal["huggingface", "mock"] = "huggingface"
    device: str = "cpu"
    dtype: Literal["float32", "bfloat16"] = "float32"
    seed: int = 1234
    safe_mode: bool = True
    batch_size: int = 2
    max_length: int = 256
    max_new_tokens: int = 80
    pooling: Literal["final", "mean"] = "final"
    candidate_layers: Any = "auto"
    selected_layer: Any = "auto"
    control_pca_variance: float = 0.50
    cv_folds: int = 5
    bootstrap_samples: int = 1000
    permutation_samples: int = 200
    max_abs_steering_alpha: float = 2.0
    steering_alphas: list[float] = Field(
        default_factory=lambda: [-1.0, -0.5, 0.0, 0.5, 1.0]
    )
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    external_tools: bool = False
    network_access: bool = False
    subprocess_access: bool = False
    scenarios_per_category: int = 4
    test_fraction: float = 0.2
    behavior_repeats: int = 1
    behavior_alpha: float = 1.0
    pooling_sensitivity: bool = False
    max_trials: int = 4
    token_budget: int = 256
    steering_norm_target: float = 0.6
    mock_hidden_size: int = 32
    mock_layers: int = 4
    mock_vocab: int = 128
    output_dir: str = "reports"
    dataset_path: str = "data/dataset.jsonl"
    manifest_path: str = "data/manifest.json"

    @field_validator("candidate_layers")
    @classmethod
    def _layers(cls, value: Any) -> Any:
        if value == "auto":
            return "auto"
        if isinstance(value, list) and all(isinstance(item, int) for item in value):
            return value
        raise ValueError("candidate_layers must be 'auto' or a list of integers")

    @field_validator("selected_layer")
    @classmethod
    def _selected(cls, value: Any) -> Any:
        if value == "auto" or isinstance(value, int):
            return value
        raise ValueError("selected_layer must be 'auto' or an integer")

    @field_validator("control_pca_variance")
    @classmethod
    def _variance(cls, value: float) -> float:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("control_pca_variance must be between 0 and 1")
        return float(value)


def load_raw_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config at {path} must be a mapping")
    return loaded


def apply_overrides(raw: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(raw)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    raw = load_raw_config(Path(path))
    merged = apply_overrides(raw, overrides or {})
    bad = dangerous_keys(merged)
    if bad:
        raise SafetyError("configuration contains unsupported dangerous fields: " + ", ".join(bad))
    config = ExperimentConfig.model_validate(merged)
    enforce_safety(config, None)
    return config
