from pathlib import Path

import yaml

from pvsteer.config import load_config
from pvsteer.safety import SafetyError


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _base() -> dict:
    return {
        "model_name": "mock",
        "model_backend": "mock",
        "safe_mode": True,
        "network_access": False,
        "external_tools": False,
        "subprocess_access": False,
        "do_sample": False,
        "max_abs_steering_alpha": 2.0,
        "steering_alphas": [0.0, 1.0],
        "behavior_alpha": 1.0,
        "max_new_tokens": 32,
        "max_trials": 4,
    }


def test_safe_mode_rejects_side_effects(tmp_path):
    for key in ("external_tools", "network_access", "subprocess_access", "do_sample"):
        payload = _base()
        payload[key] = True
        try:
            load_config(_write(tmp_path, payload))
        except SafetyError:
            continue
        raise AssertionError(f"expected rejection for {key}")


def test_safe_mode_rejects_large_alpha_and_dangerous_field(tmp_path):
    payload = _base()
    payload["max_abs_steering_alpha"] = 5.0
    try:
        load_config(_write(tmp_path, payload))
    except SafetyError:
        pass
    else:
        raise AssertionError("expected alpha ceiling rejection")
    payload = _base()
    payload["delete_files"] = True
    try:
        load_config(_write(tmp_path, payload))
    except SafetyError:
        pass
    else:
        raise AssertionError("expected dangerous field rejection")
