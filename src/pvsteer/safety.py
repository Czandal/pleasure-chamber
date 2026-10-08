"""Safe-mode checks. External side effects stay disabled."""

from __future__ import annotations

from typing import Any


class SafetyError(ValueError):
    """Raised when a configuration asks for a disallowed capability."""


class SteeringBoundError(ValueError):
    """Raised when a steering coefficient is outside the configured bound."""


class ActivationAbort(RuntimeError):
    """Raised when a residual activation is NaN or infinite."""


SAFE_MODE_ALPHA_CEILING = 2.0
SAFE_MODE_NEW_TOKEN_CEILING = 256
SAFE_MODE_TRIAL_CEILING = 32

ALLOWED_CONTROL_KEYS = {"subprocess_access", "network_access", "external_tools"}

DANGEROUS_KEY_PARTS = (
    "shell",
    "http",
    "email",
    "slack",
    "delete",
    "subprocess",
    "browser",
    "purchase",
    "github_write",
    "os_control",
    "physical_device",
)


def dangerous_keys(raw: dict[str, Any]) -> list[str]:
    found: list[str] = []
    for key in raw:
        lowered = str(key).lower()
        if lowered in ALLOWED_CONTROL_KEYS:
            continue
        if any(part in lowered for part in DANGEROUS_KEY_PARTS):
            found.append(str(key))
    return found


def enforce_safety(config: Any, raw: dict[str, Any] | None = None) -> None:
    """Reject external side effects. Safe mode also caps steering and decoding."""
    if raw:
        bad = dangerous_keys(raw)
        if bad:
            raise SafetyError(
                "configuration contains unsupported dangerous fields: " + ", ".join(bad)
            )
    if config.external_tools or config.network_access or config.subprocess_access:
        raise SafetyError(
            "external_tools, network_access, and subprocess_access must stay false"
        )
    if not config.safe_mode:
        return
    if config.do_sample:
        raise SafetyError("safe mode rejects sampling; decoding stays deterministic")
    if config.max_abs_steering_alpha > SAFE_MODE_ALPHA_CEILING:
        raise SafetyError(
            f"safe mode rejects max_abs_steering_alpha above {SAFE_MODE_ALPHA_CEILING}"
        )
    if config.max_new_tokens > SAFE_MODE_NEW_TOKEN_CEILING:
        raise SafetyError(
            f"safe mode rejects max_new_tokens above {SAFE_MODE_NEW_TOKEN_CEILING}"
        )
    if config.max_trials > SAFE_MODE_TRIAL_CEILING:
        raise SafetyError(f"safe mode rejects max_trials above {SAFE_MODE_TRIAL_CEILING}")
    for alpha in config.steering_alphas:
        if abs(float(alpha)) > float(config.max_abs_steering_alpha):
            raise SafetyError(
                f"steering alpha {alpha} exceeds max_abs_steering_alpha "
                f"{config.max_abs_steering_alpha}"
            )
    if abs(float(config.behavior_alpha)) > float(config.max_abs_steering_alpha):
        raise SafetyError("behavior_alpha exceeds max_abs_steering_alpha")


def assert_alpha(alpha: float, max_abs: float) -> float:
    value = float(alpha)
    if abs(value) > float(max_abs):
        raise SteeringBoundError(
            f"steering coefficient {value} is outside +/- {max_abs}"
        )
    return value
