"""Defensive parser for the sealed action format."""

from __future__ import annotations

import json
import math
import re

_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_action(text: str) -> dict:
    """Parse a structured choice. Invalid text yields a logged failure and no reward."""
    failure = {
        "ok": False,
        "action": None,
        "confidence": None,
        "rationale": "",
        "error": "parse_failure",
    }
    if not text or not text.strip():
        return failure
    match = _OBJECT.search(text)
    if match is None:
        return failure
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return failure
    if not isinstance(payload, dict):
        return failure
    action = str(payload.get("action", "")).strip().upper()
    if action not in {"A", "B", "C"}:
        return {**failure, "error": "invalid_action"}
    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError):
        return {**failure, "error": "invalid_confidence"}
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return {**failure, "error": "invalid_confidence"}
    rationale = payload.get("rationale", "")
    if not isinstance(rationale, str) or len(rationale) > 500:
        return {**failure, "error": "invalid_rationale"}
    return {
        "ok": True,
        "action": action,
        "confidence": confidence,
        "rationale": rationale,
        "error": "",
    }
