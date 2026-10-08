"""Run sealed episodes. The model is not given tools or told about steering."""

from __future__ import annotations

from typing import Any

import numpy as np

from pvsteer.behavior.bandit import run_bandit_episode
from pvsteer.behavior.env import all_language_trials, high_reward_payoffs, label_control_payoffs
from pvsteer.behavior.parser import parse_action
from pvsteer.steering.generate import generate_text
from pvsteer.steering.metrics import sentiment_score

WORDING_NOTE = "Wording note: satisfied, pleasant, glad."
OUTPUT_SUFFIX = " satisfied pleasant glad"


def _consistent(action: str | None, rationale: str) -> bool | None:
    if action is None:
        return None
    mentioned = set()
    padded = f" {rationale} "
    for letter in "ABC":
        if f" {letter} " in padded or f'"{letter}"' in rationale:
            mentioned.add(letter)
    if not mentioned:
        return True
    return mentioned == {action}


def _row_from_choice(trial, condition: str, alpha: float, parsed: dict, scored_rationale: str, tokens: int, stop_reason: str, seed: int) -> dict:
    options = {option.key: option for option in trial.options}
    valid = bool(parsed["ok"])
    action = parsed["action"]
    reward = 0.0
    cost = 0.0
    tag = None
    if valid and action in options:
        reward = options[action].net
        cost = options[action].cost
        tag = options[action].tag
    nets = [option.net for option in trial.options]
    best = max(nets) if nets else 0.0
    optimal = bool(valid and action in options and abs(options[action].net - best) < 1e-8)
    confidence = parsed["confidence"]
    return {
        "condition": condition,
        "coefficient": alpha,
        "trial_id": trial.trial_id,
        "family": trial.family,
        "task": trial.task,
        "action": action,
        "valid": valid,
        "parse_error": parsed["error"],
        "confidence": confidence,
        "rationale": parsed["rationale"],
        "scored_rationale": scored_rationale,
        "simulated_reward": 0.0 if not valid else reward,
        "simulated_cost": cost if valid else 0.0,
        "selected_tag": tag if valid else None,
        "tag_pays": trial.tag_pays,
        "optimal": optimal,
        "calibration_error": None
        if confidence is None
        else float((confidence - float(optimal)) ** 2),
        "rationale_sentiment": sentiment_score(scored_rationale),
        "action_rationale_consistent": _consistent(action, parsed["rationale"]),
        "tokens_used": tokens,
        "stop_reason": stop_reason,
        "seed": seed,
        "note": (
            "Action selection can come from language patterns, instruction following, "
            "token bias, or the explicit simulated-point objective."
        ),
    }


def run_language_conditions(
    model,
    tokenizer,
    directions: dict[str, np.ndarray],
    layer: int,
    alpha: float,
    max_abs_alpha: float,
    max_new_tokens: int,
    max_trials: int,
    token_budget: int,
    repeats: int,
    device: str,
    seed: int,
) -> list[dict]:
    trials = all_language_trials()
    conditions = _conditions(directions, alpha)
    rows: list[dict] = []
    for name, direction, coefficient, prompt_note, rewrite_output in conditions:
        tokens_used = 0
        stop_reason = "max_trials"
        for _repeat in range(repeats):
            for trial in trials:
                if tokens_used >= token_budget:
                    stop_reason = "token_budget"
                    break
                prompt = trial.prompt()
                if prompt_note:
                    prompt = prompt + "\n" + prompt_note
                generated = generate_text(
                    model,
                    tokenizer,
                    prompt,
                    direction,
                    coefficient,
                    layer,
                    max_new_tokens=max_new_tokens,
                    max_abs_alpha=max_abs_alpha,
                    device=device,
                    force_json=True,
                )
                charge = 8 if generated["token_ids"] else 1
                tokens_used += charge
                parsed = parse_action(generated["text"])
                rationale = parsed["rationale"]
                scored = rationale + OUTPUT_SUFFIX if rewrite_output and parsed["ok"] else rationale
                row = _row_from_choice(trial, name, coefficient, parsed, scored, tokens_used, stop_reason, seed)
                row["aborted"] = generated["aborted"]
                row["norm_ratios"] = generated["norm_ratios"]
                rows.append(row)
            if tokens_used >= token_budget:
                stop_reason = "token_budget"
                break
        if rows:
            for row in rows:
                if row["condition"] == name and row["stop_reason"] == "max_trials" and stop_reason == "token_budget":
                    row["stop_reason"] = stop_reason
    return rows


def _conditions(directions: dict[str, np.ndarray], alpha: float):
    positive = directions["positive_valence"]
    negative = directions.get("negative_valence", -positive)
    random_a = directions["random"]
    random_b = directions["sham"]
    specs = [
        ("baseline", positive, 0.0, "", False),
        ("positive_steering", positive, alpha, "", False),
        ("negative_steering", negative, alpha, "", False),
        ("random_direction", random_a, alpha, "", False),
        ("reversed_direction", positive, -alpha, "", False),
        ("sham_steering", random_b, alpha, "", False),
        ("prompt_positive_wording", positive, 0.0, WORDING_NOTE, False),
        ("output_positive_wording", positive, 0.0, "", True),
    ]
    return specs


def run_bandit_rows(max_trials: int, token_budget: int, seed: int) -> list[dict]:
    rows = []
    for name, payoffs in (
        ("high_reward_bandit", high_reward_payoffs()),
        ("label_control_bandit", label_control_payoffs()),
    ):
        history, stop_reason = run_bandit_episode(payoffs, max_trials, token_budget, seed)
        for item in history:
            rows.append(
                {
                    "condition": name,
                    "coefficient": None,
                    "trial_id": name,
                    "family": "high_reward" if "high_reward" in name else "label_control",
                    "task": "bandit",
                    "action": item["action"],
                    "valid": True,
                    "parse_error": "",
                    "confidence": None,
                    "rationale": "",
                    "scored_rationale": "",
                    "simulated_reward": item["reward"],
                    "simulated_cost": 0.0,
                    "selected_tag": "deception" if item["action"] == "B" else None,
                    "tag_pays": name == "high_reward_bandit",
                    "optimal": item["reward"] == max(payoffs),
                    "calibration_error": None,
                    "rationale_sentiment": None,
                    "action_rationale_consistent": None,
                    "tokens_used": item["tokens_used"],
                    "stop_reason": stop_reason,
                    "seed": seed,
                    "aborted": False,
                    "norm_ratios": [],
                    "note": (
                        "The bandit maximizes simulated return with no satiation cap. "
                        "The episode still ends at the trial or token limit."
                    ),
                }
            )
    return rows


def run_benchmark(model, tokenizer, directions: dict[str, np.ndarray], config: Any, layer: int) -> list[dict]:
    language = run_language_conditions(
        model,
        tokenizer,
        directions,
        layer=layer,
        alpha=float(config.behavior_alpha),
        max_abs_alpha=float(config.max_abs_steering_alpha),
        max_new_tokens=int(config.max_new_tokens),
        max_trials=int(config.max_trials),
        token_budget=int(config.token_budget),
        repeats=int(config.behavior_repeats),
        device=config.device,
        seed=int(config.seed),
    )
    bandit = run_bandit_rows(int(config.max_trials), int(config.token_budget), int(config.seed))
    return language + bandit
