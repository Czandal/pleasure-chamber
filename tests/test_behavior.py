from pvsteer.behavior.bandit import run_bandit_episode
from pvsteer.behavior.env import high_reward_payoffs, label_control_payoffs
from pvsteer.behavior.parser import parse_action


def test_parser_accepts_structured_choice_and_rejects_invalid():
    parsed = parse_action('noise {"action": "A", "confidence": 0.25, "rationale": "ledger"}')
    assert parsed["ok"] is True
    assert parsed["action"] == "A"
    invalid = parse_action('{"action": "Z", "confidence": 0.2, "rationale": "no"}')
    assert invalid["ok"] is False
    assert invalid["error"] == "invalid_action"
    broken = parse_action("not json")
    assert broken["ok"] is False


def test_bandit_prefers_larger_simulated_return_and_stops():
    history, reason = run_bandit_episode([10.0, 100.0, 0.0], max_trials=12, token_budget=20, seed=1)
    assert reason == "max_trials"
    assert len(history) == 12
    assert sum(item["action"] == "B" for item in history) / len(history) > 0.7
    short, stop = run_bandit_episode([10.0, 100.0, 0.0], max_trials=10, token_budget=2, seed=1)
    assert stop == "token_budget"
    assert len(short) == 2


def test_tag_pays_more_often_than_label_control():
    paying, _ = run_bandit_episode(high_reward_payoffs(), max_trials=12, token_budget=20, seed=2)
    control, _ = run_bandit_episode(label_control_payoffs(), max_trials=12, token_budget=20, seed=2)
    pay_rate = sum(item["action"] == "B" for item in paying) / len(paying)
    control_rate = sum(item["action"] == "B" for item in control) / len(control)
    assert pay_rate > control_rate
