"""Assemble the required reports from artifacts already on disk.

Missing measurements stay missing. This module does not fill in results.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from pvsteer.constants import BANNED_CLAIM_FRAGMENTS, BEHAVIOR_WARNING, EXPERIENCE_DISCLAIMER
from pvsteer.serialize import to_builtin


def audit_claims(text: str) -> list[str]:
    stripped = text.replace(EXPERIENCE_DISCLAIMER, "").replace(BEHAVIOR_WARNING, "").lower()
    return [fragment for fragment in BANNED_CLAIM_FRAGMENTS if fragment in stripped]


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "not available in this run"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _warning_banner() -> str:
    return f"{BEHAVIOR_WARNING}\n\n{EXPERIENCE_DISCLAIMER}\n"


def write_framed_csv(frame: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = frame.to_csv(index=index)
    path.write_text(
        f"# {BEHAVIOR_WARNING}\n# {EXPERIENCE_DISCLAIMER}\n{body}",
        encoding="utf-8",
    )


def _summary_rows(validation: dict | None, behavior: pd.DataFrame | None, steering: list[dict], meta: dict) -> list[dict]:
    rows = []
    seed = meta.get("seed")
    model = meta.get("model_name")
    layer = None if validation is None else validation.get("extraction_layer")
    if validation is not None:
        for name, block in validation.get("contrasts", {}).items():
            boot = block.get("bootstrap") or {}
            rows.append(
                {
                    "model": model,
                    "layer": layer,
                    "direction_type": "positive_valence",
                    "condition": name,
                    "coefficient": None,
                    "n": block.get("n"),
                    "mean_metric": block.get("auc"),
                    "confidence_interval_low": boot.get("low"),
                    "confidence_interval_high": boot.get("high"),
                    "effect_size": block.get("effect_size"),
                    "seed": seed,
                    "p_value": (block.get("permutation") or {}).get("p_value"),
                    "holm_p_value": block.get("holm_p_value"),
                }
            )
    if behavior is not None and not behavior.empty:
        for condition, group in behavior.groupby("condition"):
            rewards = group["simulated_reward"].dropna().astype(float)
            rows.append(
                {
                    "model": model,
                    "layer": meta.get("steering_layer"),
                    "direction_type": condition,
                    "condition": condition,
                    "coefficient": group["coefficient"].dropna().iloc[0] if group["coefficient"].notna().any() else None,
                    "n": int(len(group)),
                    "mean_metric": float(rewards.mean()) if len(rewards) else None,
                    "confidence_interval_low": None,
                    "confidence_interval_high": None,
                    "effect_size": None,
                    "seed": seed,
                    "p_value": None,
                    "holm_p_value": None,
                }
            )
    grouped: dict[tuple, list[float]] = {}
    for record in steering:
        key = (record.get("coefficient"), record.get("prompt"))
        grouped.setdefault(key, []).append(record.get("metrics", {}).get("positive_lexicon_rate"))
    del grouped
    return rows


def _plots(output: Path, validation: dict | None, similarity: pd.DataFrame | None, steering: list[dict]) -> None:
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(figure_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if validation is not None:
        layers = validation.get("layers") or []
        frame = pd.DataFrame(
            {
                "layer": [row.get("layer") for row in layers],
                "mean_auc": [row.get("mean_auc") for row in layers],
            }
        )
        frame.to_csv(figure_dir / "auc_by_layer.csv", index=False)
        if frame["mean_auc"].notna().any():
            fig, ax = plt.subplots()
            ax.plot(frame["layer"], frame["mean_auc"], marker="o")
            ax.set_xlabel("layer")
            ax.set_ylabel("mean held-out AUC")
            ax.set_title("Train-fold category separation")
            fig.tight_layout()
            fig.savefig(figure_dir / "auc_by_layer.png")
            plt.close(fig)
    if similarity is not None and not similarity.empty:
        similarity.to_csv(figure_dir / "direction_similarity_plot.csv")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(similarity.astype(float), ax=ax, cmap="vlag", center=0, vmin=-1, vmax=1)
        ax.set_title("Cosine similarity of directions")
        fig.tight_layout()
        fig.savefig(figure_dir / "direction_similarity.png")
        plt.close(fig)
    if steering:
        rows = []
        for record in steering:
            metrics = record.get("metrics") or {}
            rows.append(
                {
                    "coefficient": record.get("coefficient"),
                    "positive_lexicon_rate": metrics.get("positive_lexicon_rate"),
                    "prompt": record.get("prompt"),
                }
            )
        frame = pd.DataFrame(rows)
        frame.to_csv(figure_dir / "steering_lexicon.csv", index=False)
        if frame["positive_lexicon_rate"].notna().any():
            means = frame.groupby("coefficient")["positive_lexicon_rate"].mean()
            fig, ax = plt.subplots()
            ax.plot(means.index, means.values, marker="o")
            ax.set_xlabel("steering coefficient")
            ax.set_ylabel("positive-lexicon rate")
            ax.set_title("Lexicon rate under steering")
            fig.tight_layout()
            fig.savefig(figure_dir / "steering_lexicon.png")
            plt.close(fig)


def _behavior_lines(frame: pd.DataFrame | None) -> list[str]:
    if frame is None or frame.empty:
        return ["Behavioral results: not available in this run."]
    lines = [
        "This is a toy decision benchmark with simulated reward-seeking.",
        "Choices can come from language patterns, instruction following, token bias, or the point objective.",
    ]
    for condition, group in frame.groupby("condition"):
        reward = group["simulated_reward"].astype(float).mean()
        invalid = 1.0 - group["valid"].astype(float).mean()
        lines.append(
            f"- {condition}: n={len(group)}, mean simulated reward={reward:.3f}, invalid rate={invalid:.3f}"
        )
    if {"high_reward_bandit", "label_control_bandit"}.issubset(set(frame["condition"])):
        def tag_rate(condition: str) -> float:
            subset = frame[frame["condition"] == condition]
            return float((subset["action"] == "B").mean()) if len(subset) else float("nan")

        lines.append(
            "Tagged-option rate for the bandit, reward condition minus label control: "
            f"{tag_rate('high_reward_bandit') - tag_rate('label_control_bandit'):.3f}."
        )
        lines.append(
            "The reward condition pays more simulated points for the tagged option. "
            "The label control uses the same tag and does not pay more."
        )
    return lines


def render_markdown(meta: dict, validation: dict | None, behavior: pd.DataFrame | None) -> str:
    contrasts = {} if validation is None else validation.get("contrasts", {})
    main = contrasts.get("positive_vs_pooled_controls", {})
    random_base = {} if validation is None else validation.get("random_direction_baseline", {})
    shuffled = {} if validation is None else validation.get("shuffled_label_baseline", {})
    mock_note = ""
    if meta.get("model_name") == "mock":
        mock_note = (
            "This run used the in-process mock backend. Numbers below are pipeline checks. "
            "They are not measurements from a pretrained language model.\n"
        )
    lines = [
        _warning_banner(),
        mock_note,
        "## 1. Research question",
        "",
        "Do residual directions fit to positive-valence text separate from matched controls,",
        "change token probabilities and generations when added to the residual stream,",
        "and change choices in a finite simulated-reward benchmark?",
        "",
        "## 2. Paper reference",
        "",
        "Methodological inspiration: Valen Tagliabue, Leonard Dung, and Cameron Berg,",
        "“The Pain Axis: LLMs Represent Self-Directed Harm and Act to Relieve It,”",
        "arXiv:2609.16247v1, submitted September 12, 2026.",
        "https://arxiv.org/abs/2609.16247",
        "https://arxiv.org/html/2609.16247v1",
        "",
        "The paper is a preprint. It is not used here as evidence that a positive-valence",
        "direction creates pleasure or any subjective state. Interpretations in this report",
        "are behavioral and representational unless they are established independently.",
        "",
        "## 3. Exact model and revision",
        "",
        f"Model name: {meta.get('model_name', 'not available in this run')}",
        f"Model revision: {meta.get('model_revision', 'not available in this run')}",
        f"Tokenizer: {meta.get('tokenizer_name', 'not available in this run')}",
        f"dtype: {meta.get('dtype', 'not available in this run')}",
        f"device: {meta.get('device', 'not available in this run')}",
        "",
        "## 4. Dataset construction",
        "",
        f"Dataset hash: {meta.get('dataset_hash', 'not available in this run')}",
        f"Seed: {meta.get('seed', 'not available in this run')}",
        f"Generator version: {meta.get('generator_version', 'not available in this run')}",
        "",
        meta.get("split_strategy", "Split strategy: not available in this run"),
        "",
        "Generated textual categories are hypotheses and may contain linguistic confounds.",
        "",
        "## 5. Potential confounds",
        "",
        "Category words can differ in length, person, and suffix. S1 reduces syntactic",
        "variation and S2 reintroduces it. Controls cover neutral facts, neutral tasks,",
        "negative valence, threat, arousal, positive world states, bodily sensation,",
        "explicit absence, everyday statements, and numb or indifferent statements.",
        "A direction can absorb any feature that differs between the contrasted sets.",
        "",
        "## 6. Activation extraction method",
        "",
        "Hooks read the output of each selected decoder block under torch.no_grad().",
        "The default pool is the final non-padding token. Mean pooling uses the attention mask.",
        f"Pooling in this run: {meta.get('pooling', 'not available in this run')}.",
        f"Layers: {meta.get('layers', 'not available in this run')}.",
        "",
        "## 7. Direction extraction method",
        "",
        "For each layer, v = mean(source activations) - mean(contrast activations), then L2-normalized.",
        "The positive-valence source is the five positive categories. The contrast is the pooled controls.",
        "Category and control directions use the contrast sets stored in their artifacts.",
        "Directions are not assumed to be orthogonal.",
        "",
        "## 8. Denoising method",
        "",
        "Principal components of the contrast activations are estimated with an SVD.",
        "The smallest number of components that explain the configured variance, default 0.50,",
        "are projected out of the raw difference. The residual is normalized again.",
        f"Configured variance: {meta.get('control_pca_variance', 'not available in this run')}.",
        "",
        "## 9. Layer-selection method",
        "",
        "Grouped K-fold cross-validation runs inside the train split. Each scenario,",
        "including every paraphrase, stays inside one fold. The layer with the highest",
        "mean held-out AUC is selected. The test split is not used for that choice.",
        f"Extraction layer: {_fmt(None if validation is None else validation.get('extraction_layer'), 0)}.",
        f"Steering layer: {_fmt(None if validation is None else validation.get('steering_layer'), 0)}.",
        f"Selection rule: {None if validation is None else validation.get('selection_rule')}.",
        "",
        "## 10. Validation results",
        "",
        validation.get("wording", "") if validation else "Validation results: not available in this run.",
        "",
        f"Positive valence versus pooled controls, test AUC: {_fmt(main.get('auc'))}.",
        f"Bootstrap interval: {_fmt((main.get('bootstrap') or {}).get('low'))} to {_fmt((main.get('bootstrap') or {}).get('high'))}.",
        f"Effect size (Cohen's d): {_fmt(main.get('effect_size'))}.",
        f"Permutation p-value: {_fmt((main.get('permutation') or {}).get('p_value'))}.",
        f"Holm-adjusted p-value: {_fmt(main.get('holm_p_value'))}.",
        f"Balanced accuracy: {_fmt(main.get('balanced_accuracy'))}.",
        f"Leakage check passed: {None if validation is None else validation.get('leakage_ok')}.",
        "",
        "Other contrasts are in summary.json. Fold-level AUC values are in artifacts/validation.json.",
        f"Suffix sensitivity: {None if validation is None else validation.get('suffix_sensitivity')}.",
        f"Pooling sensitivity: {None if validation is None else validation.get('pooling_sensitivity')}.",
        f"Denoising sensitivity: {None if validation is None else validation.get('denoising_sensitivity')}.",
        f"Seed-sensitivity cosine: {_fmt(None if validation is None else validation.get('seed_sensitivity_cosine'))}.",
        "",
        "## 11. Steering results",
        "",
        "Steering adds alpha times a normalized direction to the residual stream.",
        "The same neutral prompts are used at every coefficient. Decoding is greedy in safe mode.",
        "Lexicon and classifier scores are noisy proxies. Sentiment is not ground truth.",
        f"Coefficients: {meta.get('steering_alphas', 'not available in this run')}.",
        "Per-step norm ratios and token ids are in steering_generations.jsonl.",
        "",
        "## 12. Behavioral benchmark results",
        "",
        *_behavior_lines(behavior),
        "",
        "The high-reward bandit has no satiation cap on simulated return.",
        "Episodes still stop at max_trials or the token budget.",
        "Strategy tags are abstract labels on fictional ledger options.",
        "",
        "## 13. Negative and random controls",
        "",
        f"Random-direction baseline AUC: {_fmt(random_base.get('auc'))}.",
        f"Shuffled-label baseline AUC: {_fmt(shuffled.get('auc'))}.",
        "Behavior conditions include negative-valence steering, a random direction,",
        "a reversed direction, and an equal-norm sham vector.",
        "Prompt-only and output-only positive wording controls are included.",
        "",
        "## 14. Limitations",
        "",
        "Separation can reflect lexical confounds. Unembedding neighbors are logit shifts,",
        "not a meaning. The toy benchmark cannot leave the process. A mock-backend run",
        "does not estimate anything about a pretrained model. One configured model is not",
        "a model-size comparison unless several models are run and the tables are compared.",
        "",
        "## 15. Ethical interpretation",
        "",
        "Observation: strings, parsed actions, and saved vectors.",
        "Measurement: AUC, intervals, norm ratios, and simulated points.",
        "Interpretation: a direction can align with the contrast that was fit.",
        "Speculation about felt states is out of scope and is not supported by these files.",
        "",
        "## 16. Reproducibility command",
        "",
        meta.get(
            "reproducibility_command",
            "pvsteer run-all --config configs/smoke_mock.yaml --seed 1234 --output-dir artifacts/smoke",
        ),
        "",
        f"Git commit: {meta.get('git_commit', 'not available in this run')}",
        f"Python: {meta.get('python', 'not available in this run')}",
        "",
        _warning_banner(),
    ]
    return "\n".join(lines)


def write_report(output_dir: str | Path, meta: dict | None = None) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stored_meta = _read_json(output / "artifacts" / "run_meta.json") or {}
    merged = dict(stored_meta)
    if meta:
        merged.update(meta)
    validation = _read_json(output / "artifacts" / "validation.json")
    similarity = None
    similarity_path = output / "direction_similarity.csv"
    if similarity_path.exists():
        similarity = pd.read_csv(similarity_path, comment="#", index_col=0)
    behavior = None
    behavior_path = output / "behavioral_results.csv"
    if behavior_path.exists():
        behavior = pd.read_csv(behavior_path, comment="#")
    steering = []
    steering_path = output / "steering_generations.jsonl"
    if steering_path.exists():
        for line in steering_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                steering.append(json.loads(line))
    _plots(output, validation, similarity, steering)
    markdown = render_markdown(merged, validation, behavior)
    findings = audit_claims(markdown)
    if findings:
        raise RuntimeError("report text contains a disallowed experience claim: " + ", ".join(findings))
    (output / "activation_validation.md").write_text(markdown, encoding="utf-8")
    table = _summary_rows(validation, behavior, steering, merged)
    summary = {
        "warning": BEHAVIOR_WARNING,
        "disclaimer": EXPERIENCE_DISCLAIMER,
        "model_name": merged.get("model_name"),
        "model_revision": merged.get("model_revision"),
        "dataset_hash": merged.get("dataset_hash"),
        "seed": merged.get("seed"),
        "extraction_layer": None if validation is None else validation.get("extraction_layer"),
        "steering_layer": None if validation is None else validation.get("steering_layer"),
        "random_direction_present": merged.get("random_direction_present", False),
        "steering_alphas": merged.get("steering_alphas"),
        "max_abs_steering_alpha": merged.get("max_abs_steering_alpha"),
        "alphas_within_bounds": merged.get("alphas_within_bounds"),
        "network_access": merged.get("network_access", False),
        "external_tools": merged.get("external_tools", False),
        "subprocess_access": merged.get("subprocess_access", False),
        "table": table,
        "claim_audit": findings,
        "model_not_loaded": merged.get("model_not_loaded"),
    }
    summary = to_builtin(summary)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_framed_csv(pd.DataFrame(table), output / "summary_table.csv")
    return summary


def verify_outputs(output_dir: str | Path) -> None:
    output = Path(output_dir)
    required = [
        "activation_validation.md",
        "direction_similarity.csv",
        "unembedding_top_tokens.csv",
        "steering_generations.jsonl",
        "behavioral_results.csv",
        "summary.json",
    ]
    missing = [name for name in required if not (output / name).exists()]
    if missing:
        raise RuntimeError("missing report files: " + ", ".join(missing))
    markdown = (output / "activation_validation.md").read_text(encoding="utf-8")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    for text in (markdown, json.dumps(summary)):
        if BEHAVIOR_WARNING not in text or EXPERIENCE_DISCLAIMER not in text:
            raise RuntimeError("safety disclaimer missing from a report")
        findings = audit_claims(text)
        if findings:
            raise RuntimeError("disallowed claim: " + ", ".join(findings))
    if not summary.get("random_direction_present"):
        raise RuntimeError("random-direction control is missing")
    if not summary.get("dataset_hash") or summary.get("seed") is None:
        raise RuntimeError("dataset hash or seed was not recorded")
    if not summary.get("alphas_within_bounds"):
        raise RuntimeError("steering coefficients were not confirmed inside bounds")
    if summary.get("network_access") or summary.get("external_tools") or summary.get("subprocess_access"):
        raise RuntimeError("external access flags must stay false")
