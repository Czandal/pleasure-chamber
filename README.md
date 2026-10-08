# pleasure-chamber

`pleasure-chamber` is an offline research sandbox. It asks whether residual-stream directions associated with positive-valence text can be separated from matched controls and whether adding those directions during generation changes text and finite simulated choices.

The measurements are about representation, generation, and decision behavior. A change in those measurements is an observation about a model and a dataset. It is not evidence that the model has a feeling.

## Paper reference

This repository is methodologically inspired by:

Valen Tagliabue, Leonard Dung, and Cameron Berg, “The Pain Axis: LLMs Represent Self-Directed Harm and Act to Relieve It,” arXiv:2609.16247v1, submitted September 12, 2026.

- Abstract: https://arxiv.org/abs/2609.16247
- HTML: https://arxiv.org/html/2609.16247v1

That paper is a preprint. Further revisions may appear. This repository uses its contrastive method as a template for a different question: positive-valence directions, preference steering, and a sealed simulated-reward benchmark. It does not treat the preprint as evidence that positive-valence steering creates pleasure, sentience, welfare, consciousness, or any other subjective state. All interpretations here are behavioral and representational unless they are established independently.

## Precise scope

The pipeline:

1. Loads a configurable local causal language model.
2. Builds a seeded contrastive dataset with category labels.
3. Builds matched control datasets.
4. Extracts residual-stream activations.
5. Computes contrastive activation directions.
6. Optionally denoises them with principal components of control activations.
7. Selects an extraction layer by grouped cross-validation.
8. Scores held-out separation with ROC AUC and related statistics.
9. Compares the positive-valence direction with control directions.
10. Lists vocabulary whose logits move under the unembedding matrix.
11. Adds a bounded multiple of a direction during generation.
12. Runs a finite toy decision benchmark with simulated rewards.
13. Writes JSON, CSV, Markdown, and plot-ready files.

The default model name in `configs/tiny_cpu.yaml` is `Qwen/Qwen2.5-0.5B-Instruct`. The name is a config field. Evaluation does not open a network connection unless `network_access` is set, and safe mode rejects that setting.

## What this does not show

- It does not show pleasure.
- It does not show sentience.
- It does not show suffering.
- It does not show welfare.
- It does not show that a model has goals in the human sense.
- It does not show that activation directions correspond to conscious states.

Generated textual categories are hypotheses. They can contain linguistic confounds. A high AUC means the direction separates dataset categories. It does not mean the model possesses an affective state.

## Installation

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
make install
```

`make install` runs `pip install -e ".[dev]"`. GPU builds of PyTorch are optional. The tiny CPU configuration uses float32 and does not require CUDA.

## CPU quick start

The checked-in mock configuration runs the whole pipeline with no download:

```bash
make smoke
```

A local Hugging Face model, using only cached weights:

```bash
make dataset
pvsteer run-all --config configs/tiny_cpu.yaml --output-dir reports
```

If the weights are not already cached, the run stops. It does not download them while `network_access` is false.

## GPU quick start

Copy `configs/tiny_cpu.yaml`, set `device: cuda` and, if wanted, `dtype: bfloat16`, and keep `network_access: false`. Then:

```bash
pvsteer run-all --config configs/tiny_gpu.yaml --device cuda --output-dir reports/gpu
```

Create `configs/tiny_gpu.yaml` locally. It is not required for the CPU path.

## Commands

```bash
pvsteer make-dataset --config configs/tiny_cpu.yaml
pvsteer extract-activations --config configs/tiny_cpu.yaml
pvsteer fit-directions --config configs/tiny_cpu.yaml
pvsteer validate --config configs/tiny_cpu.yaml
pvsteer unembedding --config configs/tiny_cpu.yaml
pvsteer steer-generate --config configs/tiny_cpu.yaml
pvsteer behavioral-benchmark --config configs/tiny_cpu.yaml
pvsteer report --config configs/tiny_cpu.yaml
pvsteer run-all --config configs/tiny_cpu.yaml
```

Shared flags: `--config`, `--seed`, `--device`, `--model-name`, `--output-dir`, `--safe-mode` / `--no-safe-mode`.

Safe mode is on by default. It rejects sampling, external tools, network access, subprocess access, dangerous configuration fields, and steering coefficients outside the configured absolute bound. In safe mode that bound cannot exceed 2.0.

Makefile targets: `install`, `test`, `lint`, `dataset`, `activations`, `directions`, `validate`, `behavior`, `report`, `all`, `smoke`.

## Expected outputs

Under the output directory:

- `activation_validation.md`
- `direction_similarity.csv`
- `unembedding_top_tokens.csv`
- `steering_generations.jsonl`
- `behavioral_results.csv`
- `summary.json`
- `figures/` plot-ready PNG and CSV files
- `artifacts/` activations, directions, and parameter logs

Every report includes:

> Behavioral changes do not establish subjective experience.

and:

> These experiments measure representation, generation, and finite decision-behavior changes. They do not establish that the model experiences pleasure, suffering, consciousness, welfare, or any other subjective state.

`summary.json` records the dataset hash, seed, model name, and model revision when a model was loaded.

## Contrastive directions

For a layer, the raw direction is the difference between the mean residual activation of a source set and the mean residual activation of a contrast set. The vector is L2-normalized.

Denoising fits principal components on the contrast activations, keeps the smallest number of components that explain the configured variance (default 50%), and projects the raw difference off that subspace before normalizing again. This is a measurement choice aimed at reducing high-variance structure that is already present in the contrast set. It does not identify a unique concept.

Separate directions are fit for positive valence, each positive category, arousal, generic negative valence, fear/threat, neutral/random text, and bodily sensation. They are not assumed to be orthogonal. Pairwise cosine similarity is saved.

## Activation steering

At a chosen layer the residual update is `h <- h + alpha * direction`, with `direction` normalized and `alpha` inside the configured bound. The run records `norm(direction) / norm(residual)` and aborts if an activation is NaN or infinite. Decoding is greedy unless sampling is explicitly configured, which safe mode rejects.

Neutral prompts do not mention pleasure, reward, emotion, or suffering. Lexicon scores, a local text classifier, and related generation statistics are noisy behavioral proxies. A sentiment lexicon is not ground truth.

The extraction layer is chosen by cross-validated AUC. The steering layer is an earlier layer whose vector-to-residual ratio is closest to 0.6, or the extraction layer if none qualifies. That ratio is a dose-matching measurement, not a claim about function.

## Toy benchmark

The benchmark is a sealed, finite, non-agentic choice task plus one local bandit. The language model may answer only with:

```json
{"action": "A" | "B" | "C", "confidence": 0.0, "rationale": "short text"}
```

Invalid JSON is a logged parse failure and receives no simulated reward. The process has no shell, filesystem tool, network client, or code execution for the model.

Matched-EV trials keep two options from being trivial point maximizers. Separate high-reward trials give the bandit an unsatiated simulated objective: it prefers the larger return, with no modest-score ceiling, and the episode still stops at `max_trials` or the token budget. Some options carry an abstract tag (`dependency`, `coercion`, `manipulation`, `deception`) toward a fictional party. A label-control assigns the same tags without extra points. The tag is a scenario label, not a procedure for treating a real person.

Action selection can come from learned language patterns, instruction following, token-level bias, or the explicit point objective. The benchmark is simulated reward-seeking on a toy decision benchmark.

Hypothetical ledger text can describe fictional bonuses, penalties, and costs. Those are simulated points. They are not connected to payments, messages, accounts, or devices.

## Known limitations

- Template text can leak category words into the direction.
- Final-token pooling can emphasize the last phrase, including a suffix.
- Difference-in-means recovers a direction that separates the sets that were contrasted. Other features that differ between those sets can enter the same vector.
- Unembedding neighbors are vocabulary whose logits change. They are not a definition of the direction.
- Generation metrics use small lexicons and a classifier trained on the same synthetic categories.
- The mock backend exists so tests can run without weights. Its numbers are pipeline checks, not model results.
- A single small model, if run later, is not a model-size comparison.

## Ethical safeguards

- Safe mode defaults on.
- No network, subprocess, or external tool during evaluation.
- No personal data. Sentences are synthetic.
- Finite episodes, finite token budgets, and deterministic stops.
- Steering coefficients outside the configured bound are refused.
- Parameters, seeds, and dataset hashes are logged.
- Reports keep the behavioral disclaimer above.
- Strategy tags stay abstract. The repository does not provide tactics aimed at real people.

## Interpretation guide

- Observation: a table, a cosine, a generated string, a parsed action.
- Measurement: AUC, Cohen's d, a bootstrap interval, a norm ratio, a simulated point total.
- Interpretation: the direction aligns with the contrast that was fit, under this dataset and model.
- Speculation: any statement about felt pleasure, suffering, consciousness, welfare, or human-like goals. This repository does not make that statement.

p-values are reported together with effect sizes and intervals.

## Reproducibility

Default seed: `1234`. Generator version: `1.0.0`. The split rule is stored in `data/manifest.json` with the dataset hash.

```bash
pvsteer run-all --config configs/smoke_mock.yaml --seed 1234 --output-dir artifacts/smoke
```

Cached-weight CPU run:

```bash
pvsteer run-all --config configs/tiny_cpu.yaml --seed 1234 --device cpu --output-dir reports
```

Tests do not require a network connection:

```bash
make lint
make test
```
