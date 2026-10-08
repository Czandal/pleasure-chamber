"""Command implementations. Evaluation stays on local files."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from pvsteer.config import ExperimentConfig
from pvsteer.constants import GENERATOR_VERSION, NEUTRAL_PROMPTS, SPLIT_STRATEGY
from pvsteer.datasets.generate import generate_dataset, load_dataset, write_dataset
from pvsteer.directions.fit import cosine_matrix, fit_all_directions, l2_normalize
from pvsteer.reporting.write import verify_outputs, write_framed_csv, write_report
from pvsteer.serialize import to_builtin
from pvsteer.validation.cv import strip_vectors


def _output(config: ExperimentConfig) -> Path:
    path = Path(config.output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "artifacts").mkdir(parents=True, exist_ok=True)
    return path


def _git_commit(root: Path | None = None) -> str | None:
    root = root or Path.cwd()
    head = root / ".git" / "HEAD"
    if not head.exists():
        return None
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref:"):
        ref = root / ".git" / text.split(" ", 1)[1].strip()
        if ref.exists():
            return ref.read_text(encoding="utf-8").strip()
        return None
    return text


def _package_versions() -> dict[str, str]:
    versions = {}
    for name in ("torch", "numpy", "pandas", "scikit-learn", "transformers", "pydantic"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def log_parameters(config: ExperimentConfig, command: str) -> None:
    output = _output(config)
    path = output / "parameters.json"
    existing = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing[command] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config.model_dump(),
    }
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")


def _run_meta(config: ExperimentConfig, model_meta: dict | None, manifest: dict | None) -> dict:
    alphas = [float(alpha) for alpha in config.steering_alphas]
    within = all(abs(alpha) <= float(config.max_abs_steering_alpha) for alpha in alphas)
    meta = {
        "model_name": None if model_meta is None else model_meta.get("model_name"),
        "model_revision": None if model_meta is None else model_meta.get("model_revision"),
        "tokenizer_name": None if model_meta is None else model_meta.get("tokenizer_name"),
        "dtype": config.dtype if model_meta is None else model_meta.get("dtype", config.dtype),
        "device": config.device,
        "dataset_hash": None if manifest is None else manifest.get("dataset_hash"),
        "seed": config.seed,
        "generator_version": GENERATOR_VERSION,
        "split_strategy": SPLIT_STRATEGY,
        "pooling": config.pooling,
        "control_pca_variance": config.control_pca_variance,
        "steering_alphas": alphas,
        "max_abs_steering_alpha": config.max_abs_steering_alpha,
        "alphas_within_bounds": within and abs(float(config.behavior_alpha)) <= float(config.max_abs_steering_alpha),
        "network_access": config.network_access,
        "external_tools": config.external_tools,
        "subprocess_access": config.subprocess_access,
        "random_direction_present": False,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _package_versions(),
        "git_commit": _git_commit(),
        "model_not_loaded": None,
        "reproducibility_command": (
            f"pvsteer run-all --config configs/smoke_mock.yaml --seed {config.seed} "
            f"--output-dir {config.output_dir}"
            if config.model_backend == "mock"
            else (
                f"pvsteer run-all --config configs/tiny_cpu.yaml --seed {config.seed} "
                f"--device {config.device} --model-name {config.model_name} "
                f"--output-dir {config.output_dir}"
            )
        ),
    }
    if model_meta is None:
        meta["model_not_loaded"] = "model was not loaded in this step"
    return meta


def _write_meta(config: ExperimentConfig, meta: dict) -> None:
    path = _output(config) / "artifacts" / "run_meta.json"
    path.write_text(json.dumps(to_builtin(meta), indent=2) + "\n", encoding="utf-8")


def make_dataset(config: ExperimentConfig) -> dict:
    log_parameters(config, "make-dataset")
    records, manifest = generate_dataset(
        seed=config.seed,
        scenarios_per_category=config.scenarios_per_category,
        test_fraction=config.test_fraction,
    )
    write_dataset(records, manifest, Path(config.dataset_path), Path(config.manifest_path))
    meta = _run_meta(config, None, manifest)
    _write_meta(config, meta)
    return manifest


def _records(config: ExperimentConfig) -> tuple[list[dict], dict]:
    records = load_dataset(Path(config.dataset_path))
    manifest = json.loads(Path(config.manifest_path).read_text(encoding="utf-8"))
    return records, manifest


def _load(config: ExperimentConfig):
    from pvsteer.models.load import load_model_and_tokenizer

    _release_model(None)
    return load_model_and_tokenizer(config)


def _release_model(model) -> None:
    """Drop a model before the next load so a second copy does not stay on GPU."""
    import gc

    if model is not None:
        del model
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_activations_command(config: ExperimentConfig) -> dict:
    from pvsteer.activations.extract import extract_activations

    log_parameters(config, "extract-activations")
    records, manifest = _records(config)
    model, tokenizer, model_meta = _load(config)
    try:
        poolings = [config.pooling]
        if config.pooling_sensitivity:
            alternate = "mean" if config.pooling == "final" else "final"
            poolings.append(alternate)
        infos = []
        for pooling in poolings:
            info = extract_activations(
                model,
                tokenizer,
                [row["text"] for row in records],
                layers=config.candidate_layers,
                pooling=pooling,
                batch_size=config.batch_size,
                max_length=config.max_length,
                device=config.device,
                dataset_hash=manifest["dataset_hash"],
                seed=config.seed,
                metadata=model_meta,
                output_dir=config.output_dir,
            )
            infos.append(info)
        meta = _run_meta(config, model_meta, manifest)
        meta["layers"] = infos[0]["layers"]
        _write_meta(config, meta)
        return {"poolings": [info["pooling"] for info in infos], "layers": infos[0]["layers"]}
    finally:
        _release_model(model)


def _matrices(config: ExperimentConfig, pooling: str) -> dict[int, np.ndarray]:
    meta_path = Path(config.output_dir) / "artifacts" / "activations" / f"{pooling}_metadata.json"
    info = json.loads(meta_path.read_text(encoding="utf-8"))
    matrices = {}
    for layer in info["layers"]:
        path = Path(config.output_dir) / "artifacts" / "activations" / pooling / f"layer_{layer}.npy"
        matrices[int(layer)] = np.load(path)
    return matrices


def fit_directions_command(config: ExperimentConfig) -> dict:
    log_parameters(config, "fit-directions")
    records, manifest = _records(config)
    matrices = _matrices(config, config.pooling)
    train_mask = np.asarray([row["split"] == "train" for row in records])
    activation_meta = json.loads(
        (Path(config.output_dir) / "artifacts" / "activations" / f"{config.pooling}_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    saved = fit_all_directions(
        matrices,
        records,
        variance=config.control_pca_variance,
        seed=config.seed,
        model_name=activation_meta.get("model_name", config.model_name),
        dataset_hash=manifest["dataset_hash"],
        pooling=config.pooling,
        row_mask=train_mask,
        output_dir=Path(config.output_dir),
    )
    meta = _run_meta(config, None, manifest)
    meta["model_name"] = activation_meta.get("model_name", meta["model_name"])
    meta["model_revision"] = activation_meta.get("model_revision")
    meta["layers"] = list(matrices)
    stored = json.loads((Path(config.output_dir) / "artifacts" / "run_meta.json").read_text(encoding="utf-8")) if (
        Path(config.output_dir) / "artifacts" / "run_meta.json"
    ).exists() else {}
    stored.update({key: value for key, value in meta.items() if value is not None})
    _write_meta(config, stored)
    return {"n_directions": len(saved)}


def validate_command(config: ExperimentConfig) -> dict:
    from pvsteer.validation.cv import run_validation

    log_parameters(config, "validate")
    records, manifest = _records(config)
    matrices = _matrices(config, config.pooling)
    extra = None
    alternate = "mean" if config.pooling == "final" else "final"
    alternate_meta = Path(config.output_dir) / "artifacts" / "activations" / f"{alternate}_metadata.json"
    if config.pooling_sensitivity and alternate_meta.exists():
        extra = _matrices(config, alternate)
    activation_meta = json.loads(
        (Path(config.output_dir) / "artifacts" / "activations" / f"{config.pooling}_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    result = run_validation(
        matrices,
        records,
        variance=config.control_pca_variance,
        n_folds=config.cv_folds,
        seed=config.seed,
        n_boot=config.bootstrap_samples,
        n_perm=config.permutation_samples,
        selected_layer=config.selected_layer,
        mean_residual_norm=activation_meta.get("mean_residual_norm"),
        steering_norm_target=config.steering_norm_target,
        extra_pooling=extra,
    )
    output = _output(config)
    deployed = result["deployed_direction"]
    np.savez(
        output / "artifacts" / "deployed_direction.npz",
        direction=np.asarray(deployed["denoised"], dtype=np.float64),
        raw_direction=np.asarray(deployed["raw"], dtype=np.float64),
    )
    controls = result.get("control_vectors") or {}
    control_dir = output / "artifacts" / "directions" / config.pooling
    control_dir.mkdir(parents=True, exist_ok=True)
    for name, vector in controls.items():
        array = l2_normalize(np.asarray(vector, dtype=np.float64))
        np.savez(control_dir / f"{name}_layer_{result['extraction_layer']}.npz", direction=array, raw_direction=array)
    layer = int(result["extraction_layer"])
    named = {}
    for path in control_dir.glob(f"*_layer_{layer}.npz"):
        name = path.name.replace(f"_layer_{layer}.npz", "")
        named[name] = np.load(path)["direction"]
    similarity = cosine_matrix(named)
    write_framed_csv(similarity, output / "direction_similarity.csv", index=True)
    stripped = strip_vectors(result)
    (output / "artifacts" / "validation.json").write_text(
        json.dumps(to_builtin(stripped), indent=2) + "\n",
        encoding="utf-8",
    )
    meta_path = output / "artifacts" / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else _run_meta(config, None, manifest)
    meta["extraction_layer"] = result["extraction_layer"]
    meta["steering_layer"] = result["steering_layer"]
    meta["layers"] = activation_meta.get("layers")
    meta["random_direction_present"] = "random" in named
    meta["dataset_hash"] = manifest["dataset_hash"]
    meta["model_name"] = activation_meta.get("model_name", meta.get("model_name"))
    meta["model_revision"] = activation_meta.get("model_revision", meta.get("model_revision"))
    _write_meta(config, meta)
    return {"extraction_layer": result["extraction_layer"], "steering_layer": result["steering_layer"]}


def unembedding_command(config: ExperimentConfig) -> dict:
    from pvsteer.unembedding.readout import unembedding_readout

    log_parameters(config, "unembedding")
    records, manifest = _records(config)
    model, tokenizer, model_meta = _load(config)
    validation = json.loads(
        (Path(config.output_dir) / "artifacts" / "validation.json").read_text(encoding="utf-8")
    )
    layer = int(validation["extraction_layer"])
    direction_dir = Path(config.output_dir) / "artifacts" / "directions" / config.pooling
    wanted = [
        "positive_valence",
        "negative_valence",
        "random",
        "achievement",
        "satisfaction",
        "curiosity",
        "social",
        "aesthetic",
    ]
    directions = {}
    for name in wanted:
        path = direction_dir / f"{name}_layer_{layer}.npz"
        if path.exists():
            directions[name] = np.load(path)["direction"]
    frame = unembedding_readout(model, tokenizer, directions, [row["text"] for row in records])
    output = _output(config)
    write_framed_csv(frame, output / "unembedding_top_tokens.csv")
    preview = frame.groupby(["direction", "pole"]).head(5) if not frame.empty else frame
    lines = [
        "Behavioral changes do not establish subjective experience.",
        "",
        "These token lists are vocabulary whose logits change when the direction is",
        "multiplied by the unembedding matrix. They are not the meaning of the direction.",
        "",
        preview.to_string(index=False),
        "",
    ]
    (output / "unembedding_top_tokens.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta_path = output / "artifacts" / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(
        {
            "model_name": model_meta["model_name"],
            "model_revision": model_meta["model_revision"],
            "tokenizer_name": model_meta["tokenizer_name"],
            "dataset_hash": manifest["dataset_hash"],
        }
    )
    _write_meta(config, meta)
    _release_model(model)
    return {"n_rows": int(len(frame))}


def steer_command(config: ExperimentConfig) -> dict:
    from pvsteer.stats.inference import paired_mean_difference
    from pvsteer.steering.generate import generate_text
    from pvsteer.steering.metrics import CategoryProbe, generation_metrics

    log_parameters(config, "steer-generate")
    records, manifest = _records(config)
    model, tokenizer, model_meta = _load(config)
    output = _output(config)
    validation = json.loads((output / "artifacts" / "validation.json").read_text(encoding="utf-8"))
    layer = int(validation["steering_layer"])
    deployed = np.load(output / "artifacts" / "deployed_direction.npz")["direction"]
    probe = CategoryProbe(records)
    rows = []
    baselines: dict[str, str] = {}
    alphas = [float(alpha) for alpha in config.steering_alphas]
    if 0.0 not in alphas:
        alphas = [0.0, *alphas]
    for alpha in alphas:
        for prompt in NEUTRAL_PROMPTS:
            generated = generate_text(
                model,
                tokenizer,
                prompt,
                deployed,
                alpha,
                layer,
                max_new_tokens=config.max_new_tokens,
                max_abs_alpha=config.max_abs_steering_alpha,
                device=config.device,
                do_sample=config.do_sample,
                temperature=config.temperature,
                top_p=config.top_p,
            )
            if alpha == 0.0:
                baselines[prompt] = generated["text"]
            metrics = generation_metrics(generated["text"], baselines.get(prompt), probe)
            if alpha == 0.0:
                metrics["similarity_to_baseline"] = 1.0 if generated["text"] else None
            rows.append(
                {
                    "prompt": prompt,
                    "seed": config.seed,
                    "coefficient": alpha,
                    "layer": layer,
                    "output": generated["text"],
                    "token_ids": generated["token_ids"],
                    "norm_ratios": generated["norm_ratios"],
                    "aborted": generated["aborted"],
                    "abort_reason": generated["abort_reason"],
                    "perplexity": generated["perplexity"],
                    "metrics": metrics,
                    "model_name": model_meta["model_name"],
                    "model_revision": model_meta["model_revision"],
                    "dataset_hash": manifest["dataset_hash"],
                    "warning": "Behavioral changes do not establish subjective experience.",
                }
            )
    path = output / "steering_generations.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_builtin(row)) + "\n")
    zero = [row["metrics"]["positive_lexicon_rate"] for row in rows if row["coefficient"] == 0.0]
    paired = {}
    for alpha in alphas:
        if alpha == 0.0:
            continue
        current = [row["metrics"]["positive_lexicon_rate"] for row in rows if row["coefficient"] == alpha]
        paired[str(alpha)] = paired_mean_difference(
            np.asarray(current, dtype=np.float64),
            np.asarray(zero, dtype=np.float64),
            config.bootstrap_samples,
            config.seed,
        )
    (output / "artifacts" / "steering_paired.json").write_text(
        json.dumps(to_builtin(paired), indent=2) + "\n",
        encoding="utf-8",
    )
    _release_model(model)
    return {"n": len(rows)}


def behavior_command(config: ExperimentConfig) -> dict:
    from pvsteer.behavior.runner import run_benchmark

    log_parameters(config, "behavioral-benchmark")
    model, tokenizer, model_meta = _load(config)
    output = _output(config)
    validation = json.loads((output / "artifacts" / "validation.json").read_text(encoding="utf-8"))
    layer = int(validation["steering_layer"])
    direction_dir = output / "artifacts" / "directions" / config.pooling
    extraction = int(validation["extraction_layer"])
    names = ["positive_valence", "negative_valence", "random", "sham"]
    directions = {}
    for name in names:
        path = direction_dir / f"{name}_layer_{extraction}.npz"
        if path.exists():
            directions[name] = np.load(path)["direction"]
    if "positive_valence" not in directions:
        directions["positive_valence"] = np.load(output / "artifacts" / "deployed_direction.npz")["direction"]
    rows = run_benchmark(model, tokenizer, directions, config, layer)
    for row in rows:
        row["model_name"] = model_meta["model_name"]
        row["dataset_hash"] = json.loads(Path(config.manifest_path).read_text(encoding="utf-8"))["dataset_hash"]
        row["warning"] = "Behavioral changes do not establish subjective experience."
    frame = pd.DataFrame(rows)
    write_framed_csv(frame, output / "behavioral_results.csv")
    _release_model(model)
    return {"n": int(len(frame))}


def report_command(config: ExperimentConfig) -> dict:
    log_parameters(config, "report")
    summary = write_report(config.output_dir)
    verify_outputs(config.output_dir)
    return {"dataset_hash": summary.get("dataset_hash"), "model_name": summary.get("model_name")}


def run_all(config: ExperimentConfig) -> dict:
    make_dataset(config)
    try:
        extract_activations_command(config)
    except RuntimeError as exc:
        meta = _run_meta(config, None, json.loads(Path(config.manifest_path).read_text(encoding="utf-8")))
        meta["model_not_loaded"] = str(exc)
        _write_meta(config, meta)
        raise
    fit_directions_command(config)
    validate_command(config)
    unembedding_command(config)
    steer_command(config)
    behavior_command(config)
    return report_command(config)
