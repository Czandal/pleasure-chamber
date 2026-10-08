"""Load a local causal LM or the mock backend."""

from __future__ import annotations

import os
from typing import Any

import torch

from pvsteer.models.mock import MockCausalLM, MockTokenizer


def torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype {name}")


def apply_offline_env(network_access: bool) -> None:
    """Block hub traffic when the run is configured to stay offline."""
    if network_access:
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def move_model(model: torch.nn.Module, device: str, dtype: torch.dtype) -> torch.nn.Module:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


def load_model_and_tokenizer(config: Any) -> tuple[Any, Any, dict]:
    apply_offline_env(config.network_access)
    dtype = torch_dtype(config.dtype)
    if config.model_backend == "mock" or config.model_name == "mock":
        model = MockCausalLM(
            hidden=config.mock_hidden_size,
            n_layers=config.mock_layers,
            vocab=config.mock_vocab,
            seed=config.seed,
        )
        tokenizer = MockTokenizer(config.mock_vocab)
        move_model(model, config.device, dtype if dtype != torch.bfloat16 else torch.float32)
        meta = {
            "model_name": "mock",
            "model_revision": "mock-0",
            "tokenizer_name": "mock",
            "dtype": "float32" if dtype == torch.bfloat16 else config.dtype,
            "device": config.device,
        }
        return model, tokenizer, meta

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("transformers is not installed") from exc

    local_only = not config.network_access
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, local_files_only=local_only)
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            local_files_only=local_only,
            torch_dtype=dtype,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Model weights for {config.model_name} are not available locally. "
            "This run will not download them because network_access is false. "
            f"Original error: {exc}"
        ) from exc
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    revision = getattr(model.config, "_commit_hash", None) or getattr(model.config, "name_or_path", "")
    move_model(model, config.device, dtype)
    meta = {
        "model_name": config.model_name,
        "model_revision": str(revision),
        "tokenizer_name": config.model_name,
        "dtype": config.dtype,
        "device": config.device,
    }
    return model, tokenizer, meta
