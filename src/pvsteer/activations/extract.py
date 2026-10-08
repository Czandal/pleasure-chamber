"""Hook-based residual extraction into memory-mapped arrays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from pvsteer.models.residual import (
    get_decoder_layers,
    hidden_from_output,
    pool_hidden,
    resolve_layers,
)


def _capture_hook(store: dict[int, torch.Tensor], layer_index: int):
    def _hook(module, inputs, output):
        del module, inputs
        store[layer_index] = hidden_from_output(output).detach()

    return _hook


def extract_activations(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: Sequence[str],
    layers: object,
    pooling: str = "final",
    batch_size: int = 2,
    max_length: int = 256,
    device: str = "cpu",
    dataset_hash: str = "",
    seed: int = 0,
    metadata: dict | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    """Extract pooled residual activations.

    Arrays are written as ``.npy`` memmaps, one layer at a time. The full
    text set and every layer are not retained together as dense tensors.
    """
    modules = get_decoder_layers(model)
    chosen = resolve_layers(len(modules), layers)
    hidden_size = int(getattr(model.config, "hidden_size"))
    n_texts = len(texts)
    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        layer_dir = output_path / "artifacts" / "activations" / pooling
        layer_dir.mkdir(parents=True, exist_ok=True)
        stores = {
            layer: np.lib.format.open_memmap(
                layer_dir / f"layer_{layer}.npy",
                mode="w+",
                dtype=np.float32,
                shape=(n_texts, hidden_size),
            )
            for layer in chosen
        }
    else:
        stores = {layer: np.zeros((n_texts, hidden_size), dtype=np.float32) for layer in chosen}

    norm_sums = {layer: 0.0 for layer in chosen}
    offset = 0
    model.eval()
    for start in range(0, n_texts, batch_size):
        batch = list(texts[start : start + batch_size])
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        captured: dict[int, torch.Tensor] = {}
        handles = []
        for layer in chosen:
            handles.append(modules[layer].register_forward_hook(_capture_hook(captured, layer)))
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        for handle in handles:
            handle.remove()
        width = input_ids.size(0)
        for layer in chosen:
            pooled = pool_hidden(captured[layer], attention_mask, pooling)
            array = pooled.float().cpu().numpy()
            stores[layer][offset : offset + width] = array
            norm_sums[layer] += float(np.linalg.norm(array, axis=1).sum())
        del captured
        offset += width

    mean_norms = {
        str(layer): (norm_sums[layer] / n_texts if n_texts else 0.0) for layer in chosen
    }
    info = {
        "model_name": (metadata or {}).get("model_name", ""),
        "model_revision": (metadata or {}).get("model_revision", ""),
        "tokenizer_name": (metadata or {}).get("tokenizer_name", ""),
        "dtype": (metadata or {}).get("dtype", ""),
        "device": device,
        "layers": chosen,
        "pooling": pooling,
        "seed": seed,
        "dataset_hash": dataset_hash,
        "n_texts": n_texts,
        "hidden_size": hidden_size,
        "mean_residual_norm": mean_norms,
    }
    if output_path is not None:
        for layer in list(stores):
            stores[layer].flush()
            del stores[layer]
        meta_path = output_path / "artifacts" / "activations" / f"{pooling}_metadata.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        info["paths"] = {
            str(layer): str(output_path / "artifacts" / "activations" / pooling / f"layer_{layer}.npy")
            for layer in chosen
        }
    else:
        info["arrays"] = stores
    return info


def load_layer_matrix(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r")
