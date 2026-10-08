"""Locate the residual stream of a decoder-only causal LM."""

from __future__ import annotations

import torch
from torch import nn


class UnsupportedArchitectureError(RuntimeError):
    """Raised when no single decoder-layer list can be found."""


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Return the decoder blocks whose outputs are the residual stream.

    Supported layouts are the ones used by Llama, Qwen, Mistral, Gemma, Phi,
    GPT-2, and GPT-NeoX style modules. Encoder-decoder and multi-stream
    models are rejected with an explicit error.
    """
    inner = getattr(model, "model", None)
    if inner is not None and isinstance(getattr(inner, "layers", None), nn.ModuleList):
        return inner.layers
    transformer = getattr(model, "transformer", None)
    if transformer is not None and isinstance(getattr(transformer, "h", None), nn.ModuleList):
        return transformer.h
    gpt_neox = getattr(model, "gpt_neox", None)
    if gpt_neox is not None and isinstance(getattr(gpt_neox, "layers", None), nn.ModuleList):
        return gpt_neox.layers
    raise UnsupportedArchitectureError(
        "No single decoder-layer residual stream was found. Supported layouts: "
        "model.model.layers, model.transformer.h, model.gpt_neox.layers."
    )


def hidden_from_output(output: object) -> torch.Tensor:
    if isinstance(output, tuple):
        hidden = output[0]
    elif hasattr(output, "last_hidden_state"):
        hidden = output.last_hidden_state
    else:
        hidden = output
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise TypeError("decoder layer output is not a (batch, sequence, hidden) tensor")
    return hidden


def replace_hidden(output: object, hidden: torch.Tensor) -> object:
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if hasattr(output, "last_hidden_state"):
        output.last_hidden_state = hidden
        return output
    return hidden


def auto_layers(n_layers: int) -> list[int]:
    if n_layers <= 0:
        raise ValueError("model has no decoder layers")
    if n_layers <= 4:
        return list(range(n_layers))
    picks = [int(round(fraction * (n_layers - 1))) for fraction in (0.25, 0.5, 0.75, 1.0)]
    return sorted(set(picks))


def resolve_layers(n_layers: int, candidate_layers: object) -> list[int]:
    if candidate_layers == "auto":
        chosen = auto_layers(n_layers)
    else:
        chosen = [int(layer) for layer in candidate_layers]  # type: ignore[union-attr]
    for layer in chosen:
        if layer < 0 or layer >= n_layers:
            raise ValueError(f"layer {layer} is outside 0..{n_layers - 1}")
    return chosen


def pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor | None, pooling: str) -> torch.Tensor:
    """Pool a residual tensor with an attention mask.

    ``final`` reads the last non-padding token. ``mean`` is the mask-weighted
    mean over tokens. Padding positions do not enter either statistic.
    """
    if attention_mask is None:
        attention_mask = torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.long)
    mask = attention_mask.to(device=hidden.device).long()
    if pooling == "final":
        index = (mask.sum(dim=1) - 1).clamp(min=0)
        batch = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch, index]
    if pooling == "mean":
        weights = mask.unsqueeze(-1).to(dtype=hidden.dtype)
        denom = weights.sum(dim=1).clamp(min=1.0)
        return (hidden * weights).sum(dim=1) / denom
    raise ValueError(f"unknown pooling mode {pooling}")
