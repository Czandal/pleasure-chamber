"""Additive residual steering during generation."""

from __future__ import annotations

import math

import numpy as np
import torch

from pvsteer.models.residual import get_decoder_layers, hidden_from_output, replace_hidden
from pvsteer.safety import ActivationAbort, assert_alpha


class ResidualSteerer:
    def __init__(self, direction: np.ndarray, alpha: float):
        self.direction = torch.tensor(np.asarray(direction, dtype=np.float32))
        self.alpha = float(alpha)
        self.norm_ratios: list[float] = []

    def __call__(self, module, inputs, output):
        del module, inputs
        hidden = hidden_from_output(output)
        if not torch.isfinite(hidden).all():
            raise ActivationAbort("non-finite residual before steering")
        direction = self.direction.to(device=hidden.device, dtype=hidden.dtype)
        last = hidden[:, -1, :]
        denominator = last.norm(dim=-1).mean().clamp(min=1e-8)
        ratio = direction.norm() / denominator
        if not torch.isfinite(ratio):
            raise ActivationAbort("non-finite norm ratio")
        self.norm_ratios.append(float(ratio.item()))
        updated = hidden + (self.alpha * direction)
        if not torch.isfinite(updated).all():
            raise ActivationAbort("non-finite residual after steering")
        return replace_hidden(output, updated)


def continuation_perplexity(model, tokenizer, prompt: str, continuation: str, device: str) -> float | None:
    """Mean token perplexity of the continuation under a forward pass without steering."""
    if not continuation.strip():
        return None
    prompt_ids = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=256)
    full = tokenizer(
        prompt + continuation,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=256,
    )
    input_ids = full["input_ids"].to(device)
    if input_ids.size(1) < 2:
        return None
    prompt_length = min(prompt_ids["input_ids"].size(1), input_ids.size(1) - 1)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=full["attention_mask"].to(device)).logits
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target = input_ids[:, 1:]
    token_log_probs = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
    continuation_log_probs = token_log_probs[:, prompt_length - 1 :]
    if continuation_log_probs.numel() == 0:
        return None
    mean_nll = -float(continuation_log_probs.mean().item())
    if not math.isfinite(mean_nll):
        return None
    return float(math.exp(mean_nll))


def generate_text(
    model,
    tokenizer,
    prompt: str,
    direction: np.ndarray | None,
    alpha: float,
    layer: int,
    max_new_tokens: int,
    max_abs_alpha: float,
    device: str,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    force_json: bool = False,
) -> dict:
    """Generate from one prompt. Hooks are removed even when the step aborts."""
    assert_alpha(alpha, max_abs_alpha)
    steerer = None
    handle = None
    aborted = False
    error = ""
    if direction is not None:
        steerer = ResidualSteerer(direction, alpha)
        layers = get_decoder_layers(model)
        handle = layers[layer].register_forward_hook(steerer)
    try:
        if force_json and getattr(model, "is_mock", False):
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                model(
                    input_ids=encoded["input_ids"].to(device),
                    attention_mask=encoded["attention_mask"].to(device),
                )
            text = '{"action": "B", "confidence": 0.55, "rationale": "simulated ledger choice"}'
            token_ids = tokenizer.encode(text)[:8]
            perplexity = None
        else:
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            prompt_length = input_ids.size(1)
            with torch.no_grad():
                generated = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                )
            new_ids = generated[0, prompt_length:].detach().cpu().tolist()
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            token_ids = new_ids
            if handle is not None:
                handle.remove()
                handle = None
            perplexity = continuation_perplexity(model, tokenizer, prompt, text, device)
    except ActivationAbort as exc:
        aborted = True
        error = str(exc)
        text = ""
        token_ids = []
        perplexity = None
    finally:
        if handle is not None:
            handle.remove()
    return {
        "text": text,
        "token_ids": token_ids,
        "norm_ratios": [] if steerer is None else list(steerer.norm_ratios),
        "aborted": aborted,
        "abort_reason": error,
        "perplexity": perplexity,
        "alpha": alpha,
        "layer": layer,
    }
