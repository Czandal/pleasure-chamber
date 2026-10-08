"""Direction-to-token scores.

These scores are vocabulary whose logits change when the direction is
projected through the output matrix. They are not a definition of the direction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch


def output_matrix(model: torch.nn.Module) -> np.ndarray:
    embedding = None
    if hasattr(model, "get_output_embeddings"):
        embedding = model.get_output_embeddings()
    if embedding is None or not hasattr(embedding, "weight"):
        raise RuntimeError("model has no output embedding matrix")
    return embedding.weight.detach().float().cpu().numpy()


def _special_ids(tokenizer: Any) -> set[int]:
    ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    for name in ("pad_token_id", "eos_token_id", "bos_token_id", "unk_token_id"):
        value = getattr(tokenizer, name, None)
        if isinstance(value, int):
            ids.add(value)
    return ids


def _dataset_frequency(tokenizer: Any, texts: list[str], vocab: int) -> np.ndarray:
    counts = np.zeros(vocab, dtype=np.int64)
    for text in texts:
        encoded = tokenizer(text, padding=False, truncation=True, max_length=256, return_tensors=None)
        ids = encoded["input_ids"]
        if ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        for token_id in ids:
            if 0 <= int(token_id) < vocab:
                counts[int(token_id)] += 1
    return counts


def _keep_token(token: str, token_id: int, specials: set[int], count: int, cap: int) -> bool:
    if token_id in specials:
        return False
    if count > cap:
        return False
    stripped = token.strip()
    if not stripped or len(stripped) > 20:
        return False
    if not any(character.isalnum() for character in stripped):
        return False
    return True


def unembedding_readout(
    model: torch.nn.Module,
    tokenizer: Any,
    directions: dict[str, np.ndarray],
    texts: list[str],
    top_k: int = 15,
) -> pd.DataFrame:
    matrix = output_matrix(model)
    vocab = matrix.shape[0]
    specials = _special_ids(tokenizer)
    counts = _dataset_frequency(tokenizer, texts, vocab)
    positive_counts = counts[counts > 0]
    cap = int(np.quantile(positive_counts, 0.99)) if len(positive_counts) else 0
    rows = []
    for name, direction in directions.items():
        vector = np.asarray(direction, dtype=np.float64)
        if vector.shape[0] != matrix.shape[1]:
            raise ValueError(f"direction {name} does not match the unembedding width")
        scores = matrix @ vector
        ranked = np.argsort(scores)
        kept_high = []
        kept_low = []
        for token_id in ranked[::-1]:
            token = tokenizer.convert_ids_to_tokens(int(token_id))
            if _keep_token(token, int(token_id), specials, int(counts[token_id]), cap):
                kept_high.append(int(token_id))
            if len(kept_high) >= top_k:
                break
        for token_id in ranked:
            token = tokenizer.convert_ids_to_tokens(int(token_id))
            if _keep_token(token, int(token_id), specials, int(counts[token_id]), cap):
                kept_low.append(int(token_id))
            if len(kept_low) >= top_k:
                break
        for pole, ids in (("positive", kept_high), ("negative", kept_low)):
            for rank, token_id in enumerate(ids, start=1):
                token = tokenizer.decode([token_id], skip_special_tokens=True)
                rows.append(
                    {
                        "direction": name,
                        "pole": pole,
                        "rank": rank,
                        "token_id": token_id,
                        "token": token,
                        "token_piece": tokenizer.convert_ids_to_tokens(token_id),
                        "score": float(scores[token_id]),
                        "dataset_count": int(counts[token_id]),
                    }
                )
    return pd.DataFrame(rows)
