"""Tiny causal LM used by tests. It does not download weights."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn


class MockOutput:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class MockBlock(nn.Module):
    def __init__(self, hidden: int, scale: float):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden)
        self.scale = scale
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(hidden))
            self.proj.bias.zero_()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden) * self.scale


class MockCausalLM(nn.Module):
    """Decoder-only stand-in with ``model.layers`` and an unembedding matrix."""

    is_mock = True

    def __init__(self, hidden: int = 32, n_layers: int = 4, vocab: int = 128, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.config = SimpleNamespace(
            hidden_size=hidden,
            vocab_size=vocab,
            model_type="mock",
            num_hidden_layers=n_layers,
        )
        self.embed = nn.Embedding(vocab, hidden)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [MockBlock(hidden, scale=1.0 + 0.05 * index) for index in range(n_layers)]
        )
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.eos_token_id = 1

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs):
        del kwargs
        hidden = self.embed(input_ids)
        positions = torch.arange(hidden.size(1), device=hidden.device, dtype=hidden.dtype)
        hidden = hidden + positions.view(1, -1, 1)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return MockOutput(self.lm_head(hidden))

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 8,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        del top_p, kwargs
        generated = input_ids
        mask = attention_mask
        for _ in range(max_new_tokens):
            output = self(generated, attention_mask=mask)
            logits = output.logits[:, -1, :]
            if do_sample:
                scaled = logits / max(temperature, 1e-6)
                probs = torch.softmax(scaled, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_id], dim=1)
            ones = torch.ones_like(next_id)
            mask = ones if mask is None else torch.cat([mask, ones], dim=1)
            if bool((next_id == self.eos_token_id).all()):
                break
        return generated

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head


class MockTokenizer:
    """Character tokenizer with padding. Special ids are 0 (pad) and 1 (eos)."""

    def __init__(self, vocab: int = 128):
        self.vocab = vocab
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 1
        self.all_special_ids = [0, 1]
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.name_or_path = "mock"

    def _encode_one(self, text: str, max_length: int) -> list[int]:
        ids = [2 + (ord(char) % (self.vocab - 2)) for char in text]
        ids = ids[: max_length - 1] + [self.eos_token_id]
        if not ids:
            ids = [self.eos_token_id]
        return ids

    def __call__(
        self,
        texts,
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 256,
        return_tensors: str | None = "pt",
        add_special_tokens: bool = True,
    ):
        del truncation, add_special_tokens
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self._encode_one(text, max_length) for text in texts]
        width = max(len(row) for row in encoded) if padding else max(len(row) for row in encoded)
        input_ids = []
        attention_mask = []
        for row in encoded:
            pad = width - len(row)
            input_ids.append(row + [self.pad_token_id] * pad)
            attention_mask.append([1] * len(row) + [0] * pad)
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        return self._encode_one(text, max_length=10_000)

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        chars = []
        for token_id in token_ids:
            if skip_special_tokens and int(token_id) in self.all_special_ids:
                continue
            chars.append(chr(32 + (int(token_id) % 95)))
        return "".join(chars)

    def convert_ids_to_tokens(self, index: int) -> str:
        if index in self.all_special_ids:
            return "<pad>" if index == 0 else "<eos>"
        return chr(32 + (int(index) % 95))

    def batch_decode(self, sequences, skip_special_tokens: bool = True) -> list[str]:
        return [self.decode(sequence, skip_special_tokens=skip_special_tokens) for sequence in sequences]
