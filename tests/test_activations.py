import torch

from pvsteer.activations.extract import extract_activations
from pvsteer.models.mock import MockCausalLM, MockTokenizer
from pvsteer.models.residual import UnsupportedArchitectureError, get_decoder_layers, pool_hidden


def test_unsupported_architecture():
    model = torch.nn.Linear(3, 3)
    try:
        get_decoder_layers(model)
    except UnsupportedArchitectureError as exc:
        assert "model.model.layers" in str(exc)
    else:
        raise AssertionError("expected unsupported architecture")


def test_padding_and_shapes(tmp_path):
    model = MockCausalLM(hidden=8, n_layers=2, vocab=64, seed=1)
    model.embed.weight.data.zero_()
    tokenizer = MockTokenizer(vocab=64)
    texts = ["ab", "abcdef"]
    info = extract_activations(
        model,
        tokenizer,
        texts,
        layers=[0],
        pooling="final",
        batch_size=2,
        max_length=32,
        output_dir=tmp_path,
        dataset_hash="abc",
        seed=1,
        metadata={"model_name": "mock", "model_revision": "mock-0", "tokenizer_name": "mock", "dtype": "float32"},
    )
    array = info["arrays"][0] if "arrays" in info else None
    if array is None:
        import numpy as np

        array = np.load(tmp_path / "artifacts" / "activations" / "final" / "layer_0.npy")
    assert array.shape[0] == 2
    assert array.shape[1] == 8
    encoded = tokenizer(texts, padding=True, return_tensors="pt")
    lengths = encoded["attention_mask"].sum(dim=1) - 1
    assert array[0, 0] == float(lengths[0])
    assert array[1, 0] == float(lengths[1])
    assert lengths[0] != lengths[1]
    hidden = torch.zeros(1, 4, 3)
    hidden[0, :, 0] = torch.arange(4)
    mask = torch.tensor([[1, 1, 1, 0]])
    pooled = pool_hidden(hidden, mask, "final")
    assert float(pooled[0, 0]) == 2.0
    mean = pool_hidden(hidden, mask, "mean")
    assert abs(float(mean[0, 0]) - 1.0) < 1e-5
    for key in ("model_name", "model_revision", "tokenizer_name", "dtype", "device", "layers", "pooling", "seed", "dataset_hash"):
        assert key in info
