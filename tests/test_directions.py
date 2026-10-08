import numpy as np

from pvsteer.directions.fit import (
    cosine_matrix,
    denoise_direction,
    difference_direction,
    l2_normalize,
)


def test_normalization_and_cosine():
    vector = l2_normalize(np.array([3.0, 4.0]))
    assert abs(np.linalg.norm(vector) - 1.0) < 1e-8
    frame = cosine_matrix({"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])})
    assert abs(frame.loc["a", "a"] - 1.0) < 1e-8
    assert abs(frame.loc["a", "b"]) < 1e-8


def test_difference_and_denoising():
    positive = np.ones((6, 4))
    control = np.zeros((6, 4))
    direction = difference_direction(positive, control)
    assert abs(np.linalg.norm(direction) - 1.0) < 1e-8
    assert direction[0] > 0
    rng = np.random.default_rng(0)
    controls = np.zeros((50, 6))
    controls[:, :2] = rng.normal(size=(50, 2))
    vector = np.zeros(6)
    vector[:2] = 1.0
    vector[5] = 4.0
    denoised, components, explained = denoise_direction(vector, controls, variance=0.99)
    assert components >= 1
    assert explained >= 0.99 or components == controls.shape[0]
    assert abs(denoised[5]) > 0.9
    assert abs(np.linalg.norm(denoised) - 1.0) < 1e-8
