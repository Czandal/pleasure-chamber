import numpy as np
import torch
from torch import nn

from pvsteer.safety import ActivationAbort, SteeringBoundError, assert_alpha
from pvsteer.steering.generate import ResidualSteerer


def test_coefficient_bounds():
    assert assert_alpha(1.5, 2.0) == 1.5
    try:
        assert_alpha(2.5, 2.0)
    except SteeringBoundError:
        return
    raise AssertionError("expected bound error")


def test_nan_activation_aborts():
    layer = nn.Identity()
    steerer = ResidualSteerer(np.ones(4), alpha=1.0)
    handle = layer.register_forward_hook(steerer)
    try:
        with torch.no_grad():
            layer(torch.tensor([[[float("nan"), 0.0, 0.0, 0.0]]]))
    except ActivationAbort:
        return
    finally:
        handle.remove()
    raise AssertionError("expected abort")
