from __future__ import annotations

import pytest
import torch

from tdv_qsm.operators.convolution import (
    AdjointConv3d,
    BinomialDownsample3d,
    ScaledConv3d,
    SymmetricPad3d,
)


def _relative_adjoint_error(
    forward_value: torch.Tensor,
    test_value: torch.Tensor,
    input_value: torch.Tensor,
    adjoint_value: torch.Tensor,
) -> torch.Tensor:
    left = torch.sum(forward_value * test_value)
    right = torch.sum(input_value * adjoint_value)
    return torch.abs(left - right) / (torch.abs(left) + torch.abs(right) + 1e-12)


@pytest.mark.parametrize("shape", [(1, 2, 5, 6, 7), (2, 2, 6, 7, 8)])
def test_symmetric_padding_has_an_exact_adjoint(shape: tuple[int, ...]) -> None:
    torch.manual_seed(1)
    padding = SymmetricPad3d((1, 2, 1))
    x = torch.randn(*shape, dtype=torch.float64)
    y = torch.randn_like(padding(x))
    error = _relative_adjoint_error(padding(x), y, x, padding.adjoint(y, x.shape))
    assert error < 1e-12


@pytest.mark.parametrize("shape", [(1, 2, 5, 6, 7), (2, 2, 6, 8, 10)])
def test_learned_convolution_has_an_exact_adjoint(shape: tuple[int, ...]) -> None:
    torch.manual_seed(2)
    operator = AdjointConv3d(2, 3, kernel_size=3).double()
    x = torch.randn(*shape, dtype=torch.float64)
    y = torch.randn_like(operator(x))
    error = _relative_adjoint_error(operator(x), y, x, operator.adjoint(y, x.shape))
    assert error < 1e-11


@pytest.mark.parametrize("shape", [(1, 2, 5, 6, 7), (1, 2, 6, 8, 10)])
def test_binomial_downsample_adjoint_recovers_odd_and_even_shapes(
    shape: tuple[int, ...],
) -> None:
    torch.manual_seed(3)
    operator = BinomialDownsample3d(channels=2).double()
    x = torch.randn(*shape, dtype=torch.float64)
    y = torch.randn_like(operator(x))
    adjoint = operator.adjoint(y, x.shape)
    assert adjoint.shape == x.shape
    error = _relative_adjoint_error(operator(x), y, x, adjoint)
    assert error < 1e-11


@pytest.mark.parametrize("shape", [(1, 2, 5, 6, 7), (1, 2, 6, 8, 10)])
def test_learned_antialiased_scaled_convolution_has_exact_adjoint(
    shape: tuple[int, ...],
) -> None:
    torch.manual_seed(10)
    operator = ScaledConv3d(2, 3, kernel_size=3).double()
    x = torch.randn(*shape, dtype=torch.float64)
    y = torch.randn_like(operator(x))
    adjoint = operator.adjoint(y, x.shape)
    assert adjoint.shape == x.shape
    error = _relative_adjoint_error(operator(x), y, x, adjoint)
    assert error < 1e-11
