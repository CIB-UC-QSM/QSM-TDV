from __future__ import annotations

import pytest
import torch

from tdv_qsm.operators.dipole import QSMOperator, build_dipole_kernel, fft3


@pytest.mark.parametrize("shape", [(5, 6, 7), (6, 8, 10)])
def test_dipole_adjoint_identity_for_anisotropic_arbitrary_b0(shape: tuple[int, int, int]) -> None:
    kernel = build_dipole_kernel(
        shape,
        voxel_size_zyx=(0.8, 1.2, 2.3),
        b0_direction_zyx=(1.0, -2.0, 0.5),
    )
    operator = QSMOperator(kernel)
    x = torch.randn(2, 1, *shape)
    y = torch.randn_like(x)
    left = torch.sum(operator.forward(x) * y)
    right = torch.sum(x * operator.adjoint(y))
    relative_error = torch.abs(left - right) / (torch.abs(left) + torch.abs(right) + 1e-8)
    assert relative_error < 1e-5
    assert kernel[0, 0, 0].item() == 0.0


def test_dipole_rejects_zero_b0_direction() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        build_dipole_kernel((5, 5, 5), b0_direction_zyx=(0.0, 0.0, 0.0))


def test_batched_metadata_normalizes_each_b0_and_supports_anisotropic_voxels() -> None:
    kernel = build_dipole_kernel(
        (5, 6, 7),
        voxel_size_zyx=torch.tensor([[1.0, 2.0, 3.0], [0.7, 1.1, 1.8]]),
        b0_direction_zyx=torch.tensor([[0.0, 0.0, 5.0], [2.0, -1.0, 3.0]]),
    )
    assert kernel.shape == (2, 1, 5, 6, 7)
    assert torch.equal(kernel[:, :, 0, 0, 0], torch.zeros(2, 1))
    assert not torch.equal(kernel[0], kernel[1])


def test_fft_contract_casts_half_input_to_complex64() -> None:
    transformed = fft3(torch.ones(1, 1, 3, 4, 5, dtype=torch.float16))
    assert transformed.dtype == torch.complex64


def test_weighted_data_gradient_matches_directional_finite_difference() -> None:
    torch.manual_seed(9)
    shape = (5, 5, 5)
    operator = QSMOperator(build_dipole_kernel(shape))
    chi = torch.randn(1, 1, *shape)
    field = torch.randn_like(chi)
    weight = torch.rand_like(chi)
    direction = torch.randn_like(chi)
    analytic = operator.adjoint(weight.square() * (operator.forward(chi) - field))
    epsilon = 1e-3
    plus = 0.5 * (weight * (operator.forward(chi + epsilon * direction) - field)).square().sum()
    minus = 0.5 * (weight * (operator.forward(chi - epsilon * direction) - field)).square().sum()
    finite_difference = (plus - minus) / (2.0 * epsilon)
    torch.testing.assert_close(
        torch.sum(analytic * direction), finite_difference, atol=2e-3, rtol=2e-3
    )


def test_zero_weight_removes_residual_contribution_and_ones_are_unweighted() -> None:
    shape = (4, 4, 4)
    operator = QSMOperator(torch.ones(shape))
    chi = torch.randn(1, 1, *shape)
    field = torch.randn_like(chi)
    residual = operator.forward(chi) - field
    torch.testing.assert_close(operator.adjoint(residual), operator.adjoint(torch.ones_like(chi) * residual))
    zero_weight = torch.ones_like(chi)
    zero_weight[..., 0, 0, 0] = 0.0
    expected_residual = residual.clone()
    expected_residual[..., 0, 0, 0] = 0.0
    torch.testing.assert_close(
        operator.adjoint(zero_weight.square() * residual),
        operator.adjoint(expected_residual),
    )
