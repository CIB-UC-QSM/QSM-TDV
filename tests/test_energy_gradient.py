from __future__ import annotations

import torch
import pytest

from tdv_qsm.models.energy import TDVEnergy3D


def test_energy_and_image_gradient_have_required_shapes_and_dtypes() -> None:
    torch.manual_seed(3)
    regularizer = TDVEnergy3D(features=1, macro_blocks=1)
    chi = torch.randn(2, 1, 5, 6, 7, requires_grad=True)
    mask = torch.ones_like(chi)
    energy = regularizer.energy(chi, mask)
    grad, = torch.autograd.grad(energy.sum(), chi, create_graph=True)
    assert energy.shape == (2,)
    assert energy.dtype == torch.float32
    assert grad.shape == chi.shape
    assert grad.dtype == torch.float32
    assert torch.isfinite(grad).all()
    means = regularizer.analysis.weight.mean(dim=(1, 2, 3, 4))
    torch.testing.assert_close(means, torch.zeros_like(means), atol=1e-7, rtol=0.0)


def test_energy_directional_derivative_matches_finite_difference() -> None:
    torch.manual_seed(4)
    regularizer = TDVEnergy3D(features=1, macro_blocks=1)
    chi = torch.randn(1, 1, 5, 5, 5, requires_grad=True)
    direction = torch.randn_like(chi)
    direction /= torch.linalg.vector_norm(direction)
    mask = torch.ones_like(chi)
    energy = regularizer.energy(chi, mask).sum()
    gradient, = torch.autograd.grad(energy, chi)
    epsilon = 2e-3
    with torch.no_grad():
        finite_difference = (
            regularizer.energy(chi + epsilon * direction, mask).sum()
            - regularizer.energy(chi - epsilon * direction, mask).sum()
        ) / (2.0 * epsilon)
    analytic = torch.sum(gradient * direction)
    torch.testing.assert_close(analytic, finite_difference, atol=2e-3, rtol=2e-2)


def test_energy_density_and_scalar_energy_are_nonnegative() -> None:
    torch.manual_seed(13)
    regularizer = TDVEnergy3D(features=2, macro_blocks=1)
    chi = torch.randn(3, 1, 5, 6, 7)
    mask = torch.rand_like(chi)

    density = regularizer.energy_density(chi)
    energy = regularizer.energy(chi, mask)

    assert torch.all(density >= 0.0)
    assert torch.all(energy >= 0.0)
    torch.testing.assert_close(
        regularizer.energy(torch.zeros_like(chi), mask),
        torch.zeros(chi.shape[0]),
        atol=0.0,
        rtol=0.0,
    )


def test_energy_rejects_negative_integration_mask() -> None:
    regularizer = TDVEnergy3D(features=1, macro_blocks=1)
    chi = torch.zeros(1, 1, 5, 5, 5)
    mask = torch.ones_like(chi)
    mask[..., 0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        regularizer.energy(chi, mask)


def test_energy_head_uses_configured_small_initialization() -> None:
    scale = 1e-2
    features = 4
    regularizer = TDVEnergy3D(
        features=features,
        macro_blocks=1,
        energy_head_initialization_scale=scale,
    )
    bound = scale / features**0.5
    assert torch.all(regularizer.energy_head.weight.abs() <= bound)
    assert torch.any(regularizer.energy_head.weight != 0.0)


def test_small_head_initialization_reduces_initial_energy_quadratically() -> None:
    torch.manual_seed(17)
    reference = TDVEnergy3D(
        features=2,
        macro_blocks=1,
        energy_head_initialization_scale=1.0,
    )
    torch.manual_seed(17)
    reduced = TDVEnergy3D(
        features=2,
        macro_blocks=1,
        energy_head_initialization_scale=1e-2,
    )
    chi = torch.randn(1, 1, 5, 5, 5)
    mask = torch.ones_like(chi)
    reference_energy = reference.energy(chi, mask)
    reduced_energy = reduced.energy(chi, mask)
    torch.testing.assert_close(
        reduced_energy,
        1e-4 * reference_energy,
        rtol=2e-5,
        atol=1e-10,
    )
