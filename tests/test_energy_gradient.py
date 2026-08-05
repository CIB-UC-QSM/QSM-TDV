from __future__ import annotations

import torch

from tdv_qsm.models.energy import TDVEnergy3D


def test_energy_and_force_have_required_shapes_and_dtypes() -> None:
    torch.manual_seed(3)
    regularizer = TDVEnergy3D(num_features=2, num_macro_blocks=1)
    chi = torch.randn(2, 1, 5, 6, 7, requires_grad=True)
    density = regularizer.energy_density(chi)
    energy = regularizer.energy(chi)
    force = regularizer.force(chi)
    assert density.shape == chi.shape
    assert energy.shape == (2,) and energy.dtype == torch.float32
    assert force.shape == chi.shape and force.dtype == torch.float32
    torch.testing.assert_close(energy, density.float().flatten(1).sum(dim=1))
    assert torch.isfinite(force).all()


def test_source_directional_scale_derivative() -> None:
    torch.manual_seed(4)
    regularizer = TDVEnergy3D(num_features=2, num_macro_blocks=1).double()
    x = torch.randn(1, 1, 5, 5, 5, dtype=torch.float64)
    scale = 0.7
    epsilon = 2e-3
    analytic = torch.sum(x.float() * regularizer.force(scale * x))
    with torch.no_grad():
        finite_difference = (
            regularizer.energy((scale + epsilon) * x).sum()
            - regularizer.energy((scale - epsilon) * x).sum()
        ) / (2.0 * epsilon)
    torch.testing.assert_close(analytic, finite_difference, atol=3e-3, rtol=3e-2)


def test_manual_force_matches_float64_autograd_reference() -> None:
    torch.manual_seed(6)
    regularizer = TDVEnergy3D(num_features=2, num_macro_blocks=2).double()
    chi = torch.randn(1, 1, 4, 5, 6, dtype=torch.float64, requires_grad=True)
    manual = regularizer.force(chi)
    reference = regularizer.force_autograd_reference(chi)
    relative_error = torch.linalg.vector_norm(manual - reference) / torch.linalg.vector_norm(
        reference
    ).clamp_min(1e-12)
    assert relative_error < 3e-5


def test_force_remains_differentiable_to_every_regularizer_parameter() -> None:
    torch.manual_seed(8)
    regularizer = TDVEnergy3D(num_features=2, num_macro_blocks=1)
    chi = torch.randn(1, 1, 4, 4, 4)
    loss = regularizer.force(chi).square().mean()
    loss.backward()
    for name, parameter in regularizer.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
