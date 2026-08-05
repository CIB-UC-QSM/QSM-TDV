from __future__ import annotations

import torch

from tdv_qsm.models.activation import student_t_pair
from tdv_qsm.models.blocks import MacroBlock3D, MicroBlock3D
from tdv_qsm.models.energy import TDVEnergy3D


def test_student_t_pair_uses_alpha_one_in_float32() -> None:
    x = torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float16)
    value, derivative = student_t_pair(x)
    assert value.dtype == torch.float32
    assert derivative.dtype == torch.float32
    torch.testing.assert_close(value, 0.5 * torch.log1p(x.float().square()))
    torch.testing.assert_close(derivative, x.float() / (1.0 + x.float().square()))


def test_three_scale_macro_has_five_micro_blocks_and_preserves_state() -> None:
    block = MacroBlock3D(num_features=2, num_scales=3)
    assert sum(isinstance(module, MicroBlock3D) for module in block.modules()) == 5
    first = block([torch.randn(1, 2, 7, 8, 9), None, None])
    assert len(first) == 3 and all(value is not None for value in first)
    second = block(first)
    assert [value.shape for value in second] == [value.shape for value in first]


def test_micro_block_manual_jacobian_transpose_matches_autograd() -> None:
    torch.manual_seed(4)
    block = MicroBlock3D(2).double()
    x = torch.randn(1, 2, 4, 5, 6, dtype=torch.float64, requires_grad=True)
    q = torch.randn_like(x)
    output = block(x)
    reference, = torch.autograd.grad(torch.sum(output * q), x, create_graph=True)
    manual = block.jacobian_transpose(x, q)
    torch.testing.assert_close(manual, reference, atol=2e-6, rtol=2e-6)


def test_macro_block_manual_jacobian_transpose_matches_autograd_with_existing_scales() -> None:
    torch.manual_seed(14)
    block = MacroBlock3D(num_features=1, num_scales=3).double()
    states = [
        torch.randn(1, 1, 5, 6, 7, dtype=torch.float64, requires_grad=True),
        torch.randn(1, 1, 3, 3, 4, dtype=torch.float64, requires_grad=True),
        torch.randn(1, 1, 2, 2, 2, dtype=torch.float64, requires_grad=True),
    ]
    outputs, cache = block.forward_with_cache(states)
    output_adjoints = [torch.randn_like(output) for output in outputs]
    reference = torch.autograd.grad(
        sum(torch.sum(output * adjoint) for output, adjoint in zip(outputs, output_adjoints)),
        states,
        create_graph=True,
    )
    manual = block.jacobian_transpose(cache, output_adjoints)
    for manual_value, reference_value in zip(manual, reference):
        assert manual_value is not None
        torch.testing.assert_close(manual_value, reference_value, atol=3e-6, rtol=3e-6)


def test_tdv_energy_density_sum_and_manual_force_contract() -> None:
    torch.manual_seed(5)
    regularizer = TDVEnergy3D(num_features=2, num_macro_blocks=2).double()
    chi = torch.randn(1, 1, 5, 6, 7, dtype=torch.float64, requires_grad=True)
    density = regularizer.energy_density(chi)
    energy = regularizer.energy(chi)
    assert density.shape == chi.shape
    assert energy.shape == (1,)
    assert energy.dtype == torch.float32
    torch.testing.assert_close(energy, density.float().flatten(1).sum(dim=1))
    torch.testing.assert_close(
        regularizer.force(chi),
        regularizer.force_autograd_reference(chi),
        atol=3e-5,
        rtol=3e-5,
    )


def test_analysis_kernel_initialization_and_projection_constraints() -> None:
    regularizer = TDVEnergy3D(num_features=3, num_macro_blocks=1)
    weight = regularizer.analysis.weight
    means = weight.mean(dim=(1, 2, 3, 4))
    norms = torch.linalg.vector_norm(weight, dim=(1, 2, 3, 4))
    torch.testing.assert_close(means, torch.zeros_like(means), atol=1e-7, rtol=0.0)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-6, rtol=0.0)
    with torch.no_grad():
        weight.mul_(4.0).add_(2.0)
    regularizer.project_analysis_kernel_()
    means = weight.mean(dim=(1, 2, 3, 4))
    norms = torch.linalg.vector_norm(weight, dim=(1, 2, 3, 4))
    torch.testing.assert_close(means, torch.zeros_like(means), atol=2e-7, rtol=0.0)
    assert torch.all(norms <= 1.0 + 1e-6)
