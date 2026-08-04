from __future__ import annotations

import torch

from tdv_qsm.models.energy import TDVEnergy3D
from tdv_qsm.models.explicit_tdv import ExplicitTDVQSM3D
from tdv_qsm.operators.dipole import DipoleOperator3D, build_dipole_kernel


def test_explicit_step_matches_weight_squared_data_gradient() -> None:
    """The first reconstruction update is exactly explicit Euler."""

    shape = (5, 6, 7)
    kernel = build_dipole_kernel(shape)
    operator = DipoleOperator3D(kernel)
    regularizer = TDVEnergy3D(features=1, macro_blocks=1)
    with torch.no_grad():
        for parameter in regularizer.parameters():
            parameter.zero_()

    model = ExplicitTDVQSM3D(
        regularizer=regularizer,
        num_steps=1,
        maximum_time=0.2,
    )
    with torch.no_grad():
        model.raw_time.fill_(0.0)

    initial = torch.randn(1, 1, *shape)
    local_field = torch.randn_like(initial)
    weight = torch.rand_like(initial)
    mask = torch.ones_like(initial)

    output = model(local_field, mask, kernel, weight, initial)

    tau = 0.1
    expected = initial - tau * operator.adjoint(
        weight.square() * (operator.forward(initial) - local_field)
    )
    torch.testing.assert_close(output.susceptibility, expected, atol=2e-6, rtol=2e-6)
    assert output.susceptibility.dtype == torch.float32


def test_physical_energy_decreases_with_zero_regularizer_and_small_step() -> None:
    shape = (5, 5, 5)
    kernel = build_dipole_kernel(shape)
    operator = DipoleOperator3D(kernel)
    regularizer = TDVEnergy3D(features=1, macro_blocks=1)
    with torch.no_grad():
        for parameter in regularizer.parameters():
            parameter.zero_()
    model = ExplicitTDVQSM3D(regularizer, num_steps=1, maximum_time=0.1)
    with torch.no_grad():
        model.raw_time.fill_(0.0)
    initial = torch.randn(1, 1, *shape)
    local_field = torch.randn_like(initial)
    mask = torch.ones_like(initial)
    output = model(local_field, mask, kernel, torch.ones_like(initial), initial)
    before = 0.5 * (operator.forward(initial) - local_field).square().sum()
    after = 0.5 * (operator.forward(output.susceptibility) - local_field).square().sum()
    assert after < before


def test_zero_physical_operator_reduces_to_gradient_descent_on_tdv_energy() -> None:
    torch.manual_seed(11)
    shape = (5, 5, 5)
    zero_kernel = torch.zeros(shape)
    regularizer = TDVEnergy3D(features=1, macro_blocks=1)
    model = ExplicitTDVQSM3D(regularizer, num_steps=1, maximum_time=0.2)
    with torch.no_grad():
        model.raw_time.fill_(0.0)
    initial = torch.randn(1, 1, *shape, requires_grad=True)
    mask = torch.ones_like(initial)
    direct_energy = regularizer.energy(initial, mask)
    direct_gradient, = torch.autograd.grad(direct_energy.sum(), initial)
    output = model(torch.zeros_like(initial), mask, zero_kernel, torch.ones_like(initial), initial)
    torch.testing.assert_close(
        output.susceptibility,
        initial - 0.1 * direct_gradient,
        atol=2e-6,
        rtol=2e-6,
    )
