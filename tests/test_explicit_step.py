from __future__ import annotations

import copy

import torch
from torch import nn

from tdv_qsm.models.energy import TDVEnergy3D
from tdv_qsm.models.explicit_tdv import ExplicitTDVQSM3D
from tdv_qsm.operators.dipole import QSMOperator, build_dipole_kernel


class QuadraticRegularizer(nn.Module):
    def force(self, chi: torch.Tensor) -> torch.Tensor:
        return chi.float()


def test_one_step_matches_separate_regularizer_and_weight_squared_data_coefficients() -> None:
    shape = (5, 6, 7)
    kernel = build_dipole_kernel(shape)
    operator = QSMOperator(kernel)
    model = ExplicitTDVQSM3D(
        QuadraticRegularizer(),  # type: ignore[arg-type]
        operator,
        num_steps=2,
        fixed_T=0.2,
        fixed_lambda=0.6,
        mask_state_each_step=False,
    )
    initial = torch.randn(1, 1, *shape)
    local_field = torch.randn_like(initial)
    weight = torch.rand_like(initial)
    mask = torch.ones_like(initial)
    output = model(local_field, mask, None, weight, initial, return_states=True)
    expected = initial
    for _ in range(2):
        data_force = operator.adjoint(
            weight.square() * (operator.forward(expected) - local_field)
        )
        expected = expected - 0.1 * expected - 0.6 * data_force
    torch.testing.assert_close(output.susceptibility, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(output.data_step, torch.tensor(0.6))
    assert output.states is not None and len(output.states) == 3


def test_zero_coefficients_disable_their_respective_forces() -> None:
    shape = (4, 4, 4)
    kernel = build_dipole_kernel(shape)
    initial = torch.randn(1, 1, *shape)
    field = torch.randn_like(initial)
    weight = torch.ones_like(initial)
    mask = torch.ones_like(initial)
    regularizer_only = ExplicitTDVQSM3D(
        QuadraticRegularizer(),  # type: ignore[arg-type]
        QSMOperator(kernel),
        fixed_T=0.2,
        fixed_lambda=0.0,
        mask_state_each_step=False,
    )
    data_only = ExplicitTDVQSM3D(
        QuadraticRegularizer(),  # type: ignore[arg-type]
        QSMOperator(kernel),
        fixed_T=0.0,
        fixed_lambda=0.2,
        mask_state_each_step=False,
    )
    torch.testing.assert_close(
        regularizer_only(field, mask, None, weight, initial).susceptibility,
        initial - 0.2 * initial,
    )
    expected_data = initial - 0.2 * QSMOperator(kernel).adjoint(
        QSMOperator(kernel).forward(initial) - field
    )
    torch.testing.assert_close(
        data_only(field, mask, None, weight, initial).susceptibility,
        expected_data,
        atol=2e-6,
        rtol=2e-6,
    )


def test_masking_the_state_is_configurable() -> None:
    shape = (4, 4, 4)
    kernel = torch.zeros(shape)
    initial = torch.ones(1, 1, *shape)
    mask = torch.ones_like(initial)
    mask[..., 0, 0, 0] = 0.0
    arguments = (torch.zeros_like(initial), mask, kernel, torch.ones_like(initial), initial)
    masked = ExplicitTDVQSM3D(
        QuadraticRegularizer(),  # type: ignore[arg-type]
        fixed_T=0.0,
        fixed_lambda=0.0,
        mask_state_each_step=True,
    )(*arguments)
    unmasked = ExplicitTDVQSM3D(
        QuadraticRegularizer(),  # type: ignore[arg-type]
        fixed_T=0.0,
        fixed_lambda=0.0,
        mask_state_each_step=False,
    )(*arguments)
    assert masked.susceptibility[..., 0, 0, 0].item() == 0.0
    assert unmasked.susceptibility[..., 0, 0, 0].item() == 1.0


def test_checkpointed_and_noncheckpointed_force_outputs_and_gradients_match() -> None:
    torch.manual_seed(12)
    shape = (4, 4, 4)
    reference = ExplicitTDVQSM3D(
        TDVEnergy3D(num_features=1, num_macro_blocks=1),
        QSMOperator(build_dipole_kernel(shape)),
        fixed_T=0.05,
        fixed_lambda=0.0,
        checkpoint_force=False,
    )
    checkpointed = copy.deepcopy(reference)
    checkpointed.checkpoint_force = True
    initial = torch.randn(1, 1, *shape)
    field = torch.zeros_like(initial)
    mask = torch.ones_like(initial)
    weight = torch.ones_like(initial)
    reference_output = reference(field, mask, None, weight, initial).susceptibility
    checkpoint_output = checkpointed(field, mask, None, weight, initial).susceptibility
    torch.testing.assert_close(checkpoint_output, reference_output, atol=1e-6, rtol=1e-6)
    reference_output.square().mean().backward()
    checkpoint_output.square().mean().backward()
    for reference_parameter, checkpoint_parameter in zip(
        reference.parameters(), checkpointed.parameters()
    ):
        torch.testing.assert_close(
            checkpoint_parameter.grad,
            reference_parameter.grad,
            atol=2e-6,
            rtol=2e-5,
        )
