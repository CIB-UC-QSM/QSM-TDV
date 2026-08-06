"""Explicit variational-network dynamics for the 3-D QSM extension."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from tdv_qsm.models.energy import TDVEnergy3D
from tdv_qsm.operators.dipole import QSMOperator


@dataclass(frozen=True)
class TDVOutput:
    """Reconstruction and opt-in state history plus compact diagnostics."""

    susceptibility: torch.Tensor
    predicted_field: torch.Tensor
    states: tuple[torch.Tensor, ...] | None
    data_gradient_norm: torch.Tensor
    regularizer_gradient_norm: torch.Tensor
    state_norm: torch.Tensor
    time: torch.Tensor
    data_coefficient: torch.Tensor
    regularizer_step: torch.Tensor
    data_step: torch.Tensor

    @property
    def step_size(self) -> torch.Tensor:
        """Compatibility name for the regularizer step ``T/S``."""

        return self.regularizer_step


def _require_image(name: str, value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if value.ndim != 5 or value.shape[1] != 1:
        raise ValueError(f"{name} must be [B, 1, Z, Y, X], got {tuple(value.shape)}.")
    if value.shape != reference.shape:
        raise ValueError(
            f"{name} must match local_field {tuple(reference.shape)}, got {tuple(value.shape)}."
        )
    value = value.float()
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values.")
    return value


def validate_magnitude_weight(
    magnitude_weight: torch.Tensor,
    local_field: torch.Tensor,
) -> torch.Tensor:
    """Validate and broadcast the stored diagonal of real nonnegative ``W``."""

    if magnitude_weight.ndim != 5 or magnitude_weight.shape[1] != 1:
        raise ValueError("magnitude_weight must have shape [B, 1, Z, Y, X].")
    try:
        weight = torch.broadcast_to(magnitude_weight.float(), local_field.shape)
    except RuntimeError as error:
        raise ValueError("magnitude_weight must broadcast to local_field.") from error
    if not torch.isfinite(weight).all():
        raise ValueError("Magnitude weights must be finite.")
    if torch.any(weight < 0.0):
        raise ValueError("Magnitude weights must be nonnegative.")
    return weight


class ExplicitTDVQSM3D(nn.Module):
    """Unroll the source-style explicit branch with QSM data physics.

    The QSM operator, magnitude weights, 3-D state, masking, and float16 AMP
    are project extensions.  The explicit update keeps the source branch's
    separate regularizer and data coefficients.
    """

    def __init__(
        self,
        regularizer: TDVEnergy3D,
        operator: QSMOperator | None = None,
        *,
        num_steps: int = 1,
        maximum_time: float = 0.25,
        maximum_lambda: float = 1.0,
        fixed_T: float | None = None,
        fixed_lambda: float | None = None,
        initial_raw_T: float = 2.0,
        initial_raw_lambda: float = 0.0,
        initial_raw_time: float | None = None,
        mask_state_each_step: bool = True,
        checkpoint_force: bool = False,
    ) -> None:
        super().__init__()
        if num_steps < 1:
            raise ValueError("num_steps must be at least one.")
        if maximum_time <= 0.0 or maximum_lambda <= 0.0:
            raise ValueError("maximum_time and maximum_lambda must be positive.")
        if fixed_T is not None and fixed_T < 0.0:
            raise ValueError("fixed_T must be nonnegative.")
        if fixed_lambda is not None and fixed_lambda < 0.0:
            raise ValueError("fixed_lambda must be nonnegative.")
        if initial_raw_time is not None:
            initial_raw_T = initial_raw_time
        self.regularizer = regularizer
        self.operator = QSMOperator() if operator is None else operator
        self.num_steps = int(num_steps)
        self.maximum_time = float(maximum_time)
        self.maximum_lambda = float(maximum_lambda)
        self.mask_state_each_step = bool(mask_state_each_step)
        self.checkpoint_force = bool(checkpoint_force)
        if fixed_T is None:
            self.raw_T = nn.Parameter(torch.tensor(float(initial_raw_T), dtype=torch.float32))
            self.register_buffer("fixed_T", None)
        else:
            self.register_parameter("raw_T", None)
            self.register_buffer("fixed_T", torch.tensor(float(fixed_T), dtype=torch.float32))
        if fixed_lambda is None:
            self.raw_lambda = nn.Parameter(
                torch.tensor(float(initial_raw_lambda), dtype=torch.float32)
            )
            self.register_buffer("fixed_lambda", None)
        else:
            self.register_parameter("raw_lambda", None)
            self.register_buffer(
                "fixed_lambda",
                torch.tensor(float(fixed_lambda), dtype=torch.float32),
            )

    @property
    def raw_time(self) -> nn.Parameter:
        """Compatibility access to the learned ``raw_T`` parameter."""

        if self.raw_T is None:
            raise AttributeError("This model uses fixed_T and has no raw_T parameter.")
        return self.raw_T

    def coefficients(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the nonnegative bounded/fixed coefficients once per call."""

        if self.raw_T is None:
            assert self.fixed_T is not None
            total_time = self.fixed_T.float()
        else:
            total_time = self.maximum_time * torch.sigmoid(self.raw_T.float())
        if self.raw_lambda is None:
            assert self.fixed_lambda is not None
            data_coefficient = self.fixed_lambda.float()
        else:
            data_coefficient = self.maximum_lambda * torch.sigmoid(self.raw_lambda.float())
        return total_time, data_coefficient

    @staticmethod
    def _norm(value: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(value.float().flatten(1), ord=2, dim=1)

    def data_force(
        self,
        chi: torch.Tensor,
        local_field: torch.Tensor,
        magnitude_weight: torch.Tensor,
        dipole_kernel: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return exactly ``A^H W^H W (A chi - b)`` in float32."""

        magnitude_weight = validate_magnitude_weight(magnitude_weight, local_field)
        return self._data_force_validated(
            chi,
            local_field,
            magnitude_weight,
            dipole_kernel,
        )

    def _data_force_validated(
        self,
        chi: torch.Tensor,
        local_field: torch.Tensor,
        magnitude_weight: torch.Tensor,
        dipole_kernel: torch.Tensor | None,
    ) -> torch.Tensor:
        predicted_field = self.operator.forward(chi.float(), dipole_kernel)
        field_residual = predicted_field - local_field.float()
        weighted_residual = magnitude_weight.float() * field_residual
        normal_weighted_residual = magnitude_weight.float() * weighted_residual
        return self.operator.adjoint(normal_weighted_residual, dipole_kernel).float()

    def _regularizer_force(self, chi: torch.Tensor) -> torch.Tensor:
        if self.checkpoint_force and torch.is_grad_enabled():
            return checkpoint(
                self.regularizer.force,
                chi,
                use_reentrant=False,
            )
        return self.regularizer.force(chi)

    def forward(
        self,
        local_field: torch.Tensor,
        brain_mask: torch.Tensor,
        dipole_kernel: torch.Tensor | None,
        magnitude_weight: torch.Tensor,
        initial: torch.Tensor,
        *,
        return_states: bool = False,
    ) -> TDVOutput:
        if local_field.ndim != 5 or local_field.shape[1] != 1:
            raise ValueError(
                "local_field must have shape [B, 1, Z, Y, X], "
                f"got {tuple(local_field.shape)}."
            )
        local_field = local_field.float()
        if not torch.isfinite(local_field).all():
            raise ValueError("local_field must contain only finite values.")
        brain_mask = _require_image("brain_mask", brain_mask, local_field)
        initial = _require_image("initial", initial, local_field)
        if torch.any(brain_mask < 0.0):
            raise ValueError("brain_mask must be nonnegative.")
        weight = validate_magnitude_weight(magnitude_weight, local_field)

        total_time, data_coefficient = self.coefficients()
        regularizer_step = total_time.float() / self.num_steps
        data_step = data_coefficient.float() / self.num_steps
        chi = initial.float()
        states: list[torch.Tensor] | None = [chi] if return_states else None
        data_norms: list[torch.Tensor] = []
        regularizer_norms: list[torch.Tensor] = []
        state_norms: list[torch.Tensor] = []

        for _ in range(self.num_steps):
            force_R = self._regularizer_force(chi).float()
            force_D = self._data_force_validated(
                chi,
                local_field,
                weight,
                dipole_kernel,
            )
            data_norms.append(self._norm(force_D).detach())
            regularizer_norms.append(self._norm(force_R).detach())
            chi = chi - regularizer_step * force_R - data_step * force_D
            if self.mask_state_each_step:
                chi = chi * brain_mask
            chi = chi.float()
            state_norms.append(self._norm(chi).detach())
            if states is not None:
                states.append(chi)

        predicted_field = self.operator.forward(chi, dipole_kernel)
        return TDVOutput(
            susceptibility=chi,
            predicted_field=predicted_field.float(),
            states=tuple(states) if states is not None else None,
            data_gradient_norm=torch.stack(data_norms).mean(dim=0),
            regularizer_gradient_norm=torch.stack(regularizer_norms).mean(dim=0),
            state_norm=torch.stack(state_norms).mean(dim=0),
            time=total_time.float(),
            data_coefficient=data_coefficient.float(),
            regularizer_step=regularizer_step.float(),
            data_step=data_step.float(),
        )
