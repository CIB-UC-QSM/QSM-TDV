"""Explicit-Euler reconstruction with a scalar TDV regularizer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from tdv_qsm.models.energy import TDVEnergy3D
from tdv_qsm.operators.dipole import DipoleOperator3D


@dataclass(frozen=True)
class TDVOutput:
    """Output of an explicit TDV-QSM reconstruction."""

    susceptibility: torch.Tensor
    predicted_field: torch.Tensor
    states: tuple[torch.Tensor, ...] | None
    data_gradient_norm: torch.Tensor
    regularizer_gradient_norm: torch.Tensor
    state_norm: torch.Tensor
    time: torch.Tensor
    step_size: torch.Tensor


def _require_image(name: str, value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if value.ndim != 5:
        raise ValueError(f"{name} must be [B, 1, Z, Y, X], got {tuple(value.shape)}.")
    if value.shape[1] != 1:
        raise ValueError(f"{name} must have one channel, got {value.shape[1]}.")
    if value.shape != reference.shape:
        raise ValueError(
            f"{name} must match local_field {tuple(reference.shape)}, got {tuple(value.shape)}."
        )
    return value.float()


class ExplicitTDVQSM3D(nn.Module):
    """Unroll explicit gradient descent on data fidelity plus TDV energy.

    The learned TDV parameters are shared at every step.  The state, FFTs,
    data term, TDV-energy reduction, and update are float32; only TDV
    convolution internals use CUDA float16 autocast.
    """

    def __init__(
        self,
        regularizer: TDVEnergy3D,
        *,
        num_steps: int = 1,
        maximum_time: float = 0.25,
        initial_raw_time: float = 2,
    ) -> None:
        super().__init__()
        if num_steps < 1:
            raise ValueError("num_steps must be at least one.")
        if maximum_time <= 0.0:
            raise ValueError("maximum_time must be positive.")
        self.regularizer = regularizer
        self.raw_time = nn.Parameter(torch.tensor(float(initial_raw_time), dtype=torch.float32))
        self.num_steps = int(num_steps)
        self.maximum_time = float(maximum_time)

    @staticmethod
    def _validate_weight(weight: torch.Tensor, local_field: torch.Tensor) -> torch.Tensor:
        if weight.ndim != 5:
            raise ValueError("magnitude_weight must have rank 5 [B, 1, Z, Y, X].")
        if weight.shape[1] != 1:
            raise ValueError("magnitude_weight must have one channel.")
        try:
            weight = torch.broadcast_to(weight.float(), local_field.shape)
        except RuntimeError as error:
            raise ValueError(
                "magnitude_weight must be broadcast-compatible with local_field."
            ) from error
        if not torch.isfinite(weight).all():
            raise ValueError("Magnitude weights must be finite.")
        if torch.any(weight < 0):
            raise ValueError("Magnitude weights must be nonnegative.")
        return weight

    @staticmethod
    def _norm(value: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(value.float().flatten(1), ord=2, dim=1)

    def forward(
        self,
        local_field: torch.Tensor,
        mask: torch.Tensor,
        dipole_kernel: torch.Tensor,
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
        mask = _require_image("mask", mask, local_field)
        initial = _require_image("initial", initial, local_field)
        if not torch.isfinite(local_field).all() or not torch.isfinite(mask).all():
            raise ValueError("local_field and mask must be finite.")
        if torch.any(mask < 0.0):
            raise ValueError("mask must be nonnegative.")
        weight = self._validate_weight(magnitude_weight, local_field)
        operator = DipoleOperator3D(dipole_kernel.float())

        # This local context deliberately works even if a caller has wrapped
        # evaluation in no_grad: evaluating ∇_chi R_theta still needs autograd.
        with torch.enable_grad():
            chi = initial.float().requires_grad_(True)
            total_time = self.maximum_time * torch.sigmoid(self.raw_time.float())
            step_size = total_time / self.num_steps
            states: list[torch.Tensor] | None = [chi] if return_states else None
            data_norms: list[torch.Tensor] = []
            regularizer_norms: list[torch.Tensor] = []
            state_norms: list[torch.Tensor] = []

            for _ in range(self.num_steps):
                predicted_field = operator.forward(chi)
                residual = predicted_field - local_field
                data_gradient = operator.adjoint(weight * (weight * residual))
                energy = self.regularizer.energy(chi, mask)
                grad_regularizer, = torch.autograd.grad(
                    energy.sum(),
                    chi,
                    create_graph=self.training,
                    retain_graph=self.training,
                )
                data_norms.append(self._norm(data_gradient).detach())
                regularizer_norms.append(self._norm(grad_regularizer).detach())
                chi = chi - step_size.float() * (data_gradient + grad_regularizer.float())
                chi = chi * mask
                state_norms.append(self._norm(chi).detach())
                if states is not None:
                    states.append(chi)

            predicted_field = operator.forward(chi)

        return TDVOutput(
            susceptibility=chi.float(),
            predicted_field=predicted_field.float(),
            states=tuple(states) if states is not None else None,
            data_gradient_norm=torch.stack(data_norms).mean(dim=0),
            regularizer_gradient_norm=torch.stack(regularizer_norms).mean(dim=0),
            state_norm=torch.stack(state_norms).mean(dim=0),
            time=total_time.float(),
            step_size=step_size.float(),
        )
