"""Training losses for explicit TDV-QSM."""

from __future__ import annotations

import torch

from tdv_qsm.operators.dipole import DipoleOperator3D


def mask_and_reference(
    value: torch.Tensor,
    brain_mask: torch.Tensor,
    *,
    convention: str,
) -> torch.Tensor:
    """Apply an explicit susceptibility reference and brain mask.

    ``already_referenced`` preserves supplied susceptibility values and only
    masks them.  ``masked_mean_zero`` independently subtracts each sample's
    in-mask mean before masking.  No implicit convention is selected.
    """

    if value.shape != brain_mask.shape or value.ndim < 2:
        raise ValueError("value and brain_mask must have identical batched shapes.")
    value = value.float()
    mask = brain_mask.float()
    if not torch.isfinite(value).all() or not torch.isfinite(mask).all():
        raise ValueError("value and brain_mask must be finite.")
    if torch.any(mask < 0.0):
        raise ValueError("brain_mask must be nonnegative.")
    count = mask.flatten(1).sum(dim=1)
    if torch.any(count <= 0.0):
        raise ValueError("brain_mask must be nonempty for every sample.")
    if convention == "already_referenced":
        referenced = value
    elif convention == "masked_mean_zero":
        mean = (value * mask).flatten(1).sum(dim=1) / count
        reshape = (value.shape[0],) + (1,) * (value.ndim - 1)
        referenced = value - mean.reshape(reshape)
    else:
        raise ValueError(
            "reference convention must be 'already_referenced' or 'masked_mean_zero'."
        )
    return referenced * mask


def nrmse(x_pred: torch.Tensor, x_true: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Mean of per-sample normalized root mean squared errors."""

    if x_pred.shape != x_true.shape:
        raise ValueError("x_pred and x_true must have identical shapes.")
    if x_pred.ndim < 2:
        raise ValueError("NRMSE expects a batch dimension and at least one feature dimension.")
    pred = x_pred.float().flatten(start_dim=1)
    true = x_true.float().flatten(start_dim=1)
    error_norm = torch.linalg.vector_norm(pred - true, ord=2, dim=1)
    true_norm = torch.linalg.vector_norm(true, ord=2, dim=1).clamp_min(eps)
    return (error_norm / true_norm).mean()


def weighted_data_consistency_loss(
    chi_pred: torch.Tensor,
    local_field: torch.Tensor,
    magnitude_weight: torch.Tensor,
    operator: DipoleOperator3D,
    *,
    field_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean per-sample energy of the exact residual ``W(Achi - b)``."""

    if chi_pred.shape != local_field.shape:
        raise ValueError("chi_pred and local_field must have matching shapes.")
    try:
        weight = torch.broadcast_to(magnitude_weight.float(), local_field.shape)
    except RuntimeError as error:
        raise ValueError("magnitude_weight must broadcast to local_field.") from error
    if not torch.isfinite(weight).all():
        raise ValueError("Magnitude weights must be finite.")
    if torch.any(weight < 0):
        raise ValueError("Magnitude weights must be nonnegative.")

    residual = operator.forward(chi_pred.float()) - local_field.float()
    weighted_residual = weight * residual
    if field_mask is not None:
        try:
            field_mask = torch.broadcast_to(field_mask.float(), local_field.shape)
        except RuntimeError as error:
            raise ValueError("field_mask must broadcast to local_field.") from error
        if not torch.isfinite(field_mask).all() or torch.any(field_mask < 0.0):
            raise ValueError("field_mask must be finite and nonnegative.")
        weighted_residual = weighted_residual * field_mask
        voxel_count = field_mask.flatten(1).sum(dim=1).clamp_min(1.0)
    else:
        voxel_count = torch.full(
            (weighted_residual.shape[0],),
            weighted_residual[0].numel(),
            dtype=torch.float32,
            device=weighted_residual.device,
        )
    residual_energy = weighted_residual.square().flatten(1).sum(dim=1)
    return (residual_energy / voxel_count).mean()
