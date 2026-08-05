from __future__ import annotations

import pytest
import torch

from tdv_qsm.losses import mask_and_reference, nrmse, weighted_data_consistency_loss
from tdv_qsm.operators.dipole import DipoleOperator3D, build_dipole_kernel


def test_nrmse_is_per_sample_then_averaged() -> None:
    truth = torch.tensor([[[[[3.0]]]], [[[[4.0]]]]])
    prediction = torch.tensor([[[[[0.0]]]], [[[[2.0]]]]])
    # Per-sample errors are 3/3 and 2/4, not one global batch ratio.
    assert nrmse(prediction, truth).item() == pytest.approx(0.75)
    assert nrmse(truth, truth).item() == pytest.approx(0.0)


def test_nrmse_uses_epsilon_only_for_a_zero_reference_norm() -> None:
    prediction = torch.ones(1, 1, 1, 1, 1)
    truth = torch.zeros_like(prediction)
    assert nrmse(prediction, truth, eps=1e-4).item() == pytest.approx(1e4)


def test_weighted_data_consistency_is_exact_w_times_residual() -> None:
    shape = (4, 4, 4)
    operator = DipoleOperator3D(build_dipole_kernel(shape))
    chi = torch.randn(1, 1, *shape)
    local_field = torch.randn_like(chi)
    weight = torch.ones_like(chi)
    got = weighted_data_consistency_loss(chi, local_field, weight, operator)
    expected = (operator.forward(chi) - local_field).square().mean()
    torch.testing.assert_close(got, expected)
    with pytest.raises(ValueError, match="nonnegative"):
        weighted_data_consistency_loss(chi, local_field, -weight, operator)
    bad = weight.clone()
    bad[..., 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        weighted_data_consistency_loss(chi, local_field, bad, operator)


def test_mask_and_reference_is_explicit_and_per_sample() -> None:
    value = torch.tensor([[[[[1.0, 3.0]]]], [[[[2.0, 6.0]]]]])
    mask = torch.ones_like(value)
    torch.testing.assert_close(
        mask_and_reference(value, mask, convention="already_referenced"), value
    )
    referenced = mask_and_reference(value, mask, convention="masked_mean_zero")
    expected = torch.tensor([[[[[-1.0, 1.0]]]], [[[[-2.0, 2.0]]]]])
    torch.testing.assert_close(referenced, expected)
    with pytest.raises(ValueError, match="convention"):
        mask_and_reference(value, mask, convention="implicit")
