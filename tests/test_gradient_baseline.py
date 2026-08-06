from __future__ import annotations

import json

import torch

from tdv_qsm.gradient_baseline import (
    GradientBaselineConfig,
    evaluate_gradient_baseline,
    gradient_descent_qsm,
)
from tdv_qsm.operators.dipole import QSMOperator, build_dipole_kernel
from tdv_qsm.train import SingleVolume


def test_gradient_descent_step_is_exact_weight_squared_data_update() -> None:
    torch.manual_seed(18)
    shape = (4, 5, 6)
    operator = QSMOperator(build_dipole_kernel(shape))
    initial = torch.randn(1, 1, *shape)
    local_field = torch.randn_like(initial)
    weight = torch.rand_like(initial)
    mask = torch.ones_like(initial)

    output = gradient_descent_qsm(
        local_field,
        weight,
        initial,
        operator,
        brain_mask=mask,
        num_steps=1,
        step_size=0.3,
        mask_state_each_step=False,
    )

    expected_force = operator.adjoint(
        weight.square() * (operator.forward(initial) - local_field)
    )
    expected = initial - 0.3 * expected_force
    torch.testing.assert_close(output.susceptibility, expected, atol=2e-6, rtol=2e-6)
    assert output.susceptibility.dtype == torch.float32
    assert output.data_energy.shape == (2, 1)


def test_gradient_descent_masks_state_and_rejects_invalid_weights() -> None:
    shape = (3, 4, 5)
    operator = QSMOperator(torch.zeros(shape))
    initial = torch.ones(1, 1, *shape)
    local_field = torch.zeros_like(initial)
    weight = torch.ones_like(initial)
    mask = torch.ones_like(initial)
    mask[..., 0, 0, 0] = 0.0

    output = gradient_descent_qsm(
        local_field,
        weight,
        initial,
        operator,
        brain_mask=mask,
        num_steps=2,
        step_size=1.0,
    )
    assert output.susceptibility[..., 0, 0, 0].item() == 0.0

    bad_weight = weight.clone()
    bad_weight[..., 0, 0, 0] = -1.0
    try:
        gradient_descent_qsm(
            local_field,
            bad_weight,
            initial,
            operator,
            num_steps=1,
            step_size=1.0,
        )
    except ValueError as error:
        assert "nonnegative" in str(error)
    else:
        raise AssertionError("Negative magnitude weights must be rejected.")


def test_baseline_evaluation_writes_reconstruction_metrics_and_conventions(tmp_path) -> None:
    torch.manual_seed(4)
    shape = (4, 4, 4)
    sample = SingleVolume(
        susceptibility=0.01 * torch.randn(1, 1, *shape),
        magnitude=torch.ones(1, 1, *shape),
        brain_mask=torch.ones(1, 1, *shape),
    )
    config = GradientBaselineConfig(
        num_steps=2,
        step_size=None,
        snr=70.0,
        seed=3,
    )

    output, metrics = evaluate_gradient_baseline(sample, config, tmp_path)

    assert output.susceptibility.shape == sample.susceptibility.shape
    assert torch.isfinite(output.susceptibility).all()
    assert metrics["final_nrmse"] >= 0.0
    assert metrics["resolved_step_size"] > 0.0
    assert (tmp_path / "history.csv").is_file()
    assert (tmp_path / "history.png").is_file()
    assert (tmp_path / "reconstruction.png").is_file()
    assert (tmp_path / "reconstruction.pt").is_file()
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["regularizer"] == "none"
    assert report["data_gradient"] == "A^H W^2 (A chi - b)"
    assert report["selection_rule"] == "fixed iteration count; ground truth is not used for stopping"
    assert report["weight_rule"].startswith("Stored W (not sqrt(W)) = dimensionless magn")
