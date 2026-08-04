from __future__ import annotations

import math

import matplotlib.image as mpimg
import numpy as np
import torch

from tdv_qsm.losses import nrmse
from tdv_qsm.models.energy import TDVEnergy3D
from tdv_qsm.models.explicit_tdv import ExplicitTDVQSM3D
from tdv_qsm.operators.dipole import DipoleOperator3D, build_dipole_kernel
from tdv_qsm.train import (
    SingleVolume,
    TrainingConfig,
    _cosmos_display_planes,
    magnitude_weight,
    simulate_noisy_local_field,
    save_reconstruction_figure,
    train_single_volume,
)


def test_weight_rule_and_epoch_noise_are_explicit() -> None:
    shape = (5, 5, 5)
    susceptibility = torch.randn(1, 1, *shape)
    magnitude = torch.ones_like(susceptibility)
    mask = torch.ones_like(susceptibility)
    operator = DipoleOperator3D(build_dipole_kernel(shape))
    torch.testing.assert_close(magnitude_weight(magnitude), math.sqrt(2.0) * magnitude)
    field_1 = simulate_noisy_local_field(susceptibility, magnitude, mask, operator, snr=70.0, phase_scale=1.0, seed=1)
    field_2 = simulate_noisy_local_field(susceptibility, magnitude, mask, operator, snr=70.0, phase_scale=1.0, seed=2)
    assert not torch.equal(field_1, field_2)


def test_tiny_training_writes_metrics_and_reconstruction(tmp_path) -> None:
    torch.manual_seed(5)
    shape = (5, 5, 5)
    sample = SingleVolume(
        susceptibility=0.02 * torch.randn(1, 1, *shape),
        magnitude=torch.ones(1, 1, *shape),
        brain_mask=torch.ones(1, 1, *shape),
    )
    config = TrainingConfig(epochs=2, features=1, macro_blocks=1, num_steps=1, learning_rate=1e-4)
    model, history = train_single_volume(sample, config, tmp_path)
    assert len(history) == 2
    assert all(math.isfinite(record["loss"]) and math.isfinite(record["nrmse"]) for record in history)
    assert all(math.isfinite(record["data_consistency_value"]) for record in history)
    assert all(math.isfinite(record["regularization_energy"]) for record in history)
    assert all(math.isfinite(record["regularization_energy_total"]) for record in history)
    assert all(
        record["regularization_energy"] <= record["regularization_energy_total"]
        for record in history
    )
    assert model.raw_time.grad is not None
    assert (tmp_path / "history.csv").is_file()
    assert (tmp_path / "reconstruction.png").is_file()


def test_100_synthetic_training_steps_stay_finite_and_reach_every_parameter() -> None:
    torch.manual_seed(12)
    shape = (4, 4, 4)
    operator = DipoleOperator3D(build_dipole_kernel(shape))
    target = 0.01 * torch.randn(1, 1, *shape)
    local_field = operator.forward(target)
    mask = torch.ones_like(target)
    model = ExplicitTDVQSM3D(
        TDVEnergy3D(features=1, macro_blocks=1),
        num_steps=1,
        maximum_time=0.1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        output = model(
            local_field,
            mask,
            operator.dipole_kernel,
            torch.ones_like(target),
            torch.zeros_like(target),
        )
        loss = nrmse(output.susceptibility, target)
        assert torch.isfinite(loss)
        loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        optimizer.step()
        model.regularizer.project_zero_mean_()


def test_reconstruction_figure_uses_three_orientation_rows(tmp_path) -> None:
    volume = torch.zeros(1, 1, 8, 8, 8)
    output_path = tmp_path / "reconstruction.png"
    save_reconstruction_figure(volume, volume, volume, output_path)
    image = mpimg.imread(output_path)
    assert output_path.is_file()
    # A three-row diagnostic must be substantially taller than the former
    # one-row 4-panel image.
    assert image.shape[0] > 0.55 * image.shape[1]


def test_cosmos_display_plane_rotations_match_requested_orientation() -> None:
    volume = torch.arange(4 * 6 * 8, dtype=torch.float32).reshape(4, 6, 8).numpy()
    sagittal, coronal, axial = _cosmos_display_planes(volume)
    expected_sagittal = volume[:, :, volume.shape[2] // 2].T
    expected_coronal = np.rot90(volume[:, volume.shape[1] // 2, :].T, k=2)
    expected_axial = np.rot90(volume[volume.shape[0] // 2], k=1)
    assert np.array_equal(sagittal, expected_sagittal)
    assert np.array_equal(coronal, expected_coronal)
    assert np.array_equal(axial, expected_axial)
