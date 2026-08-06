from __future__ import annotations

import json
import math

import matplotlib.image as mpimg
import numpy as np
import torch

import tdv_qsm.train as train_module

from tdv_qsm.losses import nrmse
from tdv_qsm.models.energy import TDVEnergy3D
from tdv_qsm.models.explicit_tdv import ExplicitTDVQSM3D
from tdv_qsm.operators.dipole import DipoleOperator3D, build_dipole_kernel
from tdv_qsm.train import (
    SingleVolume,
    TrainingConfig,
    _cosmos_display_planes,
    add_random_phase_outliers,
    apply_geometry_augmentation,
    magnitude_weight,
    print_model_architecture,
    simulate_noisy_local_field,
    summarize_model_architecture,
    save_reconstruction_figure,
    train_single_volume,
)


def test_model_architecture_summary_counts_tdv_structure_and_parameters(capsys) -> None:
    model = ExplicitTDVQSM3D(
        TDVEnergy3D(num_features=2, num_macro_blocks=2, use_amp=False),
        num_steps=1,
    )

    summary = summarize_model_architecture(model)

    assert summary.microblocks == 10
    assert summary.macroblocks == 2
    assert summary.convolutions == 30
    assert summary.total_parameters == sum(parameter.numel() for parameter in model.parameters())
    assert summary.trainable_parameters == sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print_model_architecture(model)
    output = capsys.readouterr().out
    assert "ExplicitTDVQSM3D(" in output
    assert "Microblocks: 10" in output
    assert "Macroblocks: 2" in output
    assert "Convolutions: 30" in output
    assert f"Parameters: {summary.total_parameters:,}" in output


def test_weight_rule_and_epoch_noise_are_explicit() -> None:
    shape = (5, 5, 5)
    susceptibility = torch.randn(1, 1, *shape)
    magnitude = torch.linspace(0.0, 2.0, susceptibility.numel()).reshape_as(susceptibility)
    mask = torch.ones_like(susceptibility)
    operator = DipoleOperator3D(build_dipole_kernel(shape))
    torch.testing.assert_close(magnitude_weight(magnitude), magnitude)
    field_1 = simulate_noisy_local_field(susceptibility, magnitude, mask, operator, snr=70.0, phase_scale=1.0, seed=1)
    field_2 = simulate_noisy_local_field(susceptibility, magnitude, mask, operator, snr=70.0, phase_scale=1.0, seed=2)
    assert not torch.equal(field_1, field_2)


def test_geometry_augmentation_transforms_only_spatial_volumes() -> None:
    shape = (2, 3, 4)
    susceptibility = torch.arange(np.prod(shape), dtype=torch.float32).reshape(1, 1, *shape)
    sample = SingleVolume(
        susceptibility=susceptibility,
        magnitude=susceptibility + 100.0,
        brain_mask=susceptibility + 200.0,
    )
    permutation = (2, 0, 1)
    mirrored = (True, False, True)

    augmented = apply_geometry_augmentation(
        sample,
        permutation_zyx=permutation,
        mirrored_zyx=mirrored,
    )

    def expected(value: torch.Tensor) -> torch.Tensor:
        return value.permute(0, 1, 4, 2, 3).flip(2, 4).contiguous()

    torch.testing.assert_close(augmented.susceptibility, expected(sample.susceptibility))
    torch.testing.assert_close(augmented.magnitude, expected(sample.magnitude))
    torch.testing.assert_close(augmented.brain_mask, expected(sample.brain_mask))


def test_phase_outliers_are_interior_scaled_and_leave_mask_unchanged() -> None:
    phase = torch.ones(1, 1, 5, 5, 5)
    mask = torch.ones_like(phase)
    original_mask = mask.clone()
    generator = torch.Generator().manual_seed(9)

    augmented, count = add_random_phase_outliers(
        phase,
        mask,
        generator=generator,
        probability=1.0,
    )

    changed = torch.nonzero(augmented != phase, as_tuple=False)
    assert count == changed.shape[0]
    assert 1 <= count <= 3
    assert torch.all((changed[:, -3:] > 0) & (changed[:, -3:] < 4))
    factors = augmented[augmented != phase]
    assert torch.all((factors >= 5.0) & (factors <= 10.0))
    torch.testing.assert_close(mask, original_mask)


def test_training_augmentation_does_not_modify_final_evaluation(
    tmp_path,
    monkeypatch,
) -> None:
    shape = (4, 4, 4)
    sample = SingleVolume(
        susceptibility=0.02 * torch.randn(1, 1, *shape),
        magnitude=torch.ones(1, 1, *shape),
        brain_mask=torch.ones(1, 1, *shape),
    )
    simulated_susceptibilities = []
    physical_metadata = []
    displayed = {}
    original_build_dipole_kernel = train_module.build_dipole_kernel

    def fake_geometry(sample_value, *, generator):
        return SingleVolume(
            susceptibility=sample_value.susceptibility + 1.0,
            magnitude=sample_value.magnitude,
            brain_mask=sample_value.brain_mask,
        )

    def fake_simulation(
        susceptibility,
        magnitude,
        brain_mask,
        operator,
        *,
        snr,
        phase_scale,
        seed,
    ):
        simulated_susceptibilities.append(susceptibility.detach().clone())
        return torch.zeros_like(susceptibility)

    def fake_outliers(local_field, brain_mask, *, generator, probability=0.5):
        return local_field, 0

    def fake_figure(initial, prediction, ground_truth, output_path, **kwargs):
        displayed["ground_truth"] = ground_truth.detach().clone()

    def capture_kernel(shape_value, voxel_size, b0_direction, **kwargs):
        physical_metadata.append((tuple(voxel_size), tuple(b0_direction)))
        return original_build_dipole_kernel(
            shape_value,
            voxel_size,
            b0_direction,
            **kwargs,
        )

    monkeypatch.setattr(train_module, "random_geometry_augmentation", fake_geometry)
    monkeypatch.setattr(train_module, "simulate_noisy_local_field", fake_simulation)
    monkeypatch.setattr(train_module, "add_random_phase_outliers", fake_outliers)
    monkeypatch.setattr(train_module, "save_reconstruction_figure", fake_figure)
    monkeypatch.setattr(train_module, "build_dipole_kernel", capture_kernel)
    config = TrainingConfig(
        epochs=1,
        features=1,
        macro_blocks=1,
        num_steps=1,
        learning_rate=1e-4,
        augmentation=True,
        use_amp=False,
        voxel_size_zyx=(1.0, 2.0, 3.0),
        b0_direction_zyx=(0.0, 1.0, 0.0),
    )

    train_single_volume(sample, config, tmp_path)

    assert len(simulated_susceptibilities) == 2
    torch.testing.assert_close(
        simulated_susceptibilities[0],
        sample.susceptibility + 1.0,
    )
    torch.testing.assert_close(simulated_susceptibilities[1], sample.susceptibility)
    torch.testing.assert_close(displayed["ground_truth"], sample.susceptibility)
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["augmentation_enabled"] is True
    assert report["config"]["augmentation"] is True
    assert physical_metadata
    assert all(
        metadata == ((1.0, 2.0, 3.0), (0.0, 1.0, 0.0))
        for metadata in physical_metadata
    )


def test_tiny_training_writes_metrics_and_reconstruction(tmp_path) -> None:
    torch.manual_seed(5)
    shape = (5, 5, 5)
    sample = SingleVolume(
        susceptibility=0.02 * torch.randn(1, 1, *shape),
        magnitude=torch.ones(1, 1, *shape),
        brain_mask=torch.ones(1, 1, *shape),
    )
    config = TrainingConfig(epochs=2, features=1, macro_blocks=1, num_steps=2, learning_rate=1e-4)
    model, history = train_single_volume(sample, config, tmp_path)
    assert len(history) == 2
    assert all(math.isfinite(record["loss"]) and math.isfinite(record["nrmse"]) for record in history)
    assert all(math.isfinite(record["data_consistency_value"]) for record in history)
    assert all(math.isfinite(record["regularization_energy"]) for record in history)
    assert all(math.isfinite(record["regularization_energy_total"]) for record in history)
    assert math.isclose(history[-1]["data_step"], history[-1]["lambda"])
    assert model.raw_T is not None and model.raw_T.grad is not None
    assert model.raw_lambda is not None and model.raw_lambda.grad is not None
    assert (tmp_path / "history.csv").is_file()
    assert (tmp_path / "reconstruction.png").is_file()
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    total_time, data_coefficient = model.coefficients()
    assert math.isclose(
        report["taus"]["regularizer"],
        float(total_time.detach()) / model.num_steps,
    )
    assert math.isclose(
        report["taus"]["data"],
        float(data_coefficient.detach()),
    )
    assert report["weight_rule"].startswith("Stored W (not sqrt(W)) = dimensionless magn")
    assert report["explicit_update_rule"].endswith("- lambda * g_D")
    checkpoint = torch.load(tmp_path / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert checkpoint["weight_rule"] == "W = magn"
    assert checkpoint["explicit_update_rule"].endswith("- lambda * g_D")


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
    losses: list[float] = []
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
        losses.append(float(loss.detach()))
        loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        optimizer.step()
        model.regularizer.project_analysis_kernel_()
    assert min(losses[-10:]) < losses[0]


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
