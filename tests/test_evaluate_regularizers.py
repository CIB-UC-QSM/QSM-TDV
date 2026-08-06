from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat

import tdv_qsm.evaluate_regularizers as evaluation_module
from tdv_qsm.evaluate_regularizers import (
    EvaluationData,
    _load_evaluation_data,
    _report_taus,
    evaluate_learned_regularizers,
    iterate_learned_regularizers,
)
from tdv_qsm.losses import nrmse
from tdv_qsm.models.energy import TDVEnergy3D


class QuadraticRegularizer:
    def force(self, chi: torch.Tensor) -> torch.Tensor:
        return chi


class IdentityOperator:
    def forward(
        self,
        chi: torch.Tensor,
        dipole_kernel: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return chi

    def adjoint(
        self,
        field: torch.Tensor,
        dipole_kernel: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return field


def test_phase_is_loaded_directly_and_missing_magn_uses_mask(tmp_path) -> None:
    shape = (3, 4, 5)
    phase = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    mask = np.ones(shape, dtype=np.float32)
    mask[0, 0, 0] = 0.0
    chi = np.full(shape, 0.25, dtype=np.float32)
    savemat(tmp_path / "phase.mat", {"phase": phase})
    savemat(tmp_path / "mask.mat", {"mask": mask})
    savemat(tmp_path / "chi.mat", {"chi": chi})

    data = _load_evaluation_data(tmp_path, torch.device("cpu"))

    torch.testing.assert_close(data.local_field[0, 0], torch.from_numpy(phase))
    torch.testing.assert_close(data.weight[0, 0], torch.from_numpy(mask))
    torch.testing.assert_close(data.ground_truth[0, 0], torch.from_numpy(chi))
    assert data.phase_was_provided
    assert data.weight_source == "mask"


def test_magn_is_loaded_without_replacing_it_by_mask(tmp_path) -> None:
    shape = (2, 3, 4)
    phase = np.ones(shape, dtype=np.float32)
    mask = np.ones(shape, dtype=np.float32)
    magn = np.linspace(0.0, 2.0, np.prod(shape), dtype=np.float32).reshape(shape)
    savemat(tmp_path / "phase.mat", {"phase": phase})
    savemat(tmp_path / "msk.mat", {"msk": mask})
    savemat(tmp_path / "magn.mat", {"magn": magn})

    data = _load_evaluation_data(tmp_path, torch.device("cpu"))

    torch.testing.assert_close(data.weight[0, 0], torch.from_numpy(magn))
    assert data.ground_truth is None
    assert data.weight_source == "magn"


def test_default_absolute_cosmos_path_falls_back_to_workspace_data(
    tmp_path,
    monkeypatch,
) -> None:
    data_directory = tmp_path / "cosmos_data"
    data_directory.mkdir()
    shape = (2, 3, 4)
    phase = np.ones(shape, dtype=np.float32)
    mask = np.ones(shape, dtype=np.float32)
    savemat(data_directory / "phase.mat", {"phase": phase})
    savemat(data_directory / "mask.mat", {"mask": mask})
    monkeypatch.chdir(tmp_path)

    data = _load_evaluation_data(Path("/cosmos_data"), torch.device("cpu"))

    torch.testing.assert_close(data.local_field[0, 0], torch.from_numpy(phase))


def test_iteration_records_update_and_ground_truth_nrmse() -> None:
    initial = torch.full((1, 1, 2, 2, 2), 2.0)
    local_field = torch.zeros_like(initial)
    mask = torch.ones_like(initial)
    weight = torch.full_like(initial, 0.5)
    ground_truth = torch.ones_like(initial)
    data = EvaluationData(
        local_field=local_field,
        brain_mask=mask,
        weight=weight,
        ground_truth=ground_truth,
        phase_was_provided=True,
        weight_source="magn",
    )

    output = iterate_learned_regularizers(
        QuadraticRegularizer(),
        IdentityOperator(),
        data,
        initial,
        total_iterations=2,
        taus={"regularizer": 0.1, "data": 0.2},
        mask_state_each_step=True,
    )

    expected_first = initial - 0.1 * initial - 0.2 * weight.square() * initial
    expected_second = expected_first - 0.1 * expected_first - 0.2 * weight.square() * expected_first
    torch.testing.assert_close(output.chi_pred, expected_second)
    assert output.tol_update.shape == (2,)
    assert output.ground_truth_nrmse is not None
    assert output.ground_truth_nrmse.shape == (2,)
    assert math.isclose(
        float(output.tol_update[0]),
        float(nrmse(expected_first, initial)),
        rel_tol=1e-6,
    )
    assert math.isclose(
        float(output.ground_truth_nrmse[1]),
        float(nrmse(expected_second, ground_truth)),
        rel_tol=1e-6,
    )


def test_report_taus_are_loaded_directly() -> None:
    report = {"taus": {"regularizer": 0.03, "data": 0.07}}

    regularizer_tau, data_tau = _report_taus(report)

    assert math.isclose(regularizer_tau, 0.03)
    assert math.isclose(data_tau, 0.07)


def test_metric_figure_uses_separate_tol_and_ground_truth_panels(
    tmp_path,
    monkeypatch,
) -> None:
    calls = {"plots": []}

    class Axis:
        def plot(self, iterations, values, **kwargs):
            calls["plots"].append(kwargs["label"])

        def set(self, **kwargs):
            return None

        def grid(self, **kwargs):
            return None

        def text(self, *args, **kwargs):
            return None

    class Figure:
        def savefig(self, path, **kwargs):
            calls["output_path"] = path

    class Pyplot:
        def subplots(self, rows, columns, **kwargs):
            calls["layout"] = (rows, columns)
            return Figure(), [Axis(), Axis()]

        def close(self, figure):
            return None

    monkeypatch.setattr(evaluation_module, "_pyplot", lambda: Pyplot())
    output_path = tmp_path / "metrics.png"

    evaluation_module._save_metric_figure(
        torch.tensor([0.2, 0.1]),
        torch.tensor([0.4, 0.3]),
        output_path,
    )

    assert calls["layout"] == (1, 2)
    assert calls["plots"] == ["tol_update", "Ground-truth NRMSE"]
    assert calls["output_path"] == output_path


def test_end_to_end_phase_path_bypasses_simulation_and_writes_figures(
    tmp_path,
    monkeypatch,
) -> None:
    shape = (4, 4, 4)
    data_directory = tmp_path / "cosmos"
    data_directory.mkdir()
    phase = np.linspace(-0.01, 0.01, np.prod(shape), dtype=np.float32).reshape(shape)
    mask = np.ones(shape, dtype=np.float32)
    chi = np.zeros(shape, dtype=np.float32)
    savemat(data_directory / "phase.mat", {"phase": phase})
    savemat(data_directory / "mask.mat", {"mask": mask})
    savemat(data_directory / "chi.mat", {"chi": chi})
    regularizer = TDVEnergy3D(num_features=1, num_macro_blocks=1, use_amp=False)
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    checkpoint_path = model_directory / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": {
                f"regularizer.{key}": value
                for key, value in regularizer.state_dict().items()
            },
            "config": {
                "features": 1,
                "macro_blocks": 1,
                "use_amp": False,
            },
        },
        checkpoint_path,
    )
    (model_directory / "report.json").write_text(
        json.dumps({"taus": {"regularizer": 0.03, "data": 0.07}}),
        encoding="utf-8",
    )

    def fail_simulation(*args, **kwargs):
        raise AssertionError("phase.mat must bypass forward simulation")

    monkeypatch.setattr(evaluation_module, "simulate_noisy_local_field", fail_simulation)
    output_directory = tmp_path / "output"
    output = evaluate_learned_regularizers(
        model_directory,
        data_directory,
        {"device": "cpu", "output_dir": output_directory, "use_amp": False},
        2,
        None,
    )

    assert output.tol_update.shape == (2,)
    assert output.ground_truth_nrmse is not None
    assert math.isclose(output.regularizer_tau, 0.03)
    assert math.isclose(output.data_tau, 0.07)
    assert (output_directory / "metrics.csv").is_file()
    assert (output_directory / "metrics.png").is_file()
    assert (output_directory / "chi_pred.mat").is_file()
    assert (output_directory / "chi_pred.png").is_file()


def test_explicit_snr_controls_field_simulation(tmp_path, monkeypatch) -> None:
    shape = (4, 4, 4)
    data_directory = tmp_path / "cosmos"
    data_directory.mkdir()
    mask = np.ones(shape, dtype=np.float32)
    chi = np.zeros(shape, dtype=np.float32)
    savemat(data_directory / "mask.mat", {"mask": mask})
    savemat(data_directory / "chi_cosmos.mat", {"chi_cosmos": chi})
    regularizer = TDVEnergy3D(num_features=1, num_macro_blocks=1, use_amp=False)
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    torch.save(
        {
            "model_state_dict": {
                f"regularizer.{key}": value
                for key, value in regularizer.state_dict().items()
            },
            "config": {
                "features": 1,
                "macro_blocks": 1,
                "use_amp": False,
                "snr": 70.0,
            },
        },
        model_directory / "checkpoint.pt",
    )
    (model_directory / "report.json").write_text(
        json.dumps({"taus": {"regularizer": 0.0, "data": 0.0}}),
        encoding="utf-8",
    )
    captured = {}

    def capture_simulation(
        susceptibility,
        magnitude,
        brain_mask,
        operator,
        *,
        snr,
        phase_scale,
        seed,
    ):
        captured["snr"] = snr
        return torch.zeros_like(susceptibility)

    monkeypatch.setattr(
        evaluation_module,
        "simulate_noisy_local_field",
        capture_simulation,
    )

    evaluate_learned_regularizers(
        model_directory,
        data_directory,
        {
            "device": "cpu",
            "output_dir": tmp_path / "output",
            "use_amp": False,
            "snr": 20.0,
        },
        1,
        None,
        snr=35.0,
    )

    assert captured["snr"] == 35.0
