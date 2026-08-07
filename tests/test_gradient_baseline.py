from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch
from scipy.io import savemat

import tdv_qsm.gradient_baseline as baseline_module
from tdv_qsm.gradient_baseline import (
    GradientBaselineConfig,
    evaluate_gradient_baseline,
    gradient_descent_qsm,
)
from tdv_qsm.operators.dipole import QSMOperator, build_dipole_kernel
from tdv_qsm.train import SingleVolume


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


def test_gradient_descent_records_ground_truth_nrmse_at_every_iteration() -> None:
    shape = (2, 3, 4)
    initial = torch.full((1, 1, *shape), 2.0)
    local_field = torch.zeros_like(initial)
    weight = torch.ones_like(initial)
    mask = torch.ones_like(initial)
    ground_truth = torch.ones_like(initial)

    output = gradient_descent_qsm(
        local_field,
        weight,
        initial,
        IdentityOperator(),
        brain_mask=mask,
        ground_truth=ground_truth,
        num_steps=2,
        step_size=0.25,
    )

    assert output.ground_truth_nrmse is not None
    torch.testing.assert_close(
        output.ground_truth_nrmse,
        torch.tensor([1.0, 0.5, 0.125]),
    )


def test_history_figure_replaces_gradient_norm_panel_with_nrmse(
    tmp_path,
    monkeypatch,
) -> None:
    calls: dict[str, object] = {"plots": [], "titles": [], "yscales": []}

    class Axis:
        def plot(self, iterations, values, **kwargs):
            calls["plots"].append(list(values))

        def set(self, **kwargs):
            calls["titles"].append(kwargs["title"])

        def set_yscale(self, scale):
            calls["yscales"].append(scale)

        def grid(self, **kwargs):
            return None

        def text(self, *args, **kwargs):
            return None

    class Figure:
        def savefig(self, path, **kwargs):
            calls["output_path"] = path

    class Pyplot:
        def subplots(self, rows, columns, **kwargs):
            return Figure(), [Axis(), Axis()]

        def close(self, figure):
            return None

    monkeypatch.setattr(baseline_module, "_pyplot", lambda: Pyplot())
    output = SimpleNamespace(
        data_energy=torch.tensor([[4.0], [3.0]]),
        data_gradient_norm=torch.tensor([[2.0], [1.0]]),
        state_norm=torch.tensor([[5.0], [4.0]]),
        ground_truth_nrmse=torch.tensor([0.5, 0.25]),
    )

    baseline_module._write_history(output, tmp_path)

    assert calls["plots"] == [[4.0, 3.0], [0.5, 0.25]]
    assert calls["titles"][1] == "Ground-truth NRMSE"
    assert calls["yscales"] == ["log"]
    assert calls["output_path"] == tmp_path / "history.png"
    assert "ground_truth_nrmse" in (tmp_path / "history.csv").read_text(
        encoding="utf-8"
    ).splitlines()[0]


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
    assert output.ground_truth_nrmse is not None
    assert output.ground_truth_nrmse.shape == (config.num_steps + 1,)
    assert metrics["final_nrmse"] == float(output.ground_truth_nrmse[-1].cpu())
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


def test_baseline_evaluation_accepts_direct_phase_dataset_without_ground_truth(
    tmp_path,
    monkeypatch,
) -> None:
    shape = (4, 4, 4)
    data_directory = tmp_path / "measured"
    data_directory.mkdir()
    phase = np.linspace(-0.01, 0.01, np.prod(shape), dtype=np.float32).reshape(shape)
    mask = np.ones(shape, dtype=np.float32)
    initial = np.full(shape, 0.02, dtype=np.float32)
    savemat(data_directory / "phase.mat", {"phase": phase})
    savemat(data_directory / "mask.mat", {"mask": mask})
    savemat(data_directory / "initial.mat", {"initial": initial})

    def fail_simulation(*args, **kwargs):
        raise AssertionError("phase.mat must bypass forward simulation")

    monkeypatch.setattr(
        baseline_module,
        "simulate_noisy_local_field",
        fail_simulation,
    )
    output_directory = tmp_path / "output"

    output, metrics = evaluate_gradient_baseline(
        data_directory,
        GradientBaselineConfig(num_steps=1, step_size=0.1),
        output_directory,
    )

    saved = torch.load(
        output_directory / "reconstruction.pt",
        map_location="cpu",
        weights_only=True,
    )
    torch.testing.assert_close(saved["local_field"][0, 0], torch.from_numpy(phase))
    torch.testing.assert_close(saved["magnitude_weight"][0, 0], torch.from_numpy(mask))
    torch.testing.assert_close(saved["initial"][0, 0], torch.from_numpy(initial))
    assert saved["ground_truth"] is None
    assert output.susceptibility.shape == (1, 1, *shape)
    assert output.ground_truth_nrmse is None
    assert saved["ground_truth_nrmse"] is None
    assert "initial_nrmse" not in metrics
    assert "final_nrmse" not in metrics
    assert (output_directory / "reconstruction.png").is_file()
    report = json.loads((output_directory / "report.json").read_text(encoding="utf-8"))
    assert report["field_source"] == "phase.mat"
    assert report["weight_source"] == "mask"
