from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from scipy.io import loadmat, savemat

from tdv_qsm.losses import mask_and_reference, nrmse
from tdv_qsm.models.energy import TDVEnergy3D
from tdv_qsm.operators.dipole import DipoleOperator3D, build_dipole_kernel
from tdv_qsm.train import (
    _array_to_volume,
    _cosmos_display_planes,
    _find_key,
    _pyplot,
    _resolve_data_path,
    initial_backprojection,
    save_reconstruction_figure,
    simulate_noisy_local_field,
)


class Regularizer(Protocol):
    def force(self, chi: torch.Tensor) -> torch.Tensor: ...


class PhysicalOperator(Protocol):
    def forward(
        self,
        chi: torch.Tensor,
        dipole_kernel: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    def adjoint(
        self,
        field: torch.Tensor,
        dipole_kernel: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class EvaluationData:
    local_field: torch.Tensor | None
    brain_mask: torch.Tensor
    weight: torch.Tensor
    ground_truth: torch.Tensor | None
    phase_was_provided: bool
    weight_source: str
    simulation_susceptibility: torch.Tensor | None = None
    initial: torch.Tensor | None = None


@dataclass(frozen=True)
class RegularizerEvaluationOutput:
    chi_pred: torch.Tensor
    tol_update: torch.Tensor
    ground_truth_nrmse: torch.Tensor | None
    local_field: torch.Tensor
    weight: torch.Tensor
    initial: torch.Tensor
    output_directory: Path
    regularizer_tau: float
    data_tau: float


def _first_existing(directory: Path, names: Sequence[str]) -> Path | None:
    return next((directory / name for name in names if (directory / name).is_file()), None)


def _load_mat_array(path: Path, keys: tuple[str, ...], label: str) -> np.ndarray:
    try:
        values = loadmat(path)
    except NotImplementedError as error:
        raise ValueError(f"{path} is MATLAB v7.3/HDF5 and is not supported.") from error
    return _find_key(values, keys, label)


def _load_volume(
    path: Path,
    keys: tuple[str, ...],
    label: str,
    device: torch.device,
) -> torch.Tensor:
    return _array_to_volume(_load_mat_array(path, keys, label), label, device)


def _validate_loaded_data(data: EvaluationData) -> EvaluationData:
    reference = data.brain_mask
    if reference.ndim != 5 or reference.shape[1] != 1:
        raise ValueError("mask must have shape [B, 1, Z, Y, X].")
    if not torch.isfinite(reference).all() or torch.any(reference < 0.0):
        raise ValueError("mask must be finite and nonnegative.")
    if not torch.any(reference > 0.0):
        raise ValueError("mask must be nonempty.")
    for name, value in (
        ("phase", data.local_field),
        ("w", data.weight),
        ("chi", data.ground_truth),
        ("simulation susceptibility", data.simulation_susceptibility),
        ("initial", data.initial),
    ):
        if value is None:
            continue
        if value.shape != reference.shape:
            raise ValueError(f"{name} must have the same shape as mask.")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite.")
    if torch.any(data.weight < 0.0):
        raise ValueError("w must be nonnegative.")
    return data


def _load_evaluation_data(
    dataset_directory_path: str | Path,
    device: torch.device,
) -> EvaluationData:
    directory = _resolve_data_path(Path(dataset_directory_path))
    if not directory.is_dir():
        raise NotADirectoryError(f"COSMOS dataset directory does not exist: {directory}")
    mask_path = _first_existing(directory, ("mask.mat", "msk.mat", "mask", "msk"))
    if mask_path is None:
        raise FileNotFoundError("COSMOS dataset requires mask.mat or msk.mat.")
    brain_mask = _load_volume(
        mask_path,
        ("mask", "msk", "brain_mask"),
        "mask",
        device,
    )
    magnitude_path = _first_existing(directory, ("magn.mat", "magn"))
    if magnitude_path is None:
        weight = brain_mask.clone()
        weight_source = "mask"
    else:
        weight = _load_volume(
            magnitude_path,
            ("magn", "magnitude"),
            "magn",
            device,
        )
        weight_source = "magn"
    phase_path = directory / "phase.mat"
    if phase_path.is_file():
        local_field = _load_volume(
            phase_path,
            ("phase", "local_field", "field"),
            "phase",
            device,
        )
        phase_was_provided = True
    else:
        local_field = None
        phase_was_provided = False
    chi_path = directory / "chi.mat"
    ground_truth = (
        _load_volume(
            chi_path,
            ("chi", "chi_cosmos", "susceptibility"),
            "chi",
            device,
        )
        if chi_path.is_file()
        else None
    )
    simulation_path = _first_existing(directory, ("chi.mat", "chi_cosmos.mat"))
    simulation_susceptibility = (
        _load_volume(
            simulation_path,
            ("chi", "chi_cosmos", "susceptibility"),
            "simulation susceptibility",
            device,
        )
        if simulation_path is not None
        else None
    )
    initial_path = _first_existing(directory, ("initial.mat", "initial"))
    initial = (
        _load_volume(initial_path, ("initial", "chi_initial", "x0"), "initial", device)
        if initial_path is not None
        else None
    )
    return _validate_loaded_data(
        EvaluationData(
            local_field=local_field,
            brain_mask=brain_mask,
            weight=weight,
            ground_truth=ground_truth,
            phase_was_provided=phase_was_provided,
            weight_source=weight_source,
            simulation_susceptibility=simulation_susceptibility,
            initial=initial,
        )
    )


def _resolved_tau(
    taus: Mapping[str, float],
    names: tuple[str, ...],
    label: str,
) -> float:
    present = [name for name in names if name in taus]
    if len(present) != 1:
        raise ValueError(f"taus must provide exactly one {label} key from {names}.")
    value = float(taus[present[0]])
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} tau must be finite and nonnegative.")
    return value


def _resolve_taus(
    taus: Mapping[str, float] | Sequence[float],
) -> tuple[float, float]:
    if isinstance(taus, Mapping):
        regularizer_tau = _resolved_tau(
            taus,
            ("regularizer", "tau_regularizer", "tau_R"),
            "regularizer",
        )
        data_tau = _resolved_tau(
            taus,
            ("data", "tau_data", "tau_D"),
            "data",
        )
        return regularizer_tau, data_tau
    values = tuple(float(value) for value in taus)
    if len(values) != 2:
        raise ValueError("taus must contain regularizer and data weights in that order.")
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("taus must be finite and nonnegative.")
    return values


def _checkpoint_model_state(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    state_value = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state_value, Mapping):
        raise ValueError("Checkpoint must contain a model_state_dict or state_dict mapping.")
    return state_value


def _normalized_model_state(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for original_key, value in _checkpoint_model_state(checkpoint).items():
        if not isinstance(original_key, str) or not isinstance(value, torch.Tensor):
            continue
        key = original_key
        while key.startswith("module.") or key.startswith("_orig_mod."):
            key = key.split(".", maxsplit=1)[1]
        if key.startswith("model."):
            key = key.removeprefix("model.")
        normalized[key] = value
    return normalized


def _load_report(report_path: Path) -> Mapping[str, Any]:
    if not report_path.is_file():
        raise FileNotFoundError(f"Model report does not exist: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, Mapping):
        raise ValueError("Model report root must be a JSON object.")
    return report


def _report_taus(report: Mapping[str, Any]) -> tuple[float, float]:
    if "taus" not in report:
        raise KeyError("Model report must contain taus.")
    report_taus = report["taus"]
    if not isinstance(report_taus, (Mapping, list, tuple)):
        raise ValueError("Model report taus must be a mapping or a two-value sequence.")
    return _resolve_taus(report_taus)


def iterate_learned_regularizers(
    regularizer: Regularizer,
    operator: PhysicalOperator,
    data: EvaluationData,
    initial: torch.Tensor,
    *,
    total_iterations: int,
    taus: Mapping[str, float] | Sequence[float],
    mask_state_each_step: bool,
) -> RegularizerEvaluationOutput:
    if total_iterations < 1:
        raise ValueError("total_iterations must be at least one.")
    if data.local_field is None:
        raise ValueError("local_field must be resolved before iteration.")
    if initial.shape != data.local_field.shape:
        raise ValueError("initial must have the same shape as local_field.")
    regularizer_tau, data_tau = _resolve_taus(taus)
    chi = initial.float()
    tol_history: list[torch.Tensor] = []
    ground_truth_history: list[torch.Tensor] = []
    reference_convention = "already_referenced"
    with torch.no_grad():
        for _ in range(total_iterations):
            previous = chi
            force_regularizer = regularizer.force(previous).float()
            residual = operator.forward(previous).float() - data.local_field.float()
            force_data = operator.adjoint(data.weight.float().square() * residual).float()
            chi = previous - regularizer_tau * force_regularizer - data_tau * force_data
            if mask_state_each_step:
                chi = chi * data.brain_mask.float()
            chi = chi.float()
            if not torch.isfinite(chi).all():
                raise FloatingPointError("Non-finite reconstruction state.")
            tol_history.append(nrmse(chi, previous).detach())
            if data.ground_truth is not None:
                prediction = mask_and_reference(
                    chi,
                    data.brain_mask,
                    convention=reference_convention,
                )
                target = mask_and_reference(
                    data.ground_truth,
                    data.brain_mask,
                    convention=reference_convention,
                )
                ground_truth_history.append(nrmse(prediction, target).detach())
    return RegularizerEvaluationOutput(
        chi_pred=chi,
        tol_update=torch.stack(tol_history),
        ground_truth_nrmse=(
            torch.stack(ground_truth_history) if ground_truth_history else None
        ),
        local_field=data.local_field,
        weight=data.weight,
        initial=initial,
        output_directory=Path(),
        regularizer_tau=regularizer_tau,
        data_tau=data_tau,
    )


def _checkpoint_regularizer_state(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    regularizer_state: dict[str, torch.Tensor] = {}
    direct_roots = ("analysis.", "macro_blocks.", "energy_head.")
    for key, value in _normalized_model_state(checkpoint).items():
        if key.startswith("regularizer."):
            regularizer_state[key.removeprefix("regularizer.")] = value
        elif key.startswith(direct_roots):
            regularizer_state[key] = value
    if not regularizer_state:
        raise ValueError("Checkpoint does not contain TDVEnergy3D parameters.")
    return regularizer_state


def _infer_architecture(
    checkpoint: Mapping[str, Any],
    state: Mapping[str, torch.Tensor],
    gradient_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    config_value = checkpoint.get("config", {})
    config = config_value if isinstance(config_value, Mapping) else {}
    analysis_weight = state.get("analysis.weight")
    if analysis_weight is None:
        raise ValueError("Checkpoint is missing analysis.weight.")
    block_indices = [
        int(match.group(1))
        for key in state
        if (match := re.match(r"macro_blocks\.(\d+)\.", key)) is not None
    ]
    scale_indices = [
        int(match.group(1))
        for key in state
        if (match := re.match(r"macro_blocks\.\d+\.down_operators\.(\d+)\.", key))
        is not None
    ]
    return {
        "num_features": int(
            gradient_parameters.get(
                "features",
                config.get("features", analysis_weight.shape[0]),
            )
        ),
        "num_macro_blocks": int(
            gradient_parameters.get(
                "macro_blocks",
                config.get("macro_blocks", max(block_indices, default=0) + 1),
            )
        ),
        "num_scales": int(
            gradient_parameters.get("num_scales", max(scale_indices, default=1) + 2)
        ),
        "kernel_size": int(gradient_parameters.get("kernel_size", analysis_weight.shape[-1])),
        "use_amp": bool(
            gradient_parameters.get("use_amp", config.get("use_amp", True))
        ),
        "energy_head_initialization_scale": float(
            gradient_parameters.get(
                "regularizer_head_initialization_scale",
                config.get("regularizer_head_initialization_scale", 0.2),
            )
        ),
    }


def _load_regularizer(
    checkpoint_path: str | Path,
    gradient_parameters: Mapping[str, Any],
    device: torch.device,
) -> tuple[TDVEnergy3D, Mapping[str, Any]]:
    checkpoint_value = torch.load(
        Path(checkpoint_path),
        map_location=device,
        weights_only=True,
    )
    if not isinstance(checkpoint_value, Mapping):
        raise ValueError("Checkpoint root must be a mapping.")
    state = _checkpoint_regularizer_state(checkpoint_value)
    regularizer = TDVEnergy3D(
        **_infer_architecture(checkpoint_value, state, gradient_parameters)
    ).to(device)
    regularizer.load_state_dict(state, strict=True)
    regularizer.eval()
    return regularizer, checkpoint_value


def _save_metric_figure(
    tol_update: torch.Tensor,
    ground_truth_nrmse: torch.Tensor | None,
    output_path: Path,
) -> None:
    plt = _pyplot()
    iterations = np.arange(1, tol_update.numel() + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    axes[0].plot(
        iterations,
        tol_update.detach().float().cpu().numpy(),
        color="tab:blue",
        label="tol_update",
    )
    axes[0].set(title="Update tolerance", xlabel="Iteration", ylabel="NRMSE")
    if ground_truth_nrmse is not None:
        axes[1].plot(
            iterations,
            ground_truth_nrmse.detach().float().cpu().numpy(),
            color="tab:orange",
            label="Ground-truth NRMSE",
        )
    else:
        axes[1].text(
            0.5,
            0.5,
            "Unavailable without chi.mat",
            horizontalalignment="center",
            verticalalignment="center",
        )
    axes[1].set(title="Ground-truth NRMSE", xlabel="Iteration", ylabel="NRMSE")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _save_prediction_only_figure(prediction: torch.Tensor, output_path: Path) -> None:
    plt = _pyplot()
    array = prediction.detach().float().cpu().numpy()[0, 0]
    display_planes = _cosmos_display_planes(array)
    planes = (
        ("Sagittal", "Y", "Z", 0),
        ("Coronal", "Z", "X", 1),
        ("Axial", "Y", "X", 2),
    )
    figure = plt.figure(figsize=(18, 13), constrained_layout=True)
    grid = figure.add_gridspec(3, 2, width_ratios=(1.0, 0.07))
    susceptibility_image = None
    for row, (plane_name, x_label, y_label, plane_index) in enumerate(planes):
        axis = figure.add_subplot(grid[row, 0])
        susceptibility_image = axis.imshow(
            display_planes[plane_index],
            cmap="gray",
            vmin=-0.1,
            vmax=0.1,
        )
        if row == 0:
            axis.set_title(r"TDV-QSM prediction $X_S$")
        axis.set_ylabel(f"{plane_name}\n{y_label}")
        axis.set_xlabel(x_label)
        axis.set_xticks([])
        axis.set_yticks([])
    assert susceptibility_image is not None
    colorbar = figure.colorbar(
        susceptibility_image,
        cax=figure.add_subplot(grid[:, 1]),
    )
    colorbar.set_label("Susceptibility (source units)")
    figure.suptitle("COSMOS TDV-QSM learned-regularizer evaluation", fontsize=14)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _write_metric_history(output: RegularizerEvaluationOutput, output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        columns = ["iteration", "tol_update"]
        if output.ground_truth_nrmse is not None:
            columns.append("ground_truth_nrmse")
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index in range(output.tol_update.numel()):
            row: dict[str, float] = {
                "iteration": float(index + 1),
                "tol_update": float(output.tol_update[index].cpu()),
            }
            if output.ground_truth_nrmse is not None:
                row["ground_truth_nrmse"] = float(
                    output.ground_truth_nrmse[index].cpu()
                )
            writer.writerow(row)


def evaluate_learned_regularizers(
    model_directory_path: str | Path,
    dataset_directory_path: str | Path,
    gradient_parameters: Mapping[str, Any],
    total_iterations: int,
    taus: Mapping[str, float] | Sequence[float] | None = None,
    *,
    snr: float | None = None,
) -> RegularizerEvaluationOutput:
    if not isinstance(gradient_parameters, Mapping):
        raise TypeError("gradient_parameters must be a mapping.")
    device = torch.device(
        gradient_parameters.get(
            "device",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
    )
    model_directory = Path(model_directory_path)
    if not model_directory.is_dir():
        raise NotADirectoryError(f"Model directory does not exist: {model_directory}")
    checkpoint_path = model_directory / "checkpoint.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {checkpoint_path}")
    report = _load_report(model_directory / "report.json")
    regularizer, checkpoint = _load_regularizer(
        checkpoint_path,
        gradient_parameters,
        device,
    )
    data = _load_evaluation_data(dataset_directory_path, device)
    checkpoint_config_value = checkpoint.get("config", {})
    checkpoint_config = (
        checkpoint_config_value if isinstance(checkpoint_config_value, Mapping) else {}
    )
    simulation_snr = float(
        snr
        if snr is not None
        else gradient_parameters.get("snr", checkpoint_config.get("snr", 70.0))
    )
    if not math.isfinite(simulation_snr) or simulation_snr <= 0.0:
        raise ValueError("snr must be finite and positive.")
    voxel_size = tuple(
        float(value)
        for value in gradient_parameters.get(
            "voxel_size_zyx",
            checkpoint_config.get("voxel_size_zyx", (1.0, 1.0, 1.0)),
        )
    )
    b0_direction = tuple(
        float(value)
        for value in gradient_parameters.get(
            "b0_direction_zyx",
            checkpoint_config.get("b0_direction_zyx", (0.0, 0.0, 1.0)),
        )
    )
    kernel = build_dipole_kernel(
        data.brain_mask.shape[-3:],
        voxel_size,
        b0_direction,
        device=device,
    )
    operator = DipoleOperator3D(kernel)
    if data.phase_was_provided:
        if data.local_field is None:
            raise RuntimeError("phase.mat was detected but phase data was not loaded.")
        local_field = data.local_field.float()
    else:
        if data.simulation_susceptibility is None:
            raise FileNotFoundError(
                "Without phase.mat, chi.mat or chi_cosmos.mat is required for simulation."
            )
        local_field = simulate_noisy_local_field(
            data.simulation_susceptibility,
            data.weight,
            data.brain_mask,
            operator,
            snr=simulation_snr,
            phase_scale=float(
                gradient_parameters.get(
                    "phase_scale",
                    checkpoint_config.get("phase_scale", 1.0),
                )
            ),
            seed=int(
                gradient_parameters.get("seed", checkpoint_config.get("seed", 0))
            ),
        )
        data = replace(data, local_field=local_field)
    initial = (
        data.initial.float()
        if data.initial is not None
        else initial_backprojection(local_field, data.weight, data.brain_mask, operator)
    )
    resolved_taus = _report_taus(report) if taus is None else _resolve_taus(taus)
    output = iterate_learned_regularizers(
        regularizer,
        operator,
        data,
        initial,
        total_iterations=total_iterations,
        taus=resolved_taus,
        mask_state_each_step=bool(
            gradient_parameters.get(
                "mask_state_each_step",
                checkpoint_config.get("mask_state_each_step", True),
            )
        ),
    )
    output_directory = Path(
        gradient_parameters.get(
            "output_dir",
            model_directory / "regularizer_evaluation",
        )
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    output = replace(output, output_directory=output_directory)
    _write_metric_history(output, output_directory / "metrics.csv")
    savemat(
        output_directory / "chi_pred.mat",
        {"chi_pred": output.chi_pred.detach().float().cpu().numpy()[0, 0]},
    )
    _save_metric_figure(
        output.tol_update,
        output.ground_truth_nrmse,
        output_directory / "metrics.png",
    )
    if data.ground_truth is not None:
        prediction = mask_and_reference(
            output.chi_pred,
            data.brain_mask,
            convention="already_referenced",
        )
        target = mask_and_reference(
            data.ground_truth,
            data.brain_mask,
            convention="already_referenced",
        )
        save_reconstruction_figure(
            initial,
            prediction,
            target,
            output_directory / "chi_pred.png",
            nrmse_value=float(output.ground_truth_nrmse[-1].cpu()),
            prediction_title=r"TDV-QSM prediction $X_S$",
            diagnostic_title="COSMOS TDV-QSM learned-regularizer evaluation",
        )
    else:
        _save_prediction_only_figure(output.chi_pred, output_directory / "chi_pred.png")
    return output


def _parse_gradient_parameters(value: str) -> dict[str, Any]:
    path = Path(value)
    parsed = json.loads(path.read_text(encoding="utf-8") if path.is_file() else value)
    if not isinstance(parsed, dict):
        raise ValueError("gradient parameters must decode to a JSON object.")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_directory_path", type=Path)
    parser.add_argument("dataset_directory_path", type=Path)
    parser.add_argument("--gradient-parameters", default="{}")
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--snr", type=float, default=None)
    parser.add_argument(
        "--taus",
        type=float,
        nargs=2,
        default=None,
        metavar=("TAU_REGULARIZER", "TAU_DATA"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    arguments = parser.parse_args()
    gradient_parameters = _parse_gradient_parameters(arguments.gradient_parameters)
    if arguments.output_dir is not None:
        gradient_parameters["output_dir"] = arguments.output_dir
    if arguments.device is not None:
        gradient_parameters["device"] = arguments.device
    output = evaluate_learned_regularizers(
        arguments.model_directory_path,
        arguments.dataset_directory_path,
        gradient_parameters,
        arguments.iterations,
        arguments.taus,
        snr=arguments.snr,
    )
    print(f"Regularizer tau: {output.regularizer_tau:.6e}")
    print(f"Data tau:        {output.data_tau:.6e}")
    print(f"Final tol_update: {float(output.tol_update[-1].cpu()):.6e}")
    if output.ground_truth_nrmse is not None:
        print(
            "Final ground-truth NRMSE: "
            f"{float(output.ground_truth_nrmse[-1].cpu()):.6e}"
        )
    print(f"Wrote {output.output_directory / 'chi_pred.png'}")


if __name__ == "__main__":
    main()
