"""Data-only gradient-descent baseline for the 3-D QSM extension.

This module deliberately does not construct or load a TDV regularizer.  It
minimizes the magnitude-weighted QSM data term

    D_W(chi; b) = 0.5 * ||W (A chi - b)||_2^2

with the exact gradient ``A^H W^2 (A chi - b)`` and a fixed iteration count.
The physical operator has periodic Fourier boundary conditions, exactly as in
the learned reconstruction path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from tdv_qsm.evaluate_regularizers import EvaluationData, _load_evaluation_data
from tdv_qsm.losses import mask_and_reference, nrmse
from tdv_qsm.operators.dipole import QSMOperator, build_dipole_kernel
from tdv_qsm.train import (
    SingleVolume,
    _cosmos_display_planes,
    _pyplot,
    initial_backprojection,
    load_single_volume,
    magnitude_weight,
    save_reconstruction_figure,
    simulate_noisy_local_field,
)


@dataclass(frozen=True)
class GradientDescentOutput:
    """Final data-only reconstruction and compact per-iteration diagnostics."""

    susceptibility: torch.Tensor
    predicted_field: torch.Tensor
    data_energy: torch.Tensor
    data_gradient_norm: torch.Tensor
    state_norm: torch.Tensor
    ground_truth_nrmse: torch.Tensor | None
    step_size: torch.Tensor


@dataclass(frozen=True)
class GradientBaselineConfig:
    """Configuration for a single-volume baseline evaluation."""

    num_steps: int = 100
    step_size: float | None = None
    snr: float = 70.0
    seed: int = 0
    mask_state_each_step: bool = True
    reference_convention: str = "already_referenced"
    phase_scale: float = 1.0
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0)
    b0_direction_zyx: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if self.num_steps < 1:
            raise ValueError("num_steps must be at least one.")
        if self.step_size is not None and (
            not math.isfinite(self.step_size) or self.step_size <= 0.0
        ):
            raise ValueError("step_size must be finite and positive when supplied.")
        if not math.isfinite(self.snr) or self.snr <= 0.0:
            raise ValueError("snr must be finite and positive.")
        if not math.isfinite(self.phase_scale) or self.phase_scale <= 0.0:
            raise ValueError("phase_scale must be finite and positive.")
        if self.reference_convention not in ("already_referenced", "masked_mean_zero"):
            raise ValueError("Unsupported susceptibility reference_convention.")
        if len(self.voxel_size_zyx) != 3 or any(
            not math.isfinite(value) or value <= 0.0 for value in self.voxel_size_zyx
        ):
            raise ValueError("voxel_size_zyx must contain three finite positive values.")
        if len(self.b0_direction_zyx) != 3 or any(
            not math.isfinite(value) for value in self.b0_direction_zyx
        ):
            raise ValueError("b0_direction_zyx must contain three finite values.")
        if sum(value * value for value in self.b0_direction_zyx) == 0.0:
            raise ValueError("b0_direction_zyx must be nonzero.")


def _require_image(
    name: str,
    value: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    if value.ndim != 5 or value.shape[1] != 1:
        raise ValueError(f"{name} must have shape [B, 1, Z, Y, X].")
    if value.shape != reference.shape:
        raise ValueError(f"{name} must have the same shape as local_field.")
    value = value.float()
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values.")
    return value


def _validate_inputs(
    local_field: torch.Tensor,
    magnitude_weight_map: torch.Tensor,
    initial: torch.Tensor,
    brain_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if local_field.ndim != 5 or local_field.shape[1] != 1:
        raise ValueError("local_field must have shape [B, 1, Z, Y, X].")
    local_field = local_field.float()
    if not torch.isfinite(local_field).all():
        raise ValueError("local_field must contain only finite values.")
    initial = _require_image("initial", initial, local_field)
    try:
        weight = torch.broadcast_to(magnitude_weight_map.float(), local_field.shape)
    except RuntimeError as error:
        raise ValueError("magnitude_weight must broadcast to local_field.") from error
    if not torch.isfinite(weight).all():
        raise ValueError("Magnitude weights must be finite.")
    if torch.any(weight < 0.0):
        raise ValueError("Magnitude weights must be nonnegative.")
    if brain_mask is not None:
        brain_mask = _require_image("brain_mask", brain_mask, local_field)
        if torch.any(brain_mask < 0.0):
            raise ValueError("brain_mask must be nonnegative.")
        if torch.any(brain_mask.flatten(1).sum(dim=1) <= 0.0):
            raise ValueError("brain_mask must be nonempty for every sample.")
    return local_field, weight, initial, brain_mask


def _data_energy(weight: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    return 0.5 * (weight * residual).square().flatten(1).sum(dim=1)


def _norm(value: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(value.float().flatten(1), ord=2, dim=1)


def gradient_descent_qsm(
    local_field: torch.Tensor,
    magnitude_weight_map: torch.Tensor,
    initial: torch.Tensor,
    operator: QSMOperator,
    *,
    brain_mask: torch.Tensor | None = None,
    ground_truth: torch.Tensor | None = None,
    reference_convention: str = "already_referenced",
    dipole_kernel: torch.Tensor | None = None,
    num_steps: int,
    step_size: float,
    mask_state_each_step: bool = True,
) -> GradientDescentOutput:
    """Run fixed-step descent on ``0.5 ||W(A chi - b)||^2`` only.

    No learned regularizer or learned coefficient participates in the update.
    The physical histories have shape ``[num_steps + 1, B]``.  When ground
    truth is supplied, ``ground_truth_nrmse`` has shape ``[num_steps + 1]`` and
    contains the batch-mean masked/referenced NRMSE, including iteration zero.
    No 3-D intermediate states are retained.
    """

    if num_steps < 1:
        raise ValueError("num_steps must be at least one.")
    if not math.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("step_size must be finite and positive.")
    local_field, weight, chi, brain_mask = _validate_inputs(
        local_field,
        magnitude_weight_map,
        initial,
        brain_mask,
    )
    step = torch.tensor(float(step_size), dtype=torch.float32, device=chi.device)
    data_energies: list[torch.Tensor] = []
    gradient_norms: list[torch.Tensor] = []
    state_norms: list[torch.Tensor] = []
    ground_truth_nrmse_history: list[torch.Tensor] = []
    target = None
    evaluation_mask = brain_mask
    if ground_truth is not None:
        ground_truth = _require_image("ground_truth", ground_truth, local_field)
        if evaluation_mask is None:
            evaluation_mask = torch.ones_like(local_field)
        target = mask_and_reference(
            ground_truth,
            evaluation_mask,
            convention=reference_convention,
        )

    def record_ground_truth_nrmse(state: torch.Tensor) -> None:
        if target is None or evaluation_mask is None:
            return
        evaluation = mask_and_reference(
            state,
            evaluation_mask,
            convention=reference_convention,
        )
        ground_truth_nrmse_history.append(nrmse(evaluation, target).detach())

    for _ in range(num_steps):
        predicted_field = operator.forward(chi.float(), dipole_kernel)
        residual = predicted_field - local_field
        data_force = operator.adjoint(weight.square() * residual, dipole_kernel).float()
        data_energies.append(_data_energy(weight, residual).detach())
        gradient_norms.append(_norm(data_force).detach())
        state_norms.append(_norm(chi).detach())
        record_ground_truth_nrmse(chi)
        chi = chi - step * data_force
        if mask_state_each_step and brain_mask is not None:
            chi = chi * brain_mask
        chi = chi.float()
        if not torch.isfinite(chi).all():
            raise FloatingPointError(
                "Non-finite gradient-descent state. Reduce --step-size or use the "
                "automatic conservative step size."
            )

    predicted_field = operator.forward(chi, dipole_kernel).float()
    final_residual = predicted_field - local_field
    final_force = operator.adjoint(weight.square() * final_residual, dipole_kernel).float()
    data_energies.append(_data_energy(weight, final_residual).detach())
    gradient_norms.append(_norm(final_force).detach())
    state_norms.append(_norm(chi).detach())
    record_ground_truth_nrmse(chi)
    return GradientDescentOutput(
        susceptibility=chi,
        predicted_field=predicted_field,
        data_energy=torch.stack(data_energies),
        data_gradient_norm=torch.stack(gradient_norms),
        state_norm=torch.stack(state_norms),
        ground_truth_nrmse=(
            torch.stack(ground_truth_nrmse_history)
            if ground_truth_nrmse_history
            else None
        ),
        step_size=step,
    )


def conservative_lipschitz_bound(
    magnitude_weight_map: torch.Tensor,
    dipole_kernel: torch.Tensor,
) -> float:
    """Return ``||W||_inf^2 ||D||_inf^2``, an upper bound for the Hessian norm."""

    weight = magnitude_weight_map.float()
    kernel = dipole_kernel.to(dtype=torch.complex64 if dipole_kernel.is_complex() else torch.float32)
    if not torch.isfinite(weight).all() or torch.any(weight < 0.0):
        raise ValueError("Magnitude weights must be finite and nonnegative.")
    if not torch.isfinite(kernel).all():
        raise ValueError("dipole_kernel must be finite.")
    bound = weight.abs().amax().square() * kernel.abs().amax().square()
    return float(bound.detach().cpu())


def _write_history(output: GradientDescentOutput, output_dir: Path) -> None:
    iterations = list(range(output.data_energy.shape[0]))
    if (
        output.ground_truth_nrmse is not None
        and output.ground_truth_nrmse.numel() != len(iterations)
    ):
        raise ValueError("ground_truth_nrmse must contain one value per iteration.")
    rows: list[dict[str, float | int]] = []
    for iteration in iterations:
        row: dict[str, float | int] = {
            "iteration": iteration,
            "data_energy": float(output.data_energy[iteration].mean().cpu()),
            "data_gradient_norm": float(output.data_gradient_norm[iteration].mean().cpu()),
            "state_norm": float(output.state_norm[iteration].mean().cpu()),
        }
        if output.ground_truth_nrmse is not None:
            row["ground_truth_nrmse"] = float(
                output.ground_truth_nrmse[iteration].cpu()
            )
        rows.append(row)
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt = _pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    axes[0].plot(iterations, [row["data_energy"] for row in rows], color="tab:blue")
    axes[0].set(
        title=r"Data energy $\frac{1}{2}\Vert W(A\chi-b)\Vert^2$",
        xlabel="Iteration",
    )
    if output.ground_truth_nrmse is not None:
        axes[1].plot(
            iterations,
            [row["ground_truth_nrmse"] for row in rows],
            color="tab:orange",
        )
    else:
        axes[1].text(
            0.5,
            0.5,
            "Unavailable without chi.mat",
            horizontalalignment="center",
            verticalalignment="center",
        )
    axes[1].set(
        title="Ground-truth NRMSE",
        xlabel="Iteration",
        ylabel="NRMSE",
    )
    axes[0].set_yscale("log")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.savefig(output_dir / "history.png", dpi=160)
    plt.close(figure)


def _save_prediction_only_figure(
    prediction: torch.Tensor,
    output_path: Path,
) -> None:
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
            axis.set_title(r"Gradient-descent QSM prediction $X_S$")
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
    figure.suptitle("Data-only gradient-descent baseline", fontsize=14)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _evaluation_data(
    source: SingleVolume | str | Path,
    device: torch.device,
) -> EvaluationData:
    if isinstance(source, (str, Path)) and Path(source).is_file():
        source = load_single_volume(Path(source), device)
    if not isinstance(source, SingleVolume):
        return _load_evaluation_data(source, device)
    susceptibility = source.susceptibility.to(device=device, dtype=torch.float32)
    brain_mask = source.brain_mask.to(device=device, dtype=torch.float32)
    weight = magnitude_weight(
        source.magnitude.to(device=device, dtype=torch.float32)
    )
    return EvaluationData(
        local_field=None,
        brain_mask=brain_mask,
        weight=weight,
        ground_truth=susceptibility,
        phase_was_provided=False,
        weight_source="magn",
        simulation_susceptibility=susceptibility,
    )


def evaluate_gradient_baseline(
    sample: SingleVolume | str | Path,
    config: GradientBaselineConfig,
    output_dir: Path,
    *,
    device: torch.device | str | None = None,
) -> tuple[GradientDescentOutput, dict[str, float]]:
    """Evaluate the data-only baseline on a legacy volume or dataset directory.

    Dataset directories follow the learned-regularizer evaluation contract:
    ``phase.mat`` is used directly when present, ``mask`` is the fallback for a
    missing ``magn``, ``chi`` is optional ground truth, and ``initial`` is an
    optional supplied starting point.  A field is simulated from ``chi`` only
    when no phase file is available.
    """

    if device is None:
        resolved_device = (
            sample.susceptibility.device
            if isinstance(sample, SingleVolume)
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    else:
        resolved_device = torch.device(device)
    data = _evaluation_data(sample, resolved_device)
    output_dir.mkdir(parents=True, exist_ok=True)
    kernel = build_dipole_kernel(
        data.brain_mask.shape[-3:],
        config.voxel_size_zyx,
        config.b0_direction_zyx,
        device=resolved_device,
    )
    operator = QSMOperator(kernel)
    weight = data.weight.float()
    if data.phase_was_provided:
        if data.local_field is None:
            raise RuntimeError("phase.mat was detected but phase data was not loaded.")
        local_field = data.local_field.float()
        field_source = "phase.mat"
        noise_rule = "Not applied; phase.mat was loaded directly as the local field"
    else:
        if data.simulation_susceptibility is None:
            raise FileNotFoundError(
                "Without phase.mat, chi.mat or chi_cosmos.mat is required for simulation."
            )
        local_field = simulate_noisy_local_field(
            data.simulation_susceptibility,
            weight,
            data.brain_mask,
            operator,
            snr=config.snr,
            phase_scale=config.phase_scale,
            seed=config.seed,
        )
        field_source = "simulation from supplied susceptibility"
        noise_rule = (
            "One complex Gaussian signal-noise realization; real/imag std = "
            "maximum magn inside brain_mask / SNR"
        )
    initial = (
        data.initial.float()
        if data.initial is not None
        else initial_backprojection(local_field, weight, data.brain_mask, operator)
    )
    initialization_rule = (
        "supplied initial.mat"
        if data.initial is not None
        else "masked A^H W^2 b backprojection"
    )
    lipschitz_bound = conservative_lipschitz_bound(weight, kernel)
    if config.step_size is None:
        resolved_step_size = 1.0 / lipschitz_bound if lipschitz_bound > 0.0 else 1.0
        step_size_rule = "automatic: 1 / (||W||_inf^2 ||D||_inf^2)"
    else:
        resolved_step_size = config.step_size
        step_size_rule = "explicit command-line/config value"

    with torch.no_grad():
        output = gradient_descent_qsm(
            local_field,
            weight,
            initial,
            operator,
            brain_mask=data.brain_mask,
            ground_truth=data.ground_truth,
            reference_convention=config.reference_convention,
            num_steps=config.num_steps,
            step_size=resolved_step_size,
            mask_state_each_step=config.mask_state_each_step,
        )
        initial_evaluation = mask_and_reference(
            initial,
            data.brain_mask,
            convention=config.reference_convention,
        )
        prediction = mask_and_reference(
            output.susceptibility,
            data.brain_mask,
            convention=config.reference_convention,
        )

    metrics = {
        "initial_data_energy": float(output.data_energy[0].mean().cpu()),
        "final_data_energy": float(output.data_energy[-1].mean().cpu()),
        "final_data_gradient_norm": float(output.data_gradient_norm[-1].mean().cpu()),
        "resolved_step_size": float(resolved_step_size),
        "conservative_lipschitz_bound": float(lipschitz_bound),
    }
    target = None
    if data.ground_truth is not None:
        if output.ground_truth_nrmse is None:
            raise RuntimeError("Ground-truth NRMSE history was not recorded.")
        with torch.no_grad():
            target = mask_and_reference(
                data.ground_truth,
                data.brain_mask,
                convention=config.reference_convention,
            )
        metrics["initial_nrmse"] = float(output.ground_truth_nrmse[0].cpu())
        metrics["final_nrmse"] = float(output.ground_truth_nrmse[-1].cpu())
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("Non-finite baseline metric.")

    _write_history(output, output_dir)
    if target is None:
        _save_prediction_only_figure(prediction, output_dir / "reconstruction.png")
    else:
        save_reconstruction_figure(
            initial_evaluation,
            prediction,
            target,
            output_dir / "reconstruction.png",
            nrmse_value=metrics["final_nrmse"],
            prediction_title=r"Gradient-descent QSM $X_S$",
            diagnostic_title="Data-only gradient-descent baseline",
        )
    torch.save(
        {
            "initial": initial.detach().cpu(),
            "local_field": local_field.detach().cpu(),
            "magnitude_weight": weight.detach().cpu(),
            "brain_mask": data.brain_mask.detach().cpu(),
            "susceptibility": output.susceptibility.detach().cpu(),
            "predicted_field": output.predicted_field.detach().cpu(),
            "ground_truth": (
                data.ground_truth.detach().cpu()
                if data.ground_truth is not None
                else None
            ),
            "ground_truth_nrmse": (
                output.ground_truth_nrmse.detach().cpu()
                if output.ground_truth_nrmse is not None
                else None
            ),
            "dipole_kernel": kernel.detach().cpu(),
            "config": asdict(config),
            "metrics": metrics,
        },
        output_dir / "reconstruction.pt",
    )
    report = {
        "evaluation_kind": "single-volume data-only gradient-descent baseline",
        "regularizer": "none",
        "objective": "0.5 * ||W (A chi - b)||_2^2",
        "data_gradient": "A^H W^2 (A chi - b)",
        "update": "chi_(s+1) = chi_s - step_size * data_gradient",
        "selection_rule": "fixed iteration count; ground truth is not used for stopping",
        "initialization": initialization_rule,
        "step_size_rule": step_size_rule,
        "physical_model": "b = F^H D F chi + eta; periodic unitary FFT, no padding/cropping/TKD",
        "field_source": field_source,
        "weight_source": data.weight_source,
        "weight_rule": (
            "Stored W (not sqrt(W)) = dimensionless magn; no normalization, clipping, "
            "or mask multiplication; zero magn gives W=0"
            if data.weight_source == "magn"
            else "Stored W (not sqrt(W)) = mask because magn.mat was absent"
        ),
        "susceptibility_reference_convention": config.reference_convention,
        "noise_rule": noise_rule,
        "source_provenance": "The baseline uses project-specific 3-D QSM physics and medical-volume handling; it contains no TDV regularizer",
        "metrics": metrics,
        "config": asdict(config),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return output, metrics


def _configuration_from_args(arguments: argparse.Namespace) -> GradientBaselineConfig:
    return GradientBaselineConfig(
        num_steps=arguments.steps,
        step_size=arguments.step_size,
        snr=arguments.snr,
        seed=arguments.seed,
        mask_state_each_step=not arguments.no_step_mask,
        reference_convention=arguments.reference_convention,
        phase_scale=arguments.phase_scale,
        voxel_size_zyx=tuple(arguments.voxel_size),
        b0_direction_zyx=tuple(arguments.b0_direction),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate data-only gradient descent as a QSM baseline.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/cosmos_data"),
        help=(
            "Dataset directory using the learned-evaluator phase/mask/magn/chi/initial "
            "MAT-file contract. Legacy self-contained COSMOS MAT/NPZ inputs are also "
            "accepted."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/cosmos-gradient-baseline"),
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--step-size",
        type=float,
        default=None,
        help=(
            "Fixed gradient-descent step. By default, use the conservative "
            "1/(||W||_inf^2 ||D||_inf^2) value."
        ),
    )
    parser.add_argument("--snr", type=float, default=70.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-step-mask", action="store_true")
    parser.add_argument(
        "--reference-convention",
        choices=("already_referenced", "masked_mean_zero"),
        default="already_referenced",
    )
    parser.add_argument("--phase-scale", type=float, default=1.0)
    parser.add_argument(
        "--voxel-size",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("VZ", "VY", "VX"),
    )
    parser.add_argument(
        "--b0-direction",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 1.0),
        metavar=("BZ", "BY", "BX"),
    )
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available, else CPU.")
    arguments = parser.parse_args()
    device = torch.device(arguments.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _, metrics = evaluate_gradient_baseline(
        arguments.data,
        _configuration_from_args(arguments),
        arguments.output_dir,
        device=device,
    )
    if "initial_nrmse" in metrics:
        print(f"Initial NRMSE: {metrics['initial_nrmse']:.6e}")
        print(f"Final NRMSE:   {metrics['final_nrmse']:.6e}")
    else:
        print("Ground-truth NRMSE: unavailable (chi.mat was not supplied)")
    print(f"Step size:     {metrics['resolved_step_size']:.6e}")
    print(f"Wrote {arguments.output_dir / 'reconstruction.png'}")


if __name__ == "__main__":
    main()
