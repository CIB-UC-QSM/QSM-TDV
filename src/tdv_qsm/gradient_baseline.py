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

from tdv_qsm.losses import mask_and_reference, nrmse
from tdv_qsm.operators.dipole import QSMOperator, build_dipole_kernel
from tdv_qsm.train import (
    SingleVolume,
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
    step_size: torch.Tensor


@dataclass(frozen=True)
class GradientBaselineConfig:
    """Configuration for a same-volume COSMOS baseline evaluation."""

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
    dipole_kernel: torch.Tensor | None = None,
    num_steps: int,
    step_size: float,
    mask_state_each_step: bool = True,
) -> GradientDescentOutput:
    """Run fixed-step descent on ``0.5 ||W(A chi - b)||^2`` only.

    No learned regularizer or learned coefficient participates in the update.
    The returned histories have shape ``[num_steps + 1, B]`` and include the
    initial and final states without retaining the 3-D intermediate states.
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

    for _ in range(num_steps):
        predicted_field = operator.forward(chi.float(), dipole_kernel)
        residual = predicted_field - local_field
        data_force = operator.adjoint(weight.square() * residual, dipole_kernel).float()
        data_energies.append(_data_energy(weight, residual).detach())
        gradient_norms.append(_norm(data_force).detach())
        state_norms.append(_norm(chi).detach())
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
    return GradientDescentOutput(
        susceptibility=chi,
        predicted_field=predicted_field,
        data_energy=torch.stack(data_energies),
        data_gradient_norm=torch.stack(gradient_norms),
        state_norm=torch.stack(state_norms),
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
    rows = [
        {
            "iteration": iteration,
            "data_energy": float(output.data_energy[iteration].mean().cpu()),
            "data_gradient_norm": float(output.data_gradient_norm[iteration].mean().cpu()),
            "state_norm": float(output.state_norm[iteration].mean().cpu()),
        }
        for iteration in iterations
    ]
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
    axes[1].plot(
        iterations,
        [row["data_gradient_norm"] for row in rows],
        color="tab:orange",
    )
    axes[1].set(title="Data-gradient norm", xlabel="Iteration")
    for axis in axes:
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
    figure.savefig(output_dir / "history.png", dpi=160)
    plt.close(figure)


def evaluate_gradient_baseline(
    sample: SingleVolume,
    config: GradientBaselineConfig,
    output_dir: Path,
) -> tuple[GradientDescentOutput, dict[str, float]]:
    """Evaluate a data-only baseline on one COSMOS-style synthetic field."""

    output_dir.mkdir(parents=True, exist_ok=True)
    device = sample.susceptibility.device
    kernel = build_dipole_kernel(
        sample.susceptibility.shape[-3:],
        config.voxel_size_zyx,
        config.b0_direction_zyx,
        device=device,
    )
    operator = QSMOperator(kernel)
    weight = magnitude_weight(sample.magnitude)
    local_field = simulate_noisy_local_field(
        sample.susceptibility,
        sample.magnitude,
        sample.brain_mask,
        operator,
        snr=config.snr,
        phase_scale=config.phase_scale,
        seed=config.seed,
    )
    initial = initial_backprojection(local_field, weight, sample.brain_mask, operator)
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
            brain_mask=sample.brain_mask,
            num_steps=config.num_steps,
            step_size=resolved_step_size,
            mask_state_each_step=config.mask_state_each_step,
        )
        target = mask_and_reference(
            sample.susceptibility,
            sample.brain_mask,
            convention=config.reference_convention,
        )
        initial_evaluation = mask_and_reference(
            initial,
            sample.brain_mask,
            convention=config.reference_convention,
        )
        prediction = mask_and_reference(
            output.susceptibility,
            sample.brain_mask,
            convention=config.reference_convention,
        )
        initial_nrmse = nrmse(initial_evaluation, target)
        final_nrmse = nrmse(prediction, target)

    metrics = {
        "initial_nrmse": float(initial_nrmse.cpu()),
        "final_nrmse": float(final_nrmse.cpu()),
        "initial_data_energy": float(output.data_energy[0].mean().cpu()),
        "final_data_energy": float(output.data_energy[-1].mean().cpu()),
        "final_data_gradient_norm": float(output.data_gradient_norm[-1].mean().cpu()),
        "resolved_step_size": float(resolved_step_size),
        "conservative_lipschitz_bound": float(lipschitz_bound),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("Non-finite baseline metric.")

    _write_history(output, output_dir)
    save_reconstruction_figure(
        initial_evaluation,
        prediction,
        target,
        output_dir / "reconstruction.png",
        nrmse_value=metrics["final_nrmse"],
        prediction_title=r"Gradient-descent QSM $X_S$",
        diagnostic_title="COSMOS data-only gradient-descent baseline",
    )
    torch.save(
        {
            "initial": initial.detach().cpu(),
            "local_field": local_field.detach().cpu(),
            "magnitude_weight": weight.detach().cpu(),
            "brain_mask": sample.brain_mask.detach().cpu(),
            "susceptibility": output.susceptibility.detach().cpu(),
            "predicted_field": output.predicted_field.detach().cpu(),
            "ground_truth": sample.susceptibility.detach().cpu(),
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
        "initialization": "masked A^H W^2 b backprojection",
        "step_size_rule": step_size_rule,
        "physical_model": "b = F^H D F chi + eta; periodic unitary FFT, no padding/cropping/TKD",
        "weight_rule": "Stored W (not sqrt(W)) = sqrt(2) * dimensionless magn; no normalization, clipping, or mask multiplication; zero magn gives W=0",
        "susceptibility_reference_convention": config.reference_convention,
        "noise_rule": "One complex Gaussian signal-noise realization; real/imag std = mean(magn inside brain_mask) / SNR",
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
        help="COSMOS directory, MAT, or NPZ file.",
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
    sample = load_single_volume(arguments.data, device)
    _, metrics = evaluate_gradient_baseline(
        sample,
        _configuration_from_args(arguments),
        arguments.output_dir,
    )
    print(f"Initial NRMSE: {metrics['initial_nrmse']:.6e}")
    print(f"Final NRMSE:   {metrics['final_nrmse']:.6e}")
    print(f"Step size:     {metrics['resolved_step_size']:.6e}")
    print(f"Wrote {arguments.output_dir / 'reconstruction.png'}")


if __name__ == "__main__":
    main()
