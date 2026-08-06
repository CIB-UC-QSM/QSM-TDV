"""Train explicit TDV-QSM on one COSMOS-style volume.

The runner is deliberately a same-volume overfit diagnostic, not a claim of
cross-subject generalization.  It regenerates one independent complex-noise
realization per epoch and uses the exact configured field weight
``W = sqrt(2) * magn`` (with no mask folded into W).  Three-dimensional QSM,
medical-volume loading, NRMSE, and CUDA AMP are project extensions to TDV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from scipy.io import loadmat

from tdv_qsm.losses import mask_and_reference, nrmse, weighted_data_consistency_loss
from tdv_qsm.models.blocks import MacroBlock3D, MicroBlock3D
from tdv_qsm.models.energy import TDVEnergy3D
from tdv_qsm.models.explicit_tdv import ExplicitTDVQSM3D
from tdv_qsm.operators.convolution import AdjointConv3d
from tdv_qsm.operators.dipole import DipoleOperator3D, build_dipole_kernel


@dataclass(frozen=True)
class SingleVolume:
    """One unbatched COSMOS target converted to the canonical tensor layout."""

    susceptibility: torch.Tensor
    magnitude: torch.Tensor
    brain_mask: torch.Tensor


@dataclass(frozen=True)
class ModelArchitectureSummary:
    microblocks: int
    macroblocks: int
    convolutions: int
    total_parameters: int
    trainable_parameters: int


def summarize_model_architecture(
    model: torch.nn.Module,
) -> ModelArchitectureSummary:
    modules = tuple(model.modules())
    parameters = tuple(model.parameters())
    return ModelArchitectureSummary(
        microblocks=sum(isinstance(module, MicroBlock3D) for module in modules),
        macroblocks=sum(isinstance(module, MacroBlock3D) for module in modules),
        convolutions=sum(isinstance(module, AdjointConv3d) for module in modules),
        total_parameters=sum(parameter.numel() for parameter in parameters),
        trainable_parameters=sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
    )


def print_model_architecture(
    model: torch.nn.Module,
) -> ModelArchitectureSummary:
    summary = summarize_model_architecture(model)
    print("Model architecture:", flush=True)
    print(model, flush=True)
    print("Model summary:", flush=True)
    print(f"  Microblocks: {summary.microblocks:,}", flush=True)
    print(f"  Macroblocks: {summary.macroblocks:,}", flush=True)
    print(f"  Convolutions: {summary.convolutions:,}", flush=True)
    print(f"  Parameters: {summary.total_parameters:,}", flush=True)
    print(f"  Trainable parameters: {summary.trainable_parameters:,}", flush=True)
    return summary


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration recorded verbatim beside every training run."""

    learning_rate: float
    epochs: int = 100
    snr: float = 70.0
    seed: int = 0
    max_gradient_norm: float = 1.0
    data_consistency_weight: float = 0.0
    features: int = 1
    macro_blocks: int = 1
    num_steps: int = 1
    maximum_time: float = 0.25
    maximum_lambda: float = 1.0
    initial_raw_T: float = 2.0
    initial_raw_lambda: float = 0.0
    mask_state_each_step: bool = True
    checkpoint_force: bool = False
    use_amp: bool = True
    reference_convention: str = "already_referenced"
    regularizer_head_initialization_scale: float = 0.2
    phase_scale: float = 1.0
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0)
    b0_direction_zyx: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.snr <= 0.0 or self.learning_rate <= 0.0:
            raise ValueError("epochs, snr, and learning_rate must be positive.")
        if self.max_gradient_norm <= 0.0 or self.data_consistency_weight < 0.0:
            raise ValueError("gradient norm must be positive and DC weight nonnegative.")
        if self.features < 1 or self.macro_blocks < 1 or self.num_steps < 1:
            raise ValueError("features, macro_blocks, and num_steps must be positive.")
        if self.maximum_time <= 0.0 or self.maximum_lambda <= 0.0 or self.phase_scale <= 0.0:
            raise ValueError("maximum_time, maximum_lambda, and phase_scale must be positive.")
        if self.regularizer_head_initialization_scale <= 0.0:
            raise ValueError("regularizer_head_initialization_scale must be positive.")
        if self.reference_convention not in ("already_referenced", "masked_mean_zero"):
            raise ValueError("Unsupported susceptibility reference_convention.")


def _pyplot():
    """Import a non-interactive plotting backend only when an artifact is saved."""

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    return pyplot


def _array_to_volume(value: np.ndarray, name: str, device: torch.device) -> torch.Tensor:
    array = np.asarray(value, dtype=np.float32)
    array = np.squeeze(array)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3-D [Z, Y, X] array, got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite.")
    return torch.from_numpy(np.ascontiguousarray(array))[None, None].to(device=device)


def _find_key(values: Mapping[str, Any], candidates: tuple[str, ...], label: str) -> np.ndarray:
    for key in candidates:
        if key in values:
            return np.asarray(values[key])
    raise KeyError(f"Could not find {label}; tried {', '.join(candidates)}.")


def _load_mat(path: Path) -> Mapping[str, Any]:
    try:
        return loadmat(path)
    except NotImplementedError as error:
        raise ValueError(
            f"{path} is MATLAB v7.3/HDF5. Convert it to a standard MAT file or NPZ first."
        ) from error


def _resolve_data_path(location: Path) -> Path:
    if location.exists():
        return location
    # The repository bundles the requested data under its workspace root when
    # an external /cosmos_data mount is unavailable (for example in CI).
    if location == Path("/cosmos_data") and Path("cosmos_data").exists():
        return Path("cosmos_data")
    raise FileNotFoundError(f"Data path does not exist: {location}")


def load_single_volume(location: Path, device: torch.device) -> SingleVolume:
    """Load either COSMOS sibling MAT files or one self-contained MAT/NPZ file.

    The loader interprets source array axes as already being ``zyx``.  It does
    not normalize, clip, or otherwise change the supplied magnitude map.
    """

    location = _resolve_data_path(location)
    if location.is_dir():
        chi_path = next(
            (candidate for candidate in (location / "chi_cosmos.mat", location / "chi.mat") if candidate.is_file()),
            None,
        )
        magnitude_path = next(
            (candidate for candidate in (location / "magn.mat", location / "magnitude.mat") if candidate.is_file()),
            None,
        )
        mask_path = next(
            (candidate for candidate in (location / "msk.mat", location / "mask.mat") if candidate.is_file()),
            None,
        )
        if chi_path is None or magnitude_path is None or mask_path is None:
            raise FileNotFoundError(
                "A COSMOS directory requires chi_cosmos.mat (or chi.mat), "
                "magn.mat (or magnitude.mat), and msk.mat (or mask.mat)."
            )
        chi = _find_key(_load_mat(chi_path), ("chi_cosmos", "chi", "susceptibility"), "susceptibility")
        magnitude = _find_key(_load_mat(magnitude_path), ("magn", "magnitude"), "magnitude")
        mask = _find_key(_load_mat(mask_path), ("msk", "mask", "brain_mask"), "brain mask")
    elif location.suffix.lower() == ".npz":
        with np.load(location) as archive:
            chi = _find_key(archive, ("chi_cosmos", "chi", "susceptibility"), "susceptibility")
            magnitude = _find_key(archive, ("magn", "magnitude"), "magnitude")
            mask = _find_key(archive, ("msk", "mask", "brain_mask"), "brain mask")
    elif location.suffix.lower() == ".mat":
        values = _load_mat(location)
        try:
            chi = _find_key(values, ("chi_cosmos", "chi", "susceptibility"), "susceptibility")
            magnitude = _find_key(values, ("magn", "magnitude"), "magnitude")
            mask = _find_key(values, ("msk", "mask", "brain_mask"), "brain mask")
        except KeyError:
            # ``chi_cosmos.mat`` commonly has magnitude/mask sibling files.
            return load_single_volume(location.parent, device)
    else:
        raise ValueError("Input must be a COSMOS directory, a .mat file, or an .npz file.")

    susceptibility = _array_to_volume(chi, "susceptibility", device)
    magnitude_tensor = _array_to_volume(magnitude, "magnitude", device)
    brain_mask = _array_to_volume(mask, "brain_mask", device)
    if susceptibility.shape != magnitude_tensor.shape or susceptibility.shape != brain_mask.shape:
        raise ValueError("susceptibility, magnitude, and brain_mask must have matching spatial shapes.")
    if torch.any(magnitude_tensor < 0.0):
        raise ValueError("magnitude must be nonnegative to construct W.")
    if torch.any(brain_mask < 0.0):
        raise ValueError("brain_mask must be nonnegative.")
    return SingleVolume(susceptibility, magnitude_tensor, brain_mask)


def magnitude_weight(magnitude: torch.Tensor) -> torch.Tensor:
    """Return the stored diagonal ``W=magn`` (not ``sqrt(W)``).

    Input magnitudes are treated as dimensionless, are not normalized or
    clipped, and are not multiplied by a mask.  Zero magnitude maps to zero
    weight and therefore removes that residual's data-term contribution.
    """

    weight = math.sqrt(2.0) * magnitude.float()
    if not torch.isfinite(weight).all() or torch.any(weight < 0.0):
        raise ValueError("magn must be finite and nonnegative.")
    return weight


def simulate_noisy_local_field(
    susceptibility: torch.Tensor,
    magnitude: torch.Tensor,
    brain_mask: torch.Tensor,
    operator: DipoleOperator3D,
    *,
    snr: float,
    phase_scale: float,
    seed: int,
) -> torch.Tensor:
    """Simulate a noisy phase field with an epoch-specific complex-noise draw.

    ``b_clean=A(chi)`` is encoded as ``magn * exp(-i * phase_scale * b)``.
    Independent real and imaginary Gaussian noise components have standard
    deviation ``mean(magn inside mask) / SNR``.  The recovered field is
    ``-angle(signal+noise)/phase_scale``.  It is masked after phase recovery;
    this is distinct from the unchanged field-domain weight ``W``.
    """

    if snr <= 0.0 or phase_scale <= 0.0:
        raise ValueError("snr and phase_scale must be positive.")
    clean_field = operator.forward(susceptibility.float())
    in_mask = magnitude[brain_mask > 0.0]
    if in_mask.numel() == 0 or torch.all(in_mask == 0.0):
        raise ValueError("The brain mask must contain at least one positive magnitude voxel.")
    noise_std = in_mask.max().float() / float(snr)
    generator = torch.Generator(device=susceptibility.device)
    generator.manual_seed(int(seed))
    real_noise = torch.randn(
        magnitude.shape,
        dtype=torch.float32,
        device=magnitude.device,
        generator=generator,
    )
    imag_noise = torch.randn(
        magnitude.shape,
        dtype=torch.float32,
        device=magnitude.device,
        generator=generator,
    )
    phase = phase_scale * clean_field
    scale = torch.pi / phase.abs().max()

    signal = torch.polar(magnitude.float(), phase.float()*scale)
    noisy_signal = signal + torch.complex(noise_std * real_noise, noise_std * imag_noise)
    return (torch.angle(noisy_signal) / scale).float() * brain_mask.float()


def initial_backprojection(
    local_field: torch.Tensor,
    magnitude_weight_map: torch.Tensor,
    brain_mask: torch.Tensor,
    operator: DipoleOperator3D,
) -> torch.Tensor:
    """Use a masked weighted normal backprojection as the displayed ``X_0``."""

    return operator.adjoint(magnitude_weight_map.square() * local_field.float()) * brain_mask.float()


def _finite_or_raise(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Non-finite {name}.")


def _save_history(history: list[dict[str, float]], output_dir: Path) -> None:
    plt = _pyplot()
    columns = list(history[0])
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 3, figsize=(15, 3.8), constrained_layout=True)
    axes[0].plot(epochs, [row["loss"] for row in history], color="tab:blue")
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(epochs, [row["nrmse"] for row in history], color="tab:orange")
    axes[1].set(title="Training NRMSE", xlabel="Epoch", ylabel="NRMSE")
    axes[2].plot(
        epochs,
        [row["data_consistency_value"] for row in history],
        color="tab:green",
        label=r"Data consistency $\|Wr\|^2/N$",
    )
    axes[2].plot(
        epochs,
        [row["regularization_energy"] for row in history],
        color="tab:purple",
        label=r"TDV regularization $R/N$",
    )
    axes[2].set(title="Per-voxel energy terms", xlabel="Epoch", ylabel="Value")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.savefig(output_dir / "history.png", dpi=160)
    plt.close(figure)


def _cosmos_display_planes(value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract COSMOS display planes with the prescribed radiological rotation.

    The sagittal display retains its native orientation, the coronal display
    is rotated 180°, and the axial display 90° counter-clockwise.  This only
    changes figure orientation, never the reconstruction tensor or physical
    metadata.
    """

    z_size, y_size, x_size = value.shape
    sagittal = value[:, :, x_size // 2].T
    coronal = np.rot90(value[:, y_size // 2, :].T, k=2)
    axial = np.rot90(value[z_size // 2], k=1)
    return sagittal, coronal, axial


def save_reconstruction_figure(
    initial: torch.Tensor,
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    output_path: Path,
    *,
    nrmse_value: float | None = None,
    prediction_title: str = r"TDV-QSM prediction $X_S$",
    diagnostic_title: str = "COSMOS TDV-QSM overfit diagnostic",
) -> None:
    """Save a three-plane COSMOS-style reconstruction diagnostic figure.

    Susceptibility panels retain the fixed ``[-0.1, 0.1]`` display range.
    The final column is absolute error, displayed on the ``[0, 0.5]`` scale
    used by the supplied COSMOS diagnostic reference.
    """

    plt = _pyplot()
    arrays = [
        initial.detach().float().cpu().numpy()[0, 0],
        prediction.detach().float().cpu().numpy()[0, 0],
        ground_truth.detach().float().cpu().numpy()[0, 0],
    ]
    absolute_error = np.abs(arrays[1] - arrays[2])
    # The axis labels account for the requested rotations, rather than merely
    # rotating the displayed pixels while retaining stale anatomical axes.
    planes = (
        ("Sagittal", "Y", "Z", 0),
        ("Coronal", "Z", "X", 1),
        ("Axial", "Y", "X", 2),
    )
    display_planes = tuple(_cosmos_display_planes(array) for array in arrays)
    error_planes = _cosmos_display_planes(absolute_error)
    titles = ("Input $X_0$", prediction_title, "Ground truth $X_{gt}$")
    figure = plt.figure(figsize=(18, 13), constrained_layout=True)
    grid = figure.add_gridspec(3, 6, width_ratios=(1.0, 1.0, 1.0, 0.07, 1.0, 0.07))
    susceptibility_image = None
    error_image = None
    for row, (plane_name, x_label, y_label, plane_index) in enumerate(planes):
        for column, title in enumerate(titles):
            axis = figure.add_subplot(grid[row, column])
            susceptibility_image = axis.imshow(
                display_planes[column][plane_index],
                cmap="gray",
                vmin=-0.1,
                vmax=0.1,
            )
            if row == 0:
                axis.set_title(title)
            if column == 0:
                axis.set_ylabel(f"{plane_name}\n{y_label}")
            axis.set_xlabel(x_label)
            axis.set_xticks([])
            axis.set_yticks([])
        error_axis = figure.add_subplot(grid[row, 4])
        error_image = error_axis.imshow(
            error_planes[plane_index],
            cmap="magma",
            vmin=0.0,
            vmax=0.5,
        )
        if row == 0:
            error_axis.set_title("Absolute error")
        error_axis.set_xlabel(x_label)
        error_axis.set_xticks([])
        error_axis.set_yticks([])
    assert susceptibility_image is not None and error_image is not None
    susceptibility_colorbar = figure.colorbar(
        susceptibility_image,
        cax=figure.add_subplot(grid[:, 3]),
    )
    susceptibility_colorbar.set_label("Susceptibility (source units)")
    error_colorbar = figure.colorbar(error_image, cax=figure.add_subplot(grid[:, 5]))
    error_colorbar.set_label("Absolute error")
    if nrmse_value is not None:
        figure.suptitle(
            f"{diagnostic_title} — masked NRMSE = {nrmse_value:.5f}",
            fontsize=14,
        )
    else:
        figure.suptitle(diagnostic_title, fontsize=14)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def train_single_volume(
    sample: SingleVolume,
    config: TrainingConfig,
    output_dir: Path,
) -> tuple[ExplicitTDVQSM3D, list[dict[str, float]]]:
    """Run the configured one-volume TDV training diagnostic."""

    output_dir.mkdir(parents=True, exist_ok=True)
    device = sample.susceptibility.device
    torch.manual_seed(config.seed)
    kernel = build_dipole_kernel(
        sample.susceptibility.shape[-3:],
        config.voxel_size_zyx,
        config.b0_direction_zyx,
        device=device,
    )
    operator = DipoleOperator3D(kernel)
    regularizer = TDVEnergy3D(
        num_features=config.features,
        num_macro_blocks=config.macro_blocks,
        use_amp=config.use_amp,
        energy_head_initialization_scale=config.regularizer_head_initialization_scale,
    ).to(device)
    model = ExplicitTDVQSM3D(
        regularizer,
        operator,
        num_steps=config.num_steps,
        maximum_time=config.maximum_time,
        maximum_lambda=config.maximum_lambda,
        initial_raw_T=config.initial_raw_T,
        initial_raw_lambda=config.initial_raw_lambda,
        mask_state_each_step=config.mask_state_each_step,
        checkpoint_force=config.checkpoint_force,
    ).to(device)
    print_model_architecture(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    weight = magnitude_weight(sample.magnitude)
    history: list[dict[str, float]] = []
    last_initial: torch.Tensor | None = None
    last_local_field: torch.Tensor | None = None

    model.train()
    for epoch in range(config.epochs):
        # Different deterministic seeds guarantee a distinct random draw each epoch.
        local_field = simulate_noisy_local_field(
            sample.susceptibility,
            sample.magnitude,
            sample.brain_mask,
            operator,
            snr=config.snr,
            phase_scale=config.phase_scale,
            seed=config.seed + epoch,
        )
        initial = initial_backprojection(local_field, weight, sample.brain_mask, operator)
        optimizer.zero_grad(set_to_none=True)
        output = model(local_field, sample.brain_mask, None, weight, initial)
        x_pred = mask_and_reference(
            output.susceptibility,
            sample.brain_mask,
            convention=config.reference_convention,
        )
        x_true = mask_and_reference(
            sample.susceptibility,
            sample.brain_mask,
            convention=config.reference_convention,
        )
        loss_nrmse = nrmse(x_pred, x_true)
        loss = loss_nrmse
        if config.data_consistency_weight > 0.0:
            data_consistency_value = weighted_data_consistency_loss(
                output.susceptibility,
                local_field,
                weight,
                operator,
                field_mask=sample.brain_mask,
            )
        else:
            with torch.no_grad():
                data_consistency_value = weighted_data_consistency_loss(
                    output.susceptibility.detach(),
                    local_field,
                    weight,
                    operator,
                    field_mask=sample.brain_mask,
                )
        with torch.no_grad():
            regularization_energy_per_sample = model.regularizer.energy(
                output.susceptibility.detach()
            )
            regularization_voxel_count = torch.full(
                (output.susceptibility.shape[0],),
                output.susceptibility[0, 0].numel(),
                dtype=torch.float32,
                device=device,
            )
            regularization_energy = (
                regularization_energy_per_sample / regularization_voxel_count
            ).mean()
            regularization_energy_total = regularization_energy_per_sample.mean()
        if config.data_consistency_weight > 0.0:
            loss = loss + config.data_consistency_weight * data_consistency_value
        _finite_or_raise("TDV loss", loss)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"Non-finite gradient: {name}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_gradient_norm)
        scaler.step(optimizer)
        scaler.update()
        model.regularizer.project_analysis_kernel_()

        record = {
            "epoch": float(epoch + 1),
            "loss": float(loss.detach().cpu()),
            "nrmse": float(loss_nrmse.detach().cpu()),
            "data_consistency_value": float(data_consistency_value.detach().cpu()),
            "regularization_energy": float(regularization_energy.detach().cpu()),
            "regularization_energy_total": float(regularization_energy_total.detach().cpu()),
            "time": float(output.time.detach().cpu()),
            "lambda": float(output.data_coefficient.detach().cpu()),
            "regularizer_step": float(output.regularizer_step.detach().cpu()),
            "data_step": float(output.data_step.detach().cpu()),
            "step_size": float(output.regularizer_step.detach().cpu()),
            "data_gradient_norm": float(output.data_gradient_norm.mean().cpu()),
            "regularizer_gradient_norm": float(output.regularizer_gradient_norm.mean().cpu()),
            "state_norm": float(output.state_norm.mean().cpu()),
            "parameter_gradient_norm": float(gradient_norm.detach().cpu()),
        }
        history.append(record)
        last_initial, last_local_field = initial.detach(), local_field.detach()
        print(
            f"epoch {epoch + 1:04d}/{config.epochs}: "
            f"loss={record['loss']:.6e} nrmse={record['nrmse']:.6e} "
            f"T={record['time']:.3e} lambda={record['lambda']:.3e} "
            f"tau_R={record['regularizer_step']:.3e} tau_D={record['data_step']:.3e}",
            flush=True,
        )

    assert last_initial is not None and last_local_field is not None
    model.eval()
    # The production force is an explicit transpose chain, so evaluation no
    # longer needs to build an input-autograd graph.
    with torch.no_grad():
        final_output = model(last_local_field, sample.brain_mask, None, weight, last_initial)
        final_prediction = mask_and_reference(
            final_output.susceptibility,
            sample.brain_mask,
            convention=config.reference_convention,
        )
        final_truth = mask_and_reference(
            sample.susceptibility,
            sample.brain_mask,
            convention=config.reference_convention,
        )
        final_nrmse = nrmse(final_prediction, final_truth)
    _finite_or_raise("final NRMSE", final_nrmse)
    _save_history(history, output_dir)
    save_reconstruction_figure(
        last_initial,
        final_prediction,
        final_truth,
        output_dir / "reconstruction.png",
        nrmse_value=float(final_nrmse.detach().cpu()),
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "final_nrmse": float(final_nrmse.detach().cpu()),
            "weight_rule": "W = sqrt(2) * magn",
            "regularizer_energy_rule": "R_theta(chi) = sum(T_theta(chi) / num_features)",
        },
        output_dir / "checkpoint.pt",
    )
    report = {
        "training_kind": "single-volume overfit diagnostic",
        "physical_model": "b = F^H D F chi + eta; periodic unitary FFT, no padding/cropping/TKD",
        "volume_layout": "[B, 1, Z, Y, X]",
        "metadata_order": "zyx",
        "weight_rule": "Stored W (not sqrt(W)) = sqrt(2) * dimensionless magn; no normalization, clipping, or mask multiplication; zero magn gives W=0",
        "regularizer_energy_rule": "R_theta(chi) = sum(T_theta(chi) / num_features)",
        "history_energy_normalization": "data_consistency_value = ||W(Achi-b)||^2/N_mask; regularization_energy = R_theta(chi)/N_volume",
        "source_provenance": "TDV energy/manual-force design follows VLOGroup/tdv; 3-D, QSM, magnitude W, NRMSE, AMP, and medical-volume handling are project extensions",
        "susceptibility_reference_convention": config.reference_convention,
        "noise_rule": "Per epoch complex Gaussian signal noise; real/imag std = mean(magn inside brain_mask) / SNR",
        "image_display_range": [-0.1, 0.1],
        "final_nrmse": float(final_nrmse.detach().cpu()),
        "taus": {
            "regularizer": float(final_output.regularizer_step.detach().cpu()),
            "data": float(final_output.data_step.detach().cpu()),
        },
        "config": asdict(config),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return model, history


def _configuration_from_args(arguments: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        epochs=arguments.epochs,
        snr=arguments.snr,
        seed=arguments.seed,
        learning_rate=arguments.learning_rate,
        max_gradient_norm=arguments.max_gradient_norm,
        data_consistency_weight=arguments.data_consistency_weight,
        features=arguments.features,
        macro_blocks=arguments.macro_blocks,
        num_steps=arguments.steps,
        maximum_time=arguments.maximum_time,
        maximum_lambda=arguments.maximum_lambda,
        initial_raw_T=arguments.initial_raw_T,
        initial_raw_lambda=arguments.initial_raw_lambda,
        mask_state_each_step=not arguments.no_step_mask,
        checkpoint_force=arguments.checkpoint_force,
        use_amp=not arguments.no_amp,
        reference_convention=arguments.reference_convention,
        regularizer_head_initialization_scale=arguments.regularizer_head_initialization_scale,
        phase_scale=arguments.phase_scale,
        voxel_size_zyx=tuple(arguments.voxel_size),
        b0_direction_zyx=tuple(arguments.b0_direction),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PyTorch explicit TDV-QSM on one COSMOS volume.")
    parser.add_argument("--data", type=Path, default=Path("/cosmos_data"), help="COSMOS directory, MAT, or NPZ file")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/cosmos-tdv"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--snr", type=float, default=70.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--learning-rate",
        type=float,
        required=True,
        help="Adam learning rate; required so every run records an explicit stability choice.",
    )
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--data-consistency-weight", type=float, default=0.0)
    parser.add_argument("--features", type=int, default=1)
    parser.add_argument("--macro-blocks", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--maximum-time", type=float, default=0.25)
    parser.add_argument("--maximum-lambda", type=float, default=2.0)
    parser.add_argument("--initial-raw-T", type=float, default=2.0)
    parser.add_argument("--initial-raw-lambda", type=float, default=2.0)
    parser.add_argument("--no-step-mask", action="store_true")
    parser.add_argument("--checkpoint-force", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--reference-convention",
        choices=("already_referenced", "masked_mean_zero"),
        default="already_referenced",
    )
    parser.add_argument(
        "--regularizer-head-initialization-scale",
        type=float,
        default=0.2,
        help="Uniform energy-head initialization scale before division by sqrt(features).",
    )
    parser.add_argument("--phase-scale", type=float, default=1.0)
    parser.add_argument("--voxel-size", type=float, nargs=3, default=(1.0, 1.0, 1.0), metavar=("VZ", "VY", "VX"))
    parser.add_argument("--b0-direction", type=float, nargs=3, default=(0.0, 0.0, 1.0), metavar=("BZ", "BY", "BX"))
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available, else CPU.")
    arguments = parser.parse_args()
    device = torch.device(arguments.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    sample = load_single_volume(arguments.data, device)
    config = _configuration_from_args(arguments)
    _, history = train_single_volume(sample, config, arguments.output_dir)
    print(f"Final training NRMSE: {history[-1]['nrmse']:.6e}")
    print(f"Wrote {arguments.output_dir / 'reconstruction.png'}")


if __name__ == "__main__":
    main()
