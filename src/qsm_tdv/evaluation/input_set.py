"""Evaluate a saved TDV-QSM checkpoint on a COSMOS-like MAT-file directory.

``phase_in.mat`` is interpreted as the already prepared real local field
``b`` in the field units expected by the checkpoint.  It is deliberately not
phase-unwrapped, rescaled, or otherwise transformed by this evaluator.  When
that file is absent, a supplied susceptibility reference is forward simulated
with the documented dipole model to create ``b = A chi``.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import jax
import numpy as np
from scipy.io import loadmat

from qsm_tdv.data.contract import effective_data_weight
from qsm_tdv.evaluation.slices import save_orthogonal_evaluation_slices
from qsm_tdv.models.tdv import TDVConfig
from qsm_tdv.physics.dipole import apply_dipole, dipole_kernel
from qsm_tdv.physics.reconstruction import ReconstructionConfig, reconstruct_trajectory


@dataclass(frozen=True)
class EvaluationInput:
    """Validated unbatched arrays and their explicitly recorded input source."""

    local_field: np.ndarray
    brain_mask: np.ndarray
    susceptibility: np.ndarray | None
    magnitude: np.ndarray | None
    input_mode: str
    source_files: dict[str, str]


def _load_mat_volume(path: Path, variable_names: Iterable[str]) -> tuple[np.ndarray, str]:
    """Load a real finite 3D volume under one of the documented variable names."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing required MAT file: {path}")
    values = loadmat(path)
    for variable_name in variable_names:
        if variable_name in values:
            array = np.asarray(values[variable_name])
            break
    else:
        available = sorted(name for name in values if not name.startswith("__"))
        expected = ", ".join(variable_names)
        raise ValueError(f"{path} must contain one of [{expected}]; found {available}")
    if array.ndim != 3:
        raise ValueError(f"{path}:{variable_name} must be a 3D array, got {array.shape}")
    if np.iscomplexobj(array):
        if not np.allclose(array.imag, 0.0):
            raise ValueError(f"{path}:{variable_name} must be real-valued")
        array = array.real
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{path}:{variable_name} contains non-finite values")
    return array, variable_name


def _optional_chi(data_dir: Path) -> tuple[np.ndarray | None, Path | None, str | None]:
    """Load either supported COSMOS-like susceptibility naming convention."""

    candidates = (
        (data_dir / "chi_cosmos.mat", ("chi_cosmos",)),
        (data_dir / "chi.mat", ("chi",)),
    )
    for path, variables in candidates:
        if path.is_file():
            volume, variable_name = _load_mat_volume(path, variables)
            return volume, path, variable_name
    return None, None, None


def _optional_magnitude(data_dir: Path) -> tuple[np.ndarray | None, Path | None, str | None]:
    """Load an optional real magnitude map using COSMOS-compatible names."""

    candidates = (
        (data_dir / "magn.mat", ("magn", "magnitude")),
        (data_dir / "magnitude.mat", ("magnitude", "magn")),
    )
    for path, variables in candidates:
        if path.is_file():
            volume, variable_name = _load_mat_volume(path, variables)
            if np.any(volume < 0):
                raise ValueError(f"{path}:{variable_name} must be non-negative")
            return volume, path, variable_name
    return None, None, None


def load_evaluation_input(
    data_dir: str | Path,
    *,
    voxel_size_zyx: tuple[float, float, float],
    b0_direction_zyx: tuple[float, float, float],
) -> EvaluationInput:
    """Load a COSMOS-like directory and construct its explicit QSM input.

    Required files are ``msk.mat`` (variable ``msk``) and either
    ``phase_in.mat`` (variable ``phase_in``) or a reference ``chi_cosmos.mat``
    (variable ``chi_cosmos``) / ``chi.mat`` (variable ``chi``).  When both
    phase and susceptibility are present, the phase input drives the solver
    and the susceptibility is retained only for per-step evaluation.
    """

    directory = Path(data_dir).expanduser().resolve()
    mask, mask_variable = _load_mat_volume(directory / "msk.mat", ("msk", "brain_mask"))
    brain_mask = (mask > 0).astype(np.float32)
    if not np.any(brain_mask):
        raise ValueError("msk.mat has no positive support voxels")

    susceptibility, chi_path, chi_variable = _optional_chi(directory)
    magnitude, magnitude_path, magnitude_variable = _optional_magnitude(directory)
    phase_path = directory / "phase_in.mat"
    source_files = {"brain_mask": f"{directory / 'msk.mat'}:{mask_variable}"}
    if phase_path.is_file():
        local_field, phase_variable = _load_mat_volume(phase_path, ("phase_in", "local_field"))
        input_mode = "phase_in"
        source_files["local_field"] = f"{phase_path}:{phase_variable}"
    else:
        if susceptibility is None:
            raise FileNotFoundError(
                f"{directory} has neither phase_in.mat nor chi_cosmos.mat/chi.mat; cannot form a QSM input"
            )
        kernel = dipole_kernel(susceptibility.shape, voxel_size_zyx, b0_direction_zyx)
        local_field = np.asarray(
            apply_dipole(susceptibility[None, ..., None], kernel), dtype=np.float32
        )[0, ..., 0]
        input_mode = "simulated_from_chi"
        source_files["local_field"] = "simulated as A(chi) with the documented periodic unitary dipole model"
    if susceptibility is not None:
        source_files["susceptibility"] = f"{chi_path}:{chi_variable}"
    if magnitude is not None:
        source_files["magnitude"] = f"{magnitude_path}:{magnitude_variable}"

    expected_shape = brain_mask.shape
    if local_field.shape != expected_shape:
        raise ValueError(
            f"Input local field shape {local_field.shape} does not match mask shape {expected_shape}"
        )
    if susceptibility is not None and susceptibility.shape != expected_shape:
        raise ValueError(f"Susceptibility shape {susceptibility.shape} does not match mask shape {expected_shape}")
    if magnitude is not None and magnitude.shape != expected_shape:
        raise ValueError(f"Magnitude shape {magnitude.shape} does not match mask shape {expected_shape}")
    # The observation support remains explicit in the solver.  Zeroing b
    # outside it keeps the stored input consistent with that support.
    local_field = local_field * brain_mask
    return EvaluationInput(
        local_field=local_field,
        brain_mask=brain_mask,
        susceptibility=susceptibility,
        magnitude=magnitude,
        input_mode=input_mode,
        source_files=source_files,
    )


def _nrmse(estimate: np.ndarray, ground_truth: np.ndarray, mask: np.ndarray) -> float:
    """Masked ``||estimate - ground_truth|| / ||ground_truth||`` on host."""

    difference = (estimate - ground_truth) * mask
    target = ground_truth * mask
    return float(np.linalg.norm(difference.reshape(-1)) / max(float(np.linalg.norm(target.reshape(-1))), 1e-12))


def _rmse(estimate: np.ndarray, ground_truth: np.ndarray, mask: np.ndarray) -> float:
    """Masked root-mean-square error on the explicit brain support."""

    support = mask > 0
    if not np.any(support):
        raise ValueError("Cannot compute RMSE with an empty brain mask")
    return float(np.sqrt(np.mean((estimate[support] - ground_truth[support]) ** 2)))


def _tol_update(current: np.ndarray, previous: np.ndarray) -> float:
    """Return ``||x_now - x_prev||_2 / ||x_prev||_2`` for successive states."""

    return float(np.linalg.norm((current - previous).reshape(-1)) / max(np.linalg.norm(previous.reshape(-1)), 1e-12))


def _read_checkpoint(path: Path) -> dict[str, Any]:
    """Read a local training checkpoint after validating required members."""

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # Pickle is intentionally restricted to a checkpoint produced by this local repository.
    with path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    required = {"parameters", "tdv_config", "reconstruction_config", "metadata"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint misses required entries: {sorted(missing)}")
    if "tdv" not in checkpoint["parameters"] or "raw_time" not in checkpoint["parameters"]:
        raise ValueError("Checkpoint parameters must contain tdv and raw_time")
    return checkpoint


def _metadata_vector(
    checkpoint_metadata: dict[str, Any],
    name: str,
    override: list[float] | None,
) -> tuple[float, float, float]:
    if override is not None:
        value = tuple(float(item) for item in override)
    elif name in checkpoint_metadata:
        value = tuple(float(item) for item in checkpoint_metadata[name])
    else:
        raise ValueError(f"Checkpoint does not contain {name}; provide it explicitly on the command line")
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values in z,y,x order")
    return value  # type: ignore[return-value]


def _write_rows(path: Path, rows: list[dict[str, float | int | None]]) -> None:
    fields = (
        "iteration",
        "rmse_to_gt",
        "nrmse_to_gt",
        "tol_update",
        "tol_update_nrmse",
        "cg_relative_residual",
        "tdv_energy",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    steps: int | None = None,
    cg_iterations: int | None = None,
    voxel_size_zyx: tuple[float, float, float] | None = None,
    b0_direction_zyx: tuple[float, float, float] | None = None,
    include_magnitude_in_weight: bool = True,
) -> dict[str, Any]:
    """Run one trusted checkpoint and save a prediction, metrics, and slices."""

    checkpoint = _read_checkpoint(Path(checkpoint_path))
    checkpoint_metadata = checkpoint["metadata"]
    resolved_voxel_size = voxel_size_zyx or _metadata_vector(checkpoint_metadata, "voxel_size_zyx", None)
    resolved_b0_direction = b0_direction_zyx or _metadata_vector(checkpoint_metadata, "b0_direction_zyx", None)
    sample = load_evaluation_input(
        data_dir,
        voxel_size_zyx=resolved_voxel_size,
        b0_direction_zyx=resolved_b0_direction,
    )
    tdv_config = TDVConfig(**checkpoint["tdv_config"])
    saved_reconstruction_config = ReconstructionConfig(**checkpoint["reconstruction_config"])
    reconstruction_config = replace(
        saved_reconstruction_config,
        steps=saved_reconstruction_config.steps if steps is None else steps,
        cg_iterations=saved_reconstruction_config.cg_iterations if cg_iterations is None else cg_iterations,
    )
    kernel = dipole_kernel(sample.local_field.shape, resolved_voxel_size, resolved_b0_direction)
    local_field = sample.local_field[None, ..., None]
    brain_mask = sample.brain_mask[None, ..., None]
    magnitude = None if sample.magnitude is None else sample.magnitude[None, ..., None]
    data_weight = effective_data_weight(
        brain_mask,
        magnitude,
        include_magnitude=include_magnitude_in_weight,
    )
    # Match training: chi_0 = W * phase_in (or W * b for simulated input).
    chi_init = data_weight * local_field
    parameters = jax.device_put(checkpoint["parameters"])
    trajectory, diagnostics = reconstruct_trajectory(
        parameters["tdv"],
        parameters["raw_time"],
        chi_init,
        local_field,
        kernel,
        tdv_config,
        reconstruction_config,
        regularizer_mask=brain_mask,
        statistical_weight=data_weight,
    )
    trajectory_np = np.asarray(jax.device_get(trajectory), dtype=np.float32)
    residuals = np.asarray(jax.device_get(diagnostics.relative_residuals), dtype=np.float32)
    energies = np.asarray(jax.device_get(diagnostics.energies), dtype=np.float32)

    rows: list[dict[str, float | int | None]] = []
    for iteration, state in enumerate(trajectory_np):
        row: dict[str, float | int | None] = {
            "iteration": iteration,
            "tol_update": None,
            "rmse_to_gt": None,
            "nrmse_to_gt": None,
            "tol_update_nrmse": None,
            "cg_relative_residual": None,
            "tdv_energy": None,
        }
        if sample.susceptibility is not None:
            row["rmse_to_gt"] = _rmse(state[0, ..., 0], sample.susceptibility, sample.brain_mask)
            row["nrmse_to_gt"] = _nrmse(state[0, ..., 0], sample.susceptibility, sample.brain_mask)
        if iteration > 0:
            row["tol_update"] = _tol_update(state[0, ..., 0], trajectory_np[iteration - 1, 0, ..., 0])
            row["tol_update_nrmse"] = _nrmse(
                state[0, ..., 0], trajectory_np[iteration - 1, 0, ..., 0], sample.brain_mask
            )
            row["cg_relative_residual"] = float(np.max(residuals[iteration - 1]))
            row["tdv_energy"] = float(np.mean(energies[iteration - 1]))
        rows.append(row)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "input_local_field.npy", sample.local_field)
    np.save(destination / "chi_initial.npy", trajectory_np[0, 0])
    np.save(destination / "prediction.npy", trajectory_np[-1, 0])
    _write_rows(destination / "iteration_metrics.csv", rows)
    final_nrmse = rows[-1]["nrmse_to_gt"]
    figure = save_orthogonal_evaluation_slices(
        trajectory_np[0],
        trajectory_np[-1],
        brain_mask,
        destination / "orthogonal_slices.png",
        chi_ground_truth=None if sample.susceptibility is None else sample.susceptibility[..., None],
        nrmse_value=None if final_nrmse is None else float(final_nrmse),
    )
    max_residual = float(np.max(residuals))
    report = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "input_directory": str(Path(data_dir).resolve()),
        "input_mode": sample.input_mode,
        "source_files": sample.source_files,
        "data_weight": "mask*magnitude" if include_magnitude_in_weight and magnitude is not None else "mask",
        "has_ground_truth": sample.susceptibility is not None,
        "voxel_size_zyx": list(resolved_voxel_size),
        "b0_direction_zyx": list(resolved_b0_direction),
        "field_units": checkpoint_metadata.get("field_units", "not recorded in checkpoint"),
        "susceptibility_units": checkpoint_metadata.get("susceptibility_units", "not recorded in checkpoint"),
        "susceptibility_reference": checkpoint_metadata.get("susceptibility_reference", "not recorded in checkpoint"),
        "tdv_config": asdict(tdv_config),
        "reconstruction_config": asdict(reconstruction_config),
        "stopping_time": float(diagnostics.time),
        "tau": float(diagnostics.tau),
        "max_cg_relative_residual": max_residual,
        "cg_relative_tolerance": reconstruction_config.cg_relative_tolerance,
        "cg_tolerance_met": max_residual <= reconstruction_config.cg_relative_tolerance,
        "rmse_definition": "masked sqrt(mean((chi_s - chi_gt)^2)) over brain_mask > 0",
        "tol_update_definition": "||x_now - x_prev||_2 / ||x_prev||_2 over the full reconstruction state",
        "final_rmse_to_gt": rows[-1]["rmse_to_gt"],
        "final_nrmse_to_gt": final_nrmse,
        "final_tol_update": rows[-1]["tol_update"],
        "final_tol_update_nrmse": rows[-1]["tol_update_nrmse"],
        "iteration_metrics": str(destination / "iteration_metrics.csv"),
        "slice_figure": str(figure),
    }
    (destination / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trusted TDV-QSM checkpoint on a COSMOS-like MAT-file input directory."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing msk.mat and phase_in.mat or chi*.mat")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trusted local checkpoint.pkl produced by this repository")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/evaluation"))
    parser.add_argument(
        "--iterations",
        "--steps",
        dest="steps",
        type=int,
        default=None,
        help="Number of semi-implicit TDV-QSM steps S; overrides the checkpoint setting",
    )
    parser.add_argument(
        "--cg-iterations",
        type=int,
        default=None,
        help="Fixed conjugate-gradient iterations per TDV-QSM step; overrides the checkpoint setting",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        nargs=3,
        default=None,
        metavar=("VZ", "VY", "VX"),
        help="Voxel size in z,y,x order; defaults to the checkpoint metadata",
    )
    parser.add_argument(
        "--b0-direction",
        type=float,
        nargs=3,
        default=None,
        metavar=("BZ", "BY", "BX"),
        help="B0 direction in z,y,x order; defaults to the checkpoint metadata",
    )
    parser.add_argument(
        "--include-magnitude-in-weight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use W = mask * magnitude when magn.mat or magnitude.mat is present (default: enabled)",
    )
    arguments = parser.parse_args()
    report = evaluate_checkpoint(
        arguments.checkpoint,
        arguments.input_dir,
        arguments.output_dir,
        steps=arguments.steps,
        cg_iterations=arguments.cg_iterations,
        voxel_size_zyx=None if arguments.voxel_size is None else tuple(arguments.voxel_size),
        b0_direction_zyx=None if arguments.b0_direction is None else tuple(arguments.b0_direction),
        include_magnitude_in_weight=arguments.include_magnitude_in_weight,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
