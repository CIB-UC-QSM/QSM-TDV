"""Standalone deterministic validation of a saved TDV-QSM checkpoint."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from qsm_tdv.data.contract import load_single_sample
from qsm_tdv.models.tdv import TDVConfig
from qsm_tdv.physics.dipole import dipole_kernel
from qsm_tdv.physics.reconstruction import (
    ReconstructionConfig,
    data_fidelity,
    reconstruct,
    require_cg_residuals_within_tolerance,
)
from qsm_tdv.training.metrics import masked_mse, nrmse


def _assert_metadata_match(saved: dict[str, Any], current: Any) -> None:
    expected = {
        "voxel_size_zyx": tuple(saved["voxel_size_zyx"]),
        "b0_direction_zyx": tuple(saved["b0_direction_zyx"]),
        "field_units": saved["field_units"],
        "susceptibility_units": saved["susceptibility_units"],
        "susceptibility_reference": saved["susceptibility_reference"],
    }
    observed = {
        "voxel_size_zyx": tuple(current.voxel_size_zyx),
        "b0_direction_zyx": tuple(current.b0_direction_zyx),
        "field_units": current.field_units,
        "susceptibility_units": current.susceptibility_units,
        "susceptibility_reference": current.susceptibility_reference,
    }
    if expected != observed:
        raise ValueError("Dataset physics/units/reference metadata does not match the checkpoint")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a trusted local TDV-QSM checkpoint on one explicit dataset.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trusted local checkpoint.pkl from train_single_dataset.py")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/validation"))
    arguments = parser.parse_args()

    sample = load_single_sample(arguments.dataset)
    # Pickle is deliberately limited to checkpoints produced locally by this repository.
    with arguments.checkpoint.open("rb") as handle:
        checkpoint = pickle.load(handle)
    _assert_metadata_match(checkpoint["metadata"], sample.metadata)
    tdv_config = TDVConfig(**checkpoint["tdv_config"])
    reconstruction_config = ReconstructionConfig(**checkpoint["reconstruction_config"])
    parameters = jax.device_put(checkpoint["parameters"])
    batch = sample.as_batch()
    local_field = batch["local_field"]
    reference = batch["susceptibility"]
    # Evaluation matches training: chi_0 is fixed to the zero volume rather
    # than an adjoint-field initialisation stored in a legacy sample.
    chi_init = jnp.zeros_like(local_field) if local_field is not None else None
    brain_mask = batch["brain_mask"]
    reference_mask = batch["reference_mask"] if batch["reference_mask"] is not None else brain_mask
    if local_field is None or reference is None or chi_init is None or brain_mask is None or reference_mask is None:
        raise RuntimeError("Validated sample has a missing required tensor")
    kernel = dipole_kernel(
        tuple(local_field.shape[1:4]), sample.metadata.voxel_size_zyx, sample.metadata.b0_direction_zyx
    )
    reconstruction, diagnostics = reconstruct(
        parameters["tdv"],
        parameters["raw_time"],
        chi_init,
        local_field,
        kernel,
        tdv_config,
        reconstruction_config,
        regularizer_mask=brain_mask,
        observation_mask=brain_mask,
        statistical_weight=batch["statistical_weight"],
    )
    require_cg_residuals_within_tolerance(diagnostics, reconstruction_config)
    mse = masked_mse(reconstruction, reference, reference_mask)
    normalized_error = nrmse(reconstruction, reference, reference_mask)
    fidelity = jnp.mean(
        data_fidelity(
            reconstruction,
            local_field,
            kernel,
            observation_mask=brain_mask,
            statistical_weight=batch["statistical_weight"],
        )
    ) / jnp.maximum(jnp.sum(brain_mask), 1.0)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(arguments.output_dir / "reconstruction.npy", np.asarray(reconstruction[0]))
    report = {
        "mse": float(mse),
        "nrmse": float(normalized_error),
        "data_fidelity": float(fidelity),
        "stopping_time": float(diagnostics.time),
        "max_cg_relative_residual": float(jnp.max(diagnostics.relative_residuals)),
        "dataset_manifest": sample.manifest,
    }
    (arguments.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
