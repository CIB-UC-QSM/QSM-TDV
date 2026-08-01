"""Strict on-disk contract for a single anonymised QSM training sample."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from qsm_tdv.physics.dipole import DipoleMetadata


REQUIRED_MANIFEST_FIELDS = {
    "anonymized_subject_id",
    "source",
    "processing_version",
    "voxel_size_zyx",
    "b0_direction_zyx",
    "field_units",
    "susceptibility_units",
    "susceptibility_reference",
}


@dataclass(frozen=True)
class QSMSample:
    """One sample with unbatched spatial arrays, all in ``[Z,Y,X,1]`` order."""

    local_field: np.ndarray
    susceptibility: np.ndarray
    chi_init: np.ndarray
    brain_mask: np.ndarray
    metadata: DipoleMetadata
    manifest: dict[str, Any]
    reference_mask: np.ndarray | None = None
    statistical_weight: np.ndarray | None = None

    def as_batch(self) -> dict[str, jnp.ndarray | None]:
        """Convert the contract's unbatched arrays to float32 NDHWC tensors."""

        def batch(array: np.ndarray | None) -> jnp.ndarray | None:
            return None if array is None else jnp.asarray(array[None, ...], dtype=jnp.float32)

        return {
            "local_field": batch(self.local_field),
            "susceptibility": batch(self.susceptibility),
            # chi_0 is fixed by the reconstruction protocol.  Do not honour
            # an adjoint initialisation stored in older .npz files.
            "chi_init": batch(np.zeros_like(self.local_field)),
            "brain_mask": batch(self.brain_mask),
            "reference_mask": batch(self.reference_mask),
            "statistical_weight": batch(self.statistical_weight),
        }


def _validate_volume(name: str, value: np.ndarray, expected_spatial: tuple[int, int, int] | None = None) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 4 or value.shape[-1] != 1:
        raise ValueError(f"{name} must have shape [Z,Y,X,1], got {value.shape}")
    if expected_spatial is not None and tuple(value.shape[:3]) != expected_spatial:
        raise ValueError(f"{name} spatial shape {value.shape[:3]} does not match {expected_spatial}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def load_single_sample(dataset_path: str | Path) -> QSMSample:
    """Load ``.npz`` plus its adjacent JSON manifest and validate all metadata.

    Required NPZ arrays are ``local_field``, ``susceptibility``, ``chi_init``,
    and ``brain_mask``.  Optional arrays are ``reference_mask`` and
    ``statistical_weight``.  The manifest shares the NPZ stem, e.g.
    ``sample.npz`` and ``sample.json``.
    """

    path = Path(dataset_path)
    manifest_path = path.with_suffix(".json")
    if path.suffix != ".npz":
        raise ValueError("A single-sample dataset must be an .npz file")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing required manifest: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    missing_metadata = REQUIRED_MANIFEST_FIELDS - manifest.keys()
    if missing_metadata:
        raise ValueError(f"Manifest missing required fields: {sorted(missing_metadata)}")

    metadata = DipoleMetadata(
        voxel_size_zyx=tuple(float(v) for v in manifest["voxel_size_zyx"]),
        b0_direction_zyx=tuple(float(v) for v in manifest["b0_direction_zyx"]),
        field_units=str(manifest["field_units"]),
        susceptibility_units=str(manifest["susceptibility_units"]),
        susceptibility_reference=str(manifest["susceptibility_reference"]),
    )
    with np.load(path, allow_pickle=False) as archive:
        missing_arrays = {"local_field", "susceptibility", "chi_init", "brain_mask"} - set(archive.files)
        if missing_arrays:
            raise ValueError(f"Dataset missing required arrays: {sorted(missing_arrays)}")
        local_field = _validate_volume("local_field", archive["local_field"])
        spatial_shape = tuple(local_field.shape[:3])
        susceptibility = _validate_volume("susceptibility", archive["susceptibility"], spatial_shape)
        chi_init = _validate_volume("chi_init", archive["chi_init"], spatial_shape)
        brain_mask = _validate_volume("brain_mask", archive["brain_mask"], spatial_shape)
        reference_mask = (
            _validate_volume("reference_mask", archive["reference_mask"], spatial_shape)
            if "reference_mask" in archive.files
            else None
        )
        statistical_weight = (
            _validate_volume("statistical_weight", archive["statistical_weight"], spatial_shape)
            if "statistical_weight" in archive.files
            else None
        )
    if np.any(brain_mask < 0):
        raise ValueError("brain_mask must be non-negative")
    if statistical_weight is not None and np.any(statistical_weight < 0):
        raise ValueError("statistical_weight must be non-negative")
    return QSMSample(
        local_field=local_field,
        susceptibility=susceptibility,
        chi_init=chi_init,
        brain_mask=brain_mask,
        metadata=metadata,
        manifest=manifest,
        reference_mask=reference_mask,
        statistical_weight=statistical_weight,
    )
