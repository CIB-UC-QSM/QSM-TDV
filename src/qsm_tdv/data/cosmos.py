"""COSMOS MAT-file adapter and noisy complex-signal simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.io import loadmat

from qsm_tdv.physics.dipole import apply_dipole, dipole_kernel


@dataclass(frozen=True)
class CosmosSimulationConfig:
    """Explicit assumptions for constructing one noisy COSMOS QSM sample."""

    snr: float = 70.0
    seed: int = 0
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0)
    b0_direction_zyx: tuple[float, float, float] = (0.0, 0.0, 1.0)
    phase_scale_radians_per_susceptibility_unit: float = 1.0

    def __post_init__(self) -> None:
        if self.snr <= 0:
            raise ValueError("snr must be positive")
        if self.phase_scale_radians_per_susceptibility_unit <= 0:
            raise ValueError("phase scale must be positive")
        if len(self.voxel_size_zyx) != 3 or any(value <= 0 for value in self.voxel_size_zyx):
            raise ValueError("voxel_size_zyx must contain three positive values")
        if len(self.b0_direction_zyx) != 3 or not any(value != 0 for value in self.b0_direction_zyx):
            raise ValueError("b0_direction_zyx must be a non-zero 3-vector")


def resolve_cosmos_directory(requested: str | Path) -> Path:
    """Resolve `/cosmos_data`, including the workspace-mounted fallback.

    Some execution environments expose the supplied directory at
    ``<workspace>/cosmos_data`` instead of the absolute mount.  The resolved
    path is reported in the manifest, so this fallback is never hidden.
    """

    candidate = Path(requested)
    workspace_fallback = Path.cwd() / "cosmos_data"
    for path in (candidate, workspace_fallback):
        if (path / "chi_cosmos.mat").is_file() and (path / "magn.mat").is_file() and (path / "msk.mat").is_file():
            return path.resolve()
    raise FileNotFoundError(
        "Expected chi_cosmos.mat, magn.mat, and msk.mat in "
        f"{candidate}; also checked workspace fallback {workspace_fallback}."
    )


def _load_volume(path: Path, key: str, *, dtype: np.dtype) -> np.ndarray:
    values = loadmat(path)
    if key not in values:
        available = sorted(name for name in values if not name.startswith("__"))
        raise ValueError(f"{path} does not contain '{key}'; found {available}")
    array = np.asarray(values[key], dtype=dtype)
    if array.ndim != 3:
        raise ValueError(f"{path}:{key} must be a 3D array, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path}:{key} contains non-finite values")
    return array


def load_cosmos_volumes(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    """Load chi, magnitude, and mask in their stored `[Z,Y,X]` ordering."""

    resolved = resolve_cosmos_directory(data_dir)
    chi = _load_volume(resolved / "chi_cosmos.mat", "chi_cosmos", dtype=np.float32)
    magnitude = _load_volume(resolved / "magn.mat", "magn", dtype=np.float32)
    mask = _load_volume(resolved / "msk.mat", "msk", dtype=np.float32)
    if chi.shape != magnitude.shape or chi.shape != mask.shape:
        raise ValueError(f"COSMOS arrays must have identical shapes, got {chi.shape}, {magnitude.shape}, {mask.shape}")
    if np.any(magnitude < 0) or np.any(mask < 0):
        raise ValueError("COSMOS magnitude and mask must be non-negative")
    return chi, magnitude, (mask > 0).astype(np.float32), resolved


def _configuration_hash(configuration: dict[str, object]) -> str:
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prepare_cosmos_overfit_sample(
    data_dir: str | Path,
    output_path: str | Path,
    config: CosmosSimulationConfig = CosmosSimulationConfig(),
) -> tuple[Path, Path]:
    """Write a metadata-complete QSM sample simulated from COSMOS chi.

    The construction is exactly:

    ``b = A(chi_cosmos)``, ``phase = phase_scale * b``,
    ``S = magnitude * exp(-1j * phase)``, and ``S_noisy = S + n``.

    ``n`` is a spatially independent circular complex Gaussian field. It is
    scaled so its *realised in-mask* L2 SNR, ``||S|| / ||n||``, is exactly the
    requested SNR. The noisy field used for inversion is
    ``-angle(S_noisy) / phase_scale``. No phase unwrapping is applied; the
    default scale is documented and should be changed explicitly for a
    protocol-specific phase model.
    """

    chi, magnitude, mask, resolved_data_dir = load_cosmos_volumes(data_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix != ".npz":
        output = output.with_suffix(".npz")

    kernel = dipole_kernel(chi.shape, config.voxel_size_zyx, config.b0_direction_zyx)
    chi_batch = jnp.asarray(chi[None, ..., None], dtype=jnp.float32)
    clean_field = np.asarray(apply_dipole(chi_batch, kernel))[0, ..., 0]
    phase = config.phase_scale_radians_per_susceptibility_unit * clean_field
    clean_signal = magnitude.astype(np.complex64) * np.exp(-1j * phase.astype(np.float32))

    rng = np.random.default_rng(config.seed)
    complex_noise = (
        rng.normal(size=chi.shape).astype(np.float32)
        + 1j * rng.normal(size=chi.shape).astype(np.float32)
    ).astype(np.complex64)
    signal_norm = np.linalg.norm((clean_signal * mask).reshape(-1))
    noise_norm = np.linalg.norm((complex_noise * mask).reshape(-1))
    if signal_norm == 0.0 or noise_norm == 0.0:
        raise ValueError("COSMOS in-mask signal and sampled noise must have non-zero L2 norm")
    complex_noise *= signal_norm / (config.snr * noise_norm)
    noisy_signal = clean_signal + complex_noise
    local_field = -np.angle(noisy_signal).astype(np.float32) / config.phase_scale_radians_per_susceptibility_unit
    local_field *= mask
    # Initialise with chi_0 = W * phase_in, W = mask * magnitude.
    chi_init = mask * magnitude * local_field

    np.savez_compressed(
        output,
        local_field=local_field[..., None],
        susceptibility=chi[..., None],
        chi_init=chi_init[..., None],
        brain_mask=mask[..., None],
        reference_mask=mask[..., None],
        magnitude=magnitude[..., None],
        statistical_weight=(mask * magnitude)[..., None],
        clean_local_field=clean_field[..., None],
        clean_signal=clean_signal[..., None],
        noisy_signal=noisy_signal[..., None],
    )
    realised_snr = float(signal_norm / np.linalg.norm((complex_noise * mask).reshape(-1)))
    simulation = asdict(config)
    simulation["voxel_size_zyx"] = list(config.voxel_size_zyx)
    simulation["b0_direction_zyx"] = list(config.b0_direction_zyx)
    manifest = {
        "schema_version": "1.0",
        "anonymized_subject_id": "cosmos-provided-sample",
        "source": "COSMOS chi_cosmos.mat, magn.mat, msk.mat",
        "processing_version": "qsm-tdv-cosmos-simulation-v1",
        "resolved_cosmos_data_dir": str(resolved_data_dir),
        "voxel_size_zyx": list(config.voxel_size_zyx),
        "b0_direction_zyx": list(config.b0_direction_zyx),
        "field_units": "source susceptibility units after noisy phase inversion",
        "susceptibility_units": "source units from chi_cosmos.mat",
        "susceptibility_reference": "as supplied by chi_cosmos.mat; no re-referencing applied",
        "phase_units": "radians",
        "forward_boundary_condition": "periodic unitary Fourier dipole operator",
        "initialization": "chi_0 = W * local_field, W = brain_mask * magnitude",
        "data_weight": "W = brain_mask * magnitude",
        "noise_model": "circular complex Gaussian; realised in-mask L2 SNR",
        "requested_snr": config.snr,
        "realised_snr": realised_snr,
        "simulation": simulation,
        "simulation_hash": _configuration_hash(simulation),
    }
    manifest_path = output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a noisy SNR-controlled QSM sample from COSMOS MAT files.")
    parser.add_argument("--cosmos-data", type=Path, default=Path("/cosmos_data"))
    parser.add_argument("--output", type=Path, default=Path("data/cosmos/cosmos_snr70.npz"))
    parser.add_argument("--snr", type=float, default=70.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--voxel-size", type=float, nargs=3, default=(1.0, 1.0, 1.0), metavar=("VZ", "VY", "VX"))
    parser.add_argument("--b0-direction", type=float, nargs=3, default=(0.0, 0.0, 1.0), metavar=("BZ", "BY", "BX"))
    parser.add_argument("--phase-scale", type=float, default=1.0, help="Radians per source susceptibility unit")
    arguments = parser.parse_args()
    dataset, manifest = prepare_cosmos_overfit_sample(
        arguments.cosmos_data,
        arguments.output,
        CosmosSimulationConfig(
            snr=arguments.snr,
            seed=arguments.seed,
            voxel_size_zyx=tuple(arguments.voxel_size),
            b0_direction_zyx=tuple(arguments.b0_direction),
            phase_scale_radians_per_susceptibility_unit=arguments.phase_scale,
        ),
    )
    print(f"Wrote {dataset}")
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
