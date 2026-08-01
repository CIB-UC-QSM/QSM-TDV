"""Deterministic, metadata-complete synthetic QSM sample generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from qsm_tdv.physics.dipole import apply_dipole, dipole_kernel


def _configuration_hash(configuration: dict[str, object]) -> str:
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def create_synthetic_sample(
    output_path: str | Path,
    *,
    shape_zyx: tuple[int, int, int] = (16, 16, 16),
    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    b0_direction_zyx: tuple[float, float, float] = (0.0, 0.0, 1.0),
    noise_std: float = 0.002,
    seed: int = 0,
) -> tuple[Path, Path]:
    """Create one non-identifiable sample and its data-contract manifest."""

    if any(n < 8 for n in shape_zyx):
        raise ValueError("Synthetic volume dimensions must each be at least 8")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix != ".npz":
        output = output.with_suffix(".npz")

    rng = np.random.default_rng(seed)
    z, y, x = np.meshgrid(
        np.linspace(-1.0, 1.0, shape_zyx[0], dtype=np.float32),
        np.linspace(-1.0, 1.0, shape_zyx[1], dtype=np.float32),
        np.linspace(-1.0, 1.0, shape_zyx[2], dtype=np.float32),
        indexing="ij",
    )
    brain = (((z / 0.92) ** 2 + (y / 0.84) ** 2 + (x / 0.78) ** 2) <= 1.0).astype(np.float32)
    susceptibility = (
        0.090 * np.exp(-((z + 0.26) / 0.22) ** 2 - ((y - 0.12) / 0.18) ** 2 - ((x + 0.08) / 0.16) ** 2)
        - 0.070 * np.exp(-((z - 0.22) / 0.18) ** 2 - ((y + 0.18) / 0.25) ** 2 - ((x - 0.18) / 0.20) ** 2)
        + 0.035 * np.exp(-((z + 0.04) / 0.42) ** 2 - ((y + 0.20) / 0.11) ** 2 - ((x - 0.27) / 0.15) ** 2)
    ).astype(np.float32)
    susceptibility *= brain
    susceptibility -= brain * (susceptibility.sum() / (brain.sum() + np.finfo(np.float32).eps))

    kernel = dipole_kernel(shape_zyx, voxel_size_zyx, b0_direction_zyx)
    susceptibility_batch = jnp.asarray(susceptibility[None, ..., None])
    clean_field = np.asarray(apply_dipole(susceptibility_batch, kernel))[0, ..., 0]
    noisy_field = clean_field + noise_std * rng.normal(size=shape_zyx).astype(np.float32) * brain
    # No magnitude is available, so W=brain and chi_0=W*local_field.
    chi_init = brain * noisy_field

    np.savez_compressed(
        output,
        local_field=noisy_field[..., None].astype(np.float32),
        susceptibility=susceptibility[..., None].astype(np.float32),
        chi_init=chi_init[..., None].astype(np.float32),
        brain_mask=brain[..., None].astype(np.float32),
        reference_mask=brain[..., None].astype(np.float32),
    )
    configuration: dict[str, object] = {
        "shape_zyx": list(shape_zyx),
        "voxel_size_zyx": list(voxel_size_zyx),
        "b0_direction_zyx": list(b0_direction_zyx),
        "noise_std": noise_std,
        "seed": seed,
        "phantom": "three smooth susceptibility inclusions",
    }
    manifest = {
        "schema_version": "1.0",
        "anonymized_subject_id": "synthetic-0001",
        "source": "deterministic synthetic QSM phantom",
        "processing_version": "qsm-tdv-synthetic-v1",
        "voxel_size_zyx": list(voxel_size_zyx),
        "b0_direction_zyx": list(b0_direction_zyx),
        "field_units": "ppm",
        "susceptibility_units": "ppm",
        "susceptibility_reference": "mean-zero within brain_mask",
        "boundary_condition": "periodic Fourier dipole operator; zero-padded TDV convolutions",
        "initialization": "chi_0 = W * local_field, W = brain_mask (magnitude unavailable)",
        "configuration": configuration,
        "configuration_hash": _configuration_hash(configuration),
    }
    manifest_path = output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a deterministic single-sample QSM TDV overfit dataset.")
    parser.add_argument("--output", type=Path, default=Path("data/synthetic/synthetic_0001.npz"))
    parser.add_argument("--shape", type=int, nargs=3, default=(16, 16, 16), metavar=("Z", "Y", "X"))
    parser.add_argument("--voxel-size", type=float, nargs=3, default=(1.0, 1.0, 1.0), metavar=("VZ", "VY", "VX"))
    parser.add_argument("--b0-direction", type=float, nargs=3, default=(0.0, 0.0, 1.0), metavar=("BZ", "BY", "BX"))
    parser.add_argument("--noise-std", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    dataset, manifest = create_synthetic_sample(
        arguments.output,
        shape_zyx=tuple(arguments.shape),
        voxel_size_zyx=tuple(arguments.voxel_size),
        b0_direction_zyx=tuple(arguments.b0_direction),
        noise_std=arguments.noise_std,
        seed=arguments.seed,
    )
    print(f"Wrote {dataset}")
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
