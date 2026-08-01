"""The periodic, unitary-Fourier QSM dipole forward operator.

The spatial layout throughout this package is ``[B, Z, Y, X, C]``.  Metadata
uses ``(z, y, x)`` order too: voxel size is in mm and B0 is a direction in the
same physical coordinate system.  The operator uses periodic boundary
conditions; padding and crop operators are intentionally not hidden in it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp

Array = jax.Array


@dataclass(frozen=True)
class DipoleMetadata:
    """Explicit spatial metadata required for a QSM forward model."""

    voxel_size_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0)
    b0_direction_zyx: tuple[float, float, float] = (0.0, 0.0, 1.0)
    field_units: str = "ppm"
    susceptibility_units: str = "ppm"
    susceptibility_reference: str = "mean-zero within brain mask"

    def __post_init__(self) -> None:
        if len(self.voxel_size_zyx) != 3 or len(self.b0_direction_zyx) != 3:
            raise ValueError("voxel_size_zyx and b0_direction_zyx must each have length 3")
        if any(v <= 0 for v in self.voxel_size_zyx):
            raise ValueError("All voxel sizes must be positive")
        if sum(float(v) ** 2 for v in self.b0_direction_zyx) == 0.0:
            raise ValueError("b0_direction_zyx must be non-zero")


def dipole_kernel(
    shape_zyx: Sequence[int],
    voxel_size_zyx: Sequence[float] = (1.0, 1.0, 1.0),
    b0_direction_zyx: Sequence[float] = (0.0, 0.0, 1.0),
    *,
    dtype: jnp.dtype = jnp.float32,
) -> Array:
    """Build the unthresholded continuous dipole kernel on an FFT grid.

    ``jnp.fft.fftfreq`` gives the native (unshifted) FFT ordering, so this
    array can be multiplied directly by an FFT.  The only special case is the
    exact DC frequency, for which the defined value is zero.
    """

    if len(shape_zyx) != 3 or any(int(n) <= 0 for n in shape_zyx):
        raise ValueError("shape_zyx must contain three positive dimensions")
    metadata = DipoleMetadata(tuple(float(v) for v in voxel_size_zyx), tuple(float(v) for v in b0_direction_zyx))
    z, y, x = (int(n) for n in shape_zyx)
    vz, vy, vx = metadata.voxel_size_zyx
    kz = jnp.fft.fftfreq(z, d=vz).astype(dtype)
    ky = jnp.fft.fftfreq(y, d=vy).astype(dtype)
    kx = jnp.fft.fftfreq(x, d=vx).astype(dtype)
    grid_z, grid_y, grid_x = jnp.meshgrid(kz, ky, kx, indexing="ij")

    b0 = jnp.asarray(metadata.b0_direction_zyx, dtype=dtype)
    b0 = b0 / jnp.linalg.norm(b0)
    k2 = grid_z**2 + grid_y**2 + grid_x**2
    k_dot_b0 = grid_z * b0[0] + grid_y * b0[1] + grid_x * b0[2]
    nonzero = k2 != 0
    safe_k2 = jnp.where(nonzero, k2, jnp.ones_like(k2))
    kernel = jnp.where(nonzero, 1.0 / 3.0 - (k_dot_b0**2) / safe_k2, 0.0)
    return kernel.astype(dtype)


def _check_volume(x: Array, kernel: Array) -> None:
    if x.ndim != 5:
        raise ValueError(f"Expected NDHWC volume [B,Z,Y,X,C], got shape {x.shape}")
    if tuple(x.shape[1:4]) != tuple(kernel.shape):
        raise ValueError(
            "Kernel spatial shape must match volume: "
            f"volume {x.shape[1:4]}, kernel {kernel.shape}"
        )


def _fft3(x: Array) -> Array:
    return jnp.fft.fftn(x.astype(jnp.complex64), axes=(1, 2, 3), norm="ortho")


def _ifft3(x: Array) -> Array:
    return jnp.fft.ifftn(x.astype(jnp.complex64), axes=(1, 2, 3), norm="ortho")


def apply_dipole(chi: Array, kernel: Array) -> Array:
    """Apply ``A = Fᴴ D F`` with a real periodic dipole kernel."""

    _check_volume(chi, kernel)
    spectrum = _fft3(chi)
    output = _ifft3(spectrum * kernel[None, ..., None].astype(jnp.complex64))
    return jnp.real(output).astype(jnp.float32)


def dipole_adjoint(field: Array, kernel: Array) -> Array:
    """Apply the exact FFT adjoint ``Aᴴ = Fᴴ conj(D) F``."""

    _check_volume(field, kernel)
    spectrum = _fft3(field)
    output = _ifft3(spectrum * jnp.conj(kernel[None, ..., None].astype(jnp.complex64)))
    return jnp.real(output).astype(jnp.float32)


def normal_operator(chi: Array, kernel: Array) -> Array:
    """Apply ``Aᴴ A`` without treating any inverse as a forward operator."""

    return dipole_adjoint(apply_dipole(chi, kernel), kernel)
