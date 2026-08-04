"""Periodic, unitary-FFT QSM dipole operators.

Images use ``[B, C, Z, Y, X]`` and metadata is always in ``zyx`` order.
The operator intentionally uses periodic Fourier boundaries: no padding,
cropping, masking, or dipole thresholding is hidden in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

FFT_DIMS = (-3, -2, -1)


def _triple(values: Sequence[float], name: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{name} must have exactly three values in zyx order.")
    result = tuple(float(value) for value in values)
    return result  # type: ignore[return-value]


def _validate_volume(value: torch.Tensor, kernel: torch.Tensor) -> None:
    if value.ndim != 5:
        raise ValueError(f"Expected [B, C, Z, Y, X], received {tuple(value.shape)}.")
    if tuple(value.shape[-3:]) != tuple(kernel.shape):
        raise ValueError(
            "The dipole-kernel spatial shape must match the input volume; "
            f"got {tuple(kernel.shape)} and {tuple(value.shape[-3:])}."
        )


def fft3(value: torch.Tensor) -> torch.Tensor:
    """Unitary float32/complex64 3-D Fourier transform."""

    return torch.fft.fftn(value.float(), dim=FFT_DIMS, norm="ortho")


def ifft3(value: torch.Tensor) -> torch.Tensor:
    """Unitary inverse transform, returned as a float32 real image."""

    return torch.fft.ifftn(value, dim=FFT_DIMS, norm="ortho").real.float()


def build_dipole_kernel(
    shape_zyx: Sequence[int],
    voxel_size_zyx: Sequence[float] = (1.0, 1.0, 1.0),
    b0_direction_zyx: Sequence[float] = (0.0, 0.0, 1.0),
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return an unthresholded continuous dipole kernel in native FFT order.

    ``torch.fft.fftfreq`` directly produces the unshifted frequency ordering
    required by :func:`fft3`.  The kernel is real float32 and ``d(0)=0``.
    """

    if len(shape_zyx) != 3 or any(int(size) <= 0 for size in shape_zyx):
        raise ValueError("shape_zyx must contain three positive dimensions.")
    voxel_size = _triple(voxel_size_zyx, "voxel_size_zyx")
    direction = _triple(b0_direction_zyx, "b0_direction_zyx")
    if any(size <= 0.0 for size in voxel_size):
        raise ValueError("voxel_size_zyx must be strictly positive.")

    b0 = torch.tensor(direction, dtype=torch.float32, device=device)
    b0_norm = torch.linalg.vector_norm(b0)
    if b0_norm.item() == 0.0:
        raise ValueError("b0_direction_zyx must be nonzero.")
    b0 = b0 / b0_norm

    z, y, x = (int(size) for size in shape_zyx)
    kz = torch.fft.fftfreq(z, d=voxel_size[0], device=device, dtype=torch.float32)
    ky = torch.fft.fftfreq(y, d=voxel_size[1], device=device, dtype=torch.float32)
    kx = torch.fft.fftfreq(x, d=voxel_size[2], device=device, dtype=torch.float32)
    grid_z, grid_y, grid_x = torch.meshgrid(kz, ky, kx, indexing="ij")
    k_squared = grid_z.square() + grid_y.square() + grid_x.square()
    k_dot_b0 = grid_z * b0[0] + grid_y * b0[1] + grid_x * b0[2]
    nonzero = k_squared > 0.0
    kernel = torch.zeros_like(k_squared)
    kernel[nonzero] = (
        1.0 / 3.0 - k_dot_b0[nonzero].square() / k_squared[nonzero]
    )
    # This is explicit even though the initialized zero above already does it.
    kernel[0, 0, 0] = 0.0
    return kernel.float()


@dataclass(frozen=True)
class DipoleOperator3D:
    """``A=FᴴDF`` and its separately exposed adjoint ``Aᴴ=Fᴴconj(D)F``."""

    dipole_kernel: torch.Tensor

    def __post_init__(self) -> None:
        kernel = self.dipole_kernel
        if kernel.ndim != 3:
            raise ValueError("dipole_kernel must have shape [Z, Y, X].")
        if not torch.isfinite(kernel).all():
            raise ValueError("dipole_kernel must contain only finite values.")

    def forward(self, chi: torch.Tensor) -> torch.Tensor:
        """Apply the physical forward operator using float32/complex64 FFTs."""

        _validate_volume(chi, self.dipole_kernel)
        kernel = self.dipole_kernel.to(device=chi.device, dtype=torch.float32)
        return ifft3(fft3(chi) * kernel[None, None, ...])

    def adjoint(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the exact Fourier adjoint without assuming it equals forward."""

        _validate_volume(value, self.dipole_kernel)
        kernel = self.dipole_kernel.to(device=value.device, dtype=torch.float32)
        return ifft3(fft3(value) * kernel.conj()[None, None, ...])
