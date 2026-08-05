"""Periodic float32/complex64 QSM dipole operator.

This QSM physics module is a project-specific extension of TDV.  It uses
periodic Fourier boundaries and performs no padding, cropping, masking,
inverse filtering, or dipole thresholding.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

FFT_DIMS = (-3, -2, -1)


def fft3(value: torch.Tensor) -> torch.Tensor:
    """Unitary FFT with a float32 real input and complex64 output."""

    return torch.fft.fftn(value.float(), dim=FFT_DIMS, norm="ortho")


def ifft3(value: torch.Tensor) -> torch.Tensor:
    """Unitary inverse FFT returned as a float32 real volume."""

    return torch.fft.ifftn(value.to(torch.complex64), dim=FFT_DIMS, norm="ortho").real.float()


def _metadata_rows(
    values: Sequence[float] | torch.Tensor,
    name: str,
    *,
    device: torch.device | str | None,
) -> tuple[torch.Tensor, bool]:
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    batched = tensor.ndim == 2
    if tensor.ndim == 1:
        tensor = tensor[None]
    if tensor.ndim != 2 or tensor.shape[1] != 3:
        raise ValueError(f"{name} must have shape [3] or [B, 3] in zyx order.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values.")
    return tensor, batched


def build_dipole_kernel(
    shape_zyx: Sequence[int],
    voxel_size_zyx: Sequence[float] | torch.Tensor = (1.0, 1.0, 1.0),
    b0_direction_zyx: Sequence[float] | torch.Tensor = (0.0, 0.0, 1.0),
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build unthresholded dipole kernels for scalar or batched metadata.

    Scalar metadata returns ``[Z,Y,X]`` for backward compatibility.  Batched
    metadata returns ``[B,1,Z,Y,X]`` and supports a distinct anisotropic voxel
    size and normalized field direction for every sample.
    """

    if len(shape_zyx) != 3 or any(int(size) <= 0 for size in shape_zyx):
        raise ValueError("shape_zyx must contain three positive dimensions.")
    voxel_size, voxel_batched = _metadata_rows(
        voxel_size_zyx,
        "voxel_size_zyx",
        device=device,
    )
    b0, b0_batched = _metadata_rows(
        b0_direction_zyx,
        "b0_direction_zyx",
        device=device,
    )
    if torch.any(voxel_size <= 0.0):
        raise ValueError("voxel_size_zyx must be strictly positive.")
    batch = max(voxel_size.shape[0], b0.shape[0])
    if voxel_size.shape[0] not in (1, batch) or b0.shape[0] not in (1, batch):
        raise ValueError("voxel_size_zyx and b0_direction_zyx batch sizes must broadcast.")
    voxel_size = voxel_size.expand(batch, -1)
    b0 = b0.expand(batch, -1)
    b0_norm = torch.linalg.vector_norm(b0, ord=2, dim=1, keepdim=True)
    if torch.any(b0_norm == 0.0):
        raise ValueError("Every b0_direction_zyx must be nonzero.")
    b0 = b0 / b0_norm

    z, y, x = (int(size) for size in shape_zyx)
    kz = torch.fft.fftfreq(z, device=device, dtype=torch.float32)[None, :, None, None]
    ky = torch.fft.fftfreq(y, device=device, dtype=torch.float32)[None, None, :, None]
    kx = torch.fft.fftfreq(x, device=device, dtype=torch.float32)[None, None, None, :]
    kz = kz / voxel_size[:, 0, None, None, None]
    ky = ky / voxel_size[:, 1, None, None, None]
    kx = kx / voxel_size[:, 2, None, None, None]
    k_squared = kz.square() + ky.square() + kx.square()
    k_dot_b0 = (
        kz * b0[:, 0, None, None, None]
        + ky * b0[:, 1, None, None, None]
        + kx * b0[:, 2, None, None, None]
    )
    nonzero = k_squared > 0.0
    kernel = torch.zeros_like(k_squared)
    kernel[nonzero] = 1.0 / 3.0 - k_dot_b0[nonzero].square() / k_squared[nonzero]
    kernel[:, 0, 0, 0] = 0.0
    if voxel_batched or b0_batched:
        return kernel[:, None].float()
    return kernel[0].float()


class QSMOperator(nn.Module):
    """Separately testable ``A=F^HDF`` and ``A^H=F^Hconj(D)F``."""

    def __init__(self, dipole_kernel: torch.Tensor | None = None) -> None:
        super().__init__()
        if dipole_kernel is not None:
            self._validate_kernel(dipole_kernel)
            dipole_kernel = dipole_kernel.detach().clone()
        self.register_buffer("dipole_kernel", dipole_kernel)

    @staticmethod
    def build_dipole_kernel(
        shape_zyx: Sequence[int],
        voxel_size_zyx: Sequence[float] | torch.Tensor = (1.0, 1.0, 1.0),
        b0_direction_zyx: Sequence[float] | torch.Tensor = (0.0, 0.0, 1.0),
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        return build_dipole_kernel(
            shape_zyx,
            voxel_size_zyx,
            b0_direction_zyx,
            device=device,
        )

    @staticmethod
    def _validate_kernel(kernel: torch.Tensor) -> None:
        if kernel.ndim not in (3, 4, 5):
            raise ValueError("dipole_kernel must be [Z,Y,X], [B,Z,Y,X], or [B,1,Z,Y,X].")
        if kernel.ndim == 5 and kernel.shape[1] != 1:
            raise ValueError("A rank-five dipole_kernel must have one channel.")
        if not torch.isfinite(kernel).all():
            raise ValueError("dipole_kernel must contain only finite values.")

    def _kernel_for(
        self,
        value: torch.Tensor,
        dipole_kernel: torch.Tensor | None,
    ) -> torch.Tensor:
        if value.ndim != 5 or value.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, Z, Y, X], got {tuple(value.shape)}.")
        kernel = self.dipole_kernel if dipole_kernel is None else dipole_kernel
        if kernel is None:
            raise ValueError("A dipole kernel must be set on the operator or passed to the call.")
        self._validate_kernel(kernel)
        if kernel.ndim == 3:
            kernel = kernel[None, None]
        elif kernel.ndim == 4:
            kernel = kernel[:, None]
        if tuple(kernel.shape[-3:]) != tuple(value.shape[-3:]):
            raise ValueError("The dipole-kernel spatial shape must match the input volume.")
        if kernel.shape[0] not in (1, value.shape[0]):
            raise ValueError("The dipole-kernel batch must be one or match the input batch.")
        dtype = torch.complex64 if kernel.is_complex() else torch.float32
        return kernel.to(device=value.device, dtype=dtype)

    def forward(
        self,
        chi: torch.Tensor,
        dipole_kernel: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kernel = self._kernel_for(chi, dipole_kernel)
        return ifft3(fft3(chi) * kernel)

    def adjoint(
        self,
        field: torch.Tensor,
        dipole_kernel: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kernel = self._kernel_for(field, dipole_kernel)
        return ifft3(fft3(field) * kernel.conj())


# Compatibility name retained for existing experiment scripts.
DipoleOperator3D = QSMOperator
