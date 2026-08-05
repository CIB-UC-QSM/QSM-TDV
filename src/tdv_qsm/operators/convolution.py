"""Three-dimensional linear operators with explicit exact adjoints.

The learned TDV path uses symmetric (edge-inclusive) extension.  Padding is
implemented as an index map and its transpose as an index accumulation; it is
therefore not confused with PyTorch's reflection padding.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _triple(value: int | Sequence[int], name: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        result = (value, value, value)
    else:
        if len(value) != 3:
            raise ValueError(f"{name} must be an integer or a length-three sequence.")
        result = tuple(int(item) for item in value)
    if any(item < 0 for item in result):
        raise ValueError(f"{name} entries must be nonnegative.")
    return result  # type: ignore[return-value]


def _spatial_shape(shape: Sequence[int]) -> tuple[int, int, int]:
    if len(shape) < 3:
        raise ValueError("output_shape must include three spatial dimensions.")
    spatial = tuple(int(item) for item in shape[-3:])
    if any(item <= 0 for item in spatial):
        raise ValueError("output_shape spatial dimensions must be positive.")
    return spatial  # type: ignore[return-value]


def _symmetric_indices(size: int, pad: int, device: torch.device) -> torch.Tensor:
    if size <= 0:
        raise ValueError("Symmetric padding requires nonempty dimensions.")
    positions = torch.arange(-pad, size + pad, device=device)
    folded = torch.remainder(positions, 2 * size)
    return torch.where(folded < size, folded, 2 * size - 1 - folded).long()


class SymmetricPad3d(nn.Module):
    """Edge-inclusive symmetric extension and its exact transpose."""

    def __init__(self, padding: int | Sequence[int]) -> None:
        super().__init__()
        self.padding = _triple(padding, "padding")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 5:
            raise ValueError("SymmetricPad3d expects [B, C, Z, Y, X].")
        result = value
        for dimension, size, pad in zip((-3, -2, -1), value.shape[-3:], self.padding):
            if pad:
                index = _symmetric_indices(int(size), pad, value.device)
                result = result.index_select(dimension, index)
        return result

    def adjoint(self, value: torch.Tensor, output_shape: Sequence[int]) -> torch.Tensor:
        """Accumulate symmetric replicas into an unpadded requested shape."""

        if value.ndim != 5:
            raise ValueError("SymmetricPad3d.adjoint expects [B, C, Z, Y, X].")
        spatial = _spatial_shape(output_shape)
        expected = tuple(size + 2 * pad for size, pad in zip(spatial, self.padding))
        if tuple(value.shape[-3:]) != expected:
            raise ValueError(
                f"Padded input has spatial shape {tuple(value.shape[-3:])}; expected {expected}."
            )
        result = value
        for dimension, size, pad in zip((-3, -2, -1), spatial, self.padding):
            if pad:
                index = _symmetric_indices(size, pad, value.device)
                target_shape = list(result.shape)
                target_shape[dimension] = size
                target = result.new_zeros(target_shape)
                result = torch.index_add(target, dimension, index, result)
        return result


class AdjointConv3d(nn.Module):
    """Bias-free same-size convolution with a separately callable adjoint."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
    ) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ValueError("in_channels and out_channels must be positive.")
        kernel = _triple(kernel_size, "kernel_size")
        if any(size < 1 or size % 2 != 1 for size in kernel):
            raise ValueError("kernel_size entries must be positive and odd.")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = kernel
        self.padding = SymmetricPad3d(tuple(size // 2 for size in kernel))
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 5 or value.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected [B, {self.in_channels}, Z, Y, X], got {tuple(value.shape)}."
            )
        return F.conv3d(self.padding(value), self.weight, bias=None)

    def adjoint(
        self,
        value: torch.Tensor,
        output_shape: Sequence[int] | None = None,
    ) -> torch.Tensor:
        if value.ndim != 5 or value.shape[1] != self.out_channels:
            raise ValueError(
                f"Expected [B, {self.out_channels}, Z, Y, X], got {tuple(value.shape)}."
            )
        if output_shape is None:
            output_shape = (value.shape[0], self.in_channels, *value.shape[-3:])
        spatial = _spatial_shape(output_shape)
        if tuple(value.shape[-3:]) != spatial:
            raise ValueError("Same-size convolution adjoint requires matching spatial shapes.")
        padded_adjoint = F.conv_transpose3d(value, self.weight, bias=None)
        return self.padding.adjoint(
            padded_adjoint,
            (value.shape[0], self.in_channels, *spatial),
        )


class BinomialDownsample3d(nn.Module):
    """Separable ``[1,4,6,4,1]/16`` filtering followed by stride two."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive.")
        self.channels = int(channels)
        h = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=torch.float32) / 16.0
        kernel = torch.einsum("i,j,k->ijk", h, h, h)[None, None]
        self.register_buffer("kernel", kernel.repeat(channels, 1, 1, 1, 1))
        self.padding = SymmetricPad3d(2)

    def _kernel_like(self, value: torch.Tensor) -> torch.Tensor:
        return self.kernel.to(device=value.device, dtype=value.dtype)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 5 or value.shape[1] != self.channels:
            raise ValueError(
                f"Expected [B, {self.channels}, Z, Y, X], got {tuple(value.shape)}."
            )
        return F.conv3d(
            self.padding(value),
            self._kernel_like(value),
            stride=2,
            groups=self.channels,
        )

    def adjoint(self, value: torch.Tensor, output_shape: Sequence[int]) -> torch.Tensor:
        if value.ndim != 5 or value.shape[1] != self.channels:
            raise ValueError(
                f"Expected [B, {self.channels}, Z, Y, X], got {tuple(value.shape)}."
            )
        spatial = _spatial_shape(output_shape)
        expected_coarse = tuple((size + 1) // 2 for size in spatial)
        if tuple(value.shape[-3:]) != expected_coarse:
            raise ValueError(
                f"Coarse shape {tuple(value.shape[-3:])} does not match requested {spatial}."
            )
        padded_shape = tuple(size + 4 for size in spatial)
        base_shape = tuple(2 * (size - 1) + 5 for size in expected_coarse)
        output_padding = tuple(target - base for target, base in zip(padded_shape, base_shape))
        if any(item not in (0, 1) for item in output_padding):
            raise ValueError("Could not recover the requested odd/even output shape exactly.")
        padded_adjoint = F.conv_transpose3d(
            value,
            self._kernel_like(value),
            stride=2,
            output_padding=output_padding,
            groups=self.channels,
        )
        return self.padding.adjoint(
            padded_adjoint,
            (value.shape[0], self.channels, *spatial),
        )


class ScaledConv3d(nn.Module):
    """Learned convolution followed by binomial antialiasing and decimation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.learned = AdjointConv3d(in_channels, out_channels, kernel_size)
        self.antialias = BinomialDownsample3d(out_channels)

    @property
    def weight(self) -> nn.Parameter:
        return self.learned.weight

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.antialias(self.learned(value))

    def adjoint(self, value: torch.Tensor, output_shape: Sequence[int]) -> torch.Tensor:
        spatial = _spatial_shape(output_shape)
        fine = self.antialias.adjoint(
            value,
            (value.shape[0], self.out_channels, *spatial),
        )
        return self.learned.adjoint(
            fine,
            (value.shape[0], self.in_channels, *spatial),
        )
