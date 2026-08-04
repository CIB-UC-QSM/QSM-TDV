"""Multiscale bias-free TDV building blocks."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from tdv_qsm.models.activation import log_student_t


class MicroBlock3D(nn.Module):
    """``u + K2(phi(K1(u)))`` with the prescribed TDV activation."""

    def __init__(self, channels: int, *, kernel_size: int = 3, nu: float = 9.0) -> None:
        super().__init__()
        if channels < 1 or kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("channels and an odd positive kernel_size are required.")
        self.nu = float(nu)
        padding = kernel_size // 2
        self.conv1 = nn.Conv3d(channels, channels, kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(log_student_t(self.conv1(x), self.nu))


class AntiAliasedDownsample3D(nn.Module):
    """Box-filter low-pass followed by decimation, including odd input sizes."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool3d(x, kernel_size=2, stride=2, ceil_mode=True)


class MacroBlock3D(nn.Module):
    """Three-scale TDV macro-block with exactly five residual micro-blocks.

    Convolutions use zero padding.  The physical dipole operator remains
    periodic and separate; this block has no hidden physical padding/cropping.
    """

    def __init__(self, features: int, *, kernel_size: int = 3, nu: float = 9.0) -> None:
        super().__init__()
        f2, f4 = 2 * features, 4 * features
        padding = kernel_size // 2
        self.pre = MicroBlock3D(features, kernel_size=kernel_size, nu=nu)
        self.downsample_1 = AntiAliasedDownsample3D()
        self.down_1 = nn.Conv3d(features, f2, kernel_size, padding=padding, bias=False)
        self.scale_2 = MicroBlock3D(f2, kernel_size=kernel_size, nu=nu)
        self.downsample_2 = AntiAliasedDownsample3D()
        self.down_2 = nn.Conv3d(f2, f4, kernel_size, padding=padding, bias=False)
        self.scale_3 = MicroBlock3D(f4, kernel_size=kernel_size, nu=nu)
        self.up_1 = nn.Conv3d(f4, f2, kernel_size=1, bias=False)
        self.fuse_1 = nn.Conv3d(2 * f2, f2, kernel_size=1, bias=False)
        self.post_up_1 = MicroBlock3D(f2, kernel_size=kernel_size, nu=nu)
        self.up_0 = nn.Conv3d(f2, features, kernel_size=1, bias=False)
        self.fuse_0 = nn.Conv3d(2 * features, features, kernel_size=1, bias=False)
        self.post_up_0 = MicroBlock3D(features, kernel_size=kernel_size, nu=nu)

    @staticmethod
    def _upsample_to(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x,
            size=reference.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        level_0 = self.pre(x)
        level_1 = self.scale_2(self.down_1(self.downsample_1(level_0)))
        level_2 = self.scale_3(self.down_2(self.downsample_2(level_1)))
        up_1 = self.up_1(self._upsample_to(level_2, level_1))
        level_1_out = self.post_up_1(self.fuse_1(torch.cat((up_1, level_1), dim=1)))
        up_0 = self.up_0(self._upsample_to(level_1_out, level_0))
        return self.post_up_0(self.fuse_0(torch.cat((up_0, level_0), dim=1)))
