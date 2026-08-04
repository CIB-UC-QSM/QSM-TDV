"""Learned scalar TDV energy for 3-D susceptibility maps."""

from __future__ import annotations

import torch
from torch import nn

from tdv_qsm.models.blocks import MacroBlock3D


class TDVEnergy3D(nn.Module):
    """Map an image to a nonnegative density and scalar energy per sample.

    The bias-free CNN produces a signed response ``h_theta(chi)``.  Its local
    potential is ``0.5 * h_theta(chi)^2``, so every voxel contribution is
    nonnegative and ``R_theta(0)=0``.
    """

    def __init__(
        self,
        *,
        features: int = 4,
        macro_blocks: int = 1,
        kernel_size: int = 3,
        nu: float = 9.0,
        energy_head_initialization_scale: float = 0.2,
    ) -> None:
        super().__init__()
        if features < 1 or macro_blocks < 1:
            raise ValueError("features and macro_blocks must be positive.")
        if energy_head_initialization_scale <= 0.0:
            raise ValueError("energy_head_initialization_scale must be positive.")
        self.energy_head_initialization_scale = float(energy_head_initialization_scale)
        padding = kernel_size // 2
        self.analysis = nn.Conv3d(1, features, kernel_size, padding=padding, bias=False)
        self.macro_blocks = nn.ModuleList(
            MacroBlock3D(features, kernel_size=kernel_size, nu=nu)
            for _ in range(macro_blocks)
        )
        self.energy_head = nn.Conv3d(features, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            bound = self.energy_head_initialization_scale / features**0.5
            self.energy_head.weight.uniform_(-bound, bound)
        self.project_zero_mean_()

    def _signed_energy_response(self, chi: torch.Tensor) -> torch.Tensor:
        if chi.ndim != 5 or chi.shape[1] != 1:
            raise ValueError(f"TDV requires chi [B, 1, Z, Y, X], got {tuple(chi.shape)}.")
        x = self.analysis(chi)
        for block in self.macro_blocks:
            x = block(x)
        return self.energy_head(x)

    def energy_density(self, chi: torch.Tensor) -> torch.Tensor:
        """Return ``0.5*h_theta(chi)^2`` as a float32 local potential."""

        response = self._signed_energy_response(chi)
        # Square in float32 even when the CNN runs under CUDA float16 autocast.
        return 0.5 * response.float().square()

    def energy(self, chi: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute float32 ``R_theta`` with AMP restricted to TDV convolutions."""

        if mask.shape != chi.shape:
            raise ValueError("mask must have the same shape as chi.")
        mask = mask.float()
        if not torch.isfinite(mask).all():
            raise ValueError("mask must contain only finite values.")
        if torch.any(mask < 0.0):
            raise ValueError("mask must be nonnegative to preserve R_theta(chi) >= 0.")
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=chi.is_cuda,
        ):
            density = self.energy_density(chi.float())
        return (density * mask).flatten(1).sum(dim=1).float()

    @torch.no_grad()
    def project_zero_mean_(self) -> None:
        """Enforce the zero-sum constraint on every analysis output filter."""

        weight = self.analysis.weight
        weight.sub_(weight.mean(dim=(1, 2, 3, 4), keepdim=True))
