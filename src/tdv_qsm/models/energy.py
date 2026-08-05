"""Source-grounded scalar TDV energy and differentiable manual force."""

from __future__ import annotations

import math

import torch
from torch import nn

from tdv_qsm.models.blocks import MacroBlock3D, MacroBlockCache
from tdv_qsm.operators.convolution import AdjointConv3d


@torch.no_grad()
def project_analysis_kernel_(weight: torch.Tensor) -> None:
    """Project analysis filters to zero mean and an at-most-unit L2 norm."""

    if weight.ndim != 5:
        raise ValueError("Analysis weight must be [out, in, kz, ky, kx].")
    reduce_dims = (1, 2, 3, 4)
    weight.sub_(weight.mean(dim=reduce_dims, keepdim=True))
    norm = torch.linalg.vector_norm(
        weight,
        ord=2,
        dim=reduce_dims,
        keepdim=True,
    )
    weight.div_(norm.clamp_min(1.0))


class TDVEnergy3D(nn.Module):
    """Three-dimensional adaptation of the source TDV energy architecture.

    The 3-D convolutions and CUDA AMP option are project extensions.  The
    source-style definition retained here is a one-channel signed energy head
    divided by ``num_features`` and summed spatially; it is not squared and is
    not an image-to-image predictor.
    """

    def __init__(
        self,
        *,
        num_features: int = 4,
        num_macro_blocks: int = 1,
        num_scales: int = 3,
        kernel_size: int = 3,
        use_amp: bool = True,
        energy_head_initialization_scale: float = 0.2,
        features: int | None = None,
        macro_blocks: int | None = None,
    ) -> None:
        super().__init__()
        # The aliases keep existing experiment configurations loadable while
        # the public terminology follows the new contract.
        if features is not None:
            num_features = features
        if macro_blocks is not None:
            num_macro_blocks = macro_blocks
        if num_features < 1 or num_macro_blocks < 1 or num_scales < 1:
            raise ValueError("num_features, num_macro_blocks, and num_scales must be positive.")
        if energy_head_initialization_scale <= 0.0:
            raise ValueError("energy_head_initialization_scale must be positive.")
        self.num_features = int(num_features)
        self.num_scales = int(num_scales)
        self.use_amp = bool(use_amp)
        self.energy_head_initialization_scale = float(energy_head_initialization_scale)
        self.analysis = AdjointConv3d(1, num_features, kernel_size)
        self.macro_blocks = nn.ModuleList(
            MacroBlock3D(
                num_features=num_features,
                num_scales=num_scales,
                kernel_size=kernel_size,
            )
            for _ in range(num_macro_blocks)
        )
        self.energy_head = AdjointConv3d(num_features, 1, kernel_size=1)
        self._initialize_analysis_kernel()
        with torch.no_grad():
            bound = self.energy_head_initialization_scale / math.sqrt(num_features)
            self.energy_head.weight.uniform_(-bound, bound)

    def _initialize_analysis_kernel(self) -> None:
        weight = self.analysis.weight
        fan_in = weight.shape[1] * weight.shape[2] * weight.shape[3] * weight.shape[4]
        with torch.no_grad():
            weight.normal_(mean=0.0, std=math.sqrt(1.0 / fan_in))
            weight.sub_(weight.mean(dim=(1, 2, 3, 4), keepdim=True))
            norm = torch.linalg.vector_norm(
                weight,
                ord=2,
                dim=(1, 2, 3, 4),
                keepdim=True,
            )
            nonzero = norm > 0.0
            weight.div_(torch.where(nonzero, norm, torch.ones_like(norm)))

    def _amp_enabled(self, chi: torch.Tensor) -> bool:
        return self.use_amp and chi.is_cuda and chi.dtype == torch.float32

    def _forward_states(
        self,
        chi: torch.Tensor,
        *,
        cache: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[MacroBlockCache]]:
        if chi.ndim != 5 or chi.shape[1] != 1:
            raise ValueError(f"TDV requires chi [B, 1, Z, Y, X], got {tuple(chi.shape)}.")
        caches: list[MacroBlockCache] = []
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=self._amp_enabled(chi),
        ):
            analysis = self.analysis(chi)
            states: list[torch.Tensor | None] = [analysis] + [None] * (self.num_scales - 1)
            for block in self.macro_blocks:
                if cache:
                    complete_states, block_cache = block.forward_with_cache(states)
                    caches.append(block_cache)
                else:
                    complete_states = block(states)
                states = complete_states
            final_states = [state for state in states if state is not None]
            response = self.energy_head(final_states[0])
        return response, final_states, caches

    def energy_density(self, chi: torch.Tensor) -> torch.Tensor:
        """Return the one-channel voxelwise TDV density ``T_theta(chi) / m``."""

        response, _, _ = self._forward_states(chi, cache=False)
        return response / self.num_features

    def energy(self, chi: torch.Tensor) -> torch.Tensor:
        """Return the float32 spatial sum of the TDV density for each sample."""

        density = self.energy_density(chi)
        return density.float().flatten(1).sum(dim=1)

    def force(self, chi: torch.Tensor) -> torch.Tensor:
        """Apply the source-style manual transpose chain for ``grad R_theta``.

        All operations remain differentiable with respect to regularizer
        parameters.  No activation is retained as mutable module state.
        """

        response, states, caches = self._forward_states(chi, cache=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=self._amp_enabled(chi),
        ):
            head_adjoint = self.energy_head.adjoint(
                torch.ones_like(response) / self.num_features,
                states[0].shape,
            )
            state_adjoints: list[torch.Tensor | None] = [head_adjoint]
            state_adjoints.extend(torch.zeros_like(state) for state in states[1:])
            for block, block_cache in zip(
                reversed(self.macro_blocks),
                reversed(caches),
            ):
                state_adjoints = block.jacobian_transpose(block_cache, state_adjoints)
            analysis_adjoint = state_adjoints[0]
            assert analysis_adjoint is not None
            force = self.analysis.adjoint(analysis_adjoint, chi.shape)
        return force.float()

    def force_autograd_reference(self, chi: torch.Tensor) -> torch.Tensor:
        """Slow autograd correctness oracle; production reconstruction uses ``force``."""

        with torch.enable_grad():
            differentiation_input = chi
            if not differentiation_input.requires_grad:
                differentiation_input = chi.detach().requires_grad_(True)
            energy = self.energy(differentiation_input)
            force, = torch.autograd.grad(
                energy.sum(),
                differentiation_input,
                create_graph=True,
            )
        return force.float()

    def project_analysis_kernel_(self) -> None:
        project_analysis_kernel_(self.analysis.weight)

    def project_zero_mean_(self) -> None:
        """Compatibility alias; now enforces both required K1 constraints."""

        self.project_analysis_kernel_()
