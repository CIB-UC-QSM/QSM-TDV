"""Source-style multiscale TDV blocks with manual Jacobian transposes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from tdv_qsm.models.activation import student_t_pair
from tdv_qsm.operators.convolution import AdjointConv3d, ScaledConv3d


@dataclass(frozen=True)
class MicroBlockCache:
    """Call-local data needed for a micro-block Jacobian transpose."""

    preactivation: torch.Tensor


class MicroBlock3D(nn.Module):
    """Bias-free ``u + K2(phi(K1(u)))`` with an explicit ``J^T`` action."""

    def __init__(self, channels: int, *, kernel_size: int = 3) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive.")
        self.conv1 = AdjointConv3d(channels, channels, kernel_size)
        self.conv2 = AdjointConv3d(channels, channels, kernel_size)

    def forward_with_cache(
        self,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, MicroBlockCache]:
        preactivation = self.conv1(value)
        activated, _ = student_t_pair(preactivation)
        output = value + self.conv2(activated.to(dtype=preactivation.dtype))
        return output, MicroBlockCache(preactivation=preactivation)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward_with_cache(value)
        return output

    def jacobian_transpose_from_cache(
        self,
        cache: MicroBlockCache,
        output_adjoint: torch.Tensor,
    ) -> torch.Tensor:
        _, derivative = student_t_pair(cache.preactivation)
        conv2_adjoint = self.conv2.adjoint(output_adjoint, cache.preactivation.shape)
        modulated = derivative.to(dtype=conv2_adjoint.dtype) * conv2_adjoint
        return output_adjoint + self.conv1.adjoint(modulated, output_adjoint.shape)

    def jacobian_transpose(
        self,
        value: torch.Tensor,
        output_adjoint: torch.Tensor,
    ) -> torch.Tensor:
        _, cache = self.forward_with_cache(value)
        return self.jacobian_transpose_from_cache(cache, output_adjoint)


@dataclass(frozen=True)
class MacroBlockCache:
    """All call-local values needed to reverse one multiscale macro-block."""

    input_present: tuple[bool, ...]
    output_states: tuple[torch.Tensor, ...]
    pre_outputs: tuple[torch.Tensor, ...]
    pre_caches: tuple[MicroBlockCache, ...]
    coarsest_cache: MicroBlockCache
    post_caches: tuple[MicroBlockCache, ...]


class MacroBlock3D(nn.Module):
    """Persistent multiscale state with source-order down and up traversals.

    With three scales this contains exactly five micro-blocks: two at each
    noncoarsest scale and one at the coarsest scale.  Down and up operators
    have independent learned weights.  The up path calls the exact adjoint of
    its own learned antialiased scaled convolution.
    """

    def __init__(
        self,
        num_features: int,
        *,
        num_scales: int = 3,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if num_features < 1 or num_scales < 1:
            raise ValueError("num_features and num_scales must be positive.")
        self.num_features = int(num_features)
        self.num_scales = int(num_scales)
        self.pre_blocks = nn.ModuleList(
            MicroBlock3D(num_features, kernel_size=kernel_size)
            for _ in range(num_scales - 1)
        )
        self.post_blocks = nn.ModuleList(
            MicroBlock3D(num_features, kernel_size=kernel_size)
            for _ in range(num_scales - 1)
        )
        self.coarsest_block = MicroBlock3D(num_features, kernel_size=kernel_size)
        self.down_operators = nn.ModuleList(
            ScaledConv3d(num_features, num_features, kernel_size)
            for _ in range(num_scales - 1)
        )
        self.up_operators = nn.ModuleList(
            ScaledConv3d(num_features, num_features, kernel_size)
            for _ in range(num_scales - 1)
        )

    def _validate_states(
        self,
        states: Sequence[torch.Tensor | None],
    ) -> list[torch.Tensor | None]:
        if len(states) != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} scale states.")
        if states[0] is None:
            raise ValueError("The finest-scale state cannot be None.")
        for state in states:
            if state is not None and (state.ndim != 5 or state.shape[1] != self.num_features):
                raise ValueError(
                    f"Every state must be [B, {self.num_features}, Z, Y, X] or None."
                )
        return list(states)

    def forward_with_cache(
        self,
        states: Sequence[torch.Tensor | None],
    ) -> tuple[list[torch.Tensor], MacroBlockCache]:
        working = self._validate_states(states)
        input_present = tuple(state is not None for state in working)
        pre_outputs: list[torch.Tensor] = []
        pre_caches: list[MicroBlockCache] = []

        for scale in range(self.num_scales - 1):
            state = working[scale]
            assert state is not None
            pre_output, pre_cache = self.pre_blocks[scale].forward_with_cache(state)
            pre_outputs.append(pre_output)
            pre_caches.append(pre_cache)
            contribution = self.down_operators[scale](pre_output)
            existing = working[scale + 1]
            if existing is not None and existing.shape != contribution.shape:
                raise ValueError("Existing coarse state has an incompatible shape.")
            working[scale + 1] = contribution if existing is None else existing + contribution

        coarsest_input = working[-1]
        assert coarsest_input is not None
        coarsest, coarsest_cache = self.coarsest_block.forward_with_cache(coarsest_input)
        outputs = [state for state in working]
        outputs[-1] = coarsest
        post_caches: list[MicroBlockCache | None] = [None] * (self.num_scales - 1)

        for scale in range(self.num_scales - 2, -1, -1):
            up = self.up_operators[scale].adjoint(
                outputs[scale + 1],
                pre_outputs[scale].shape,
            )
            fused = pre_outputs[scale] + up
            outputs[scale], post_cache = self.post_blocks[scale].forward_with_cache(fused)
            post_caches[scale] = post_cache

        complete_outputs = [state for state in outputs if state is not None]
        assert len(complete_outputs) == self.num_scales
        cache = MacroBlockCache(
            input_present=input_present,
            output_states=tuple(complete_outputs),
            pre_outputs=tuple(pre_outputs),
            pre_caches=tuple(pre_caches),
            coarsest_cache=coarsest_cache,
            post_caches=tuple(cache for cache in post_caches if cache is not None),
        )
        return complete_outputs, cache

    def forward(
        self,
        states: Sequence[torch.Tensor | None] | torch.Tensor,
    ) -> list[torch.Tensor]:
        if isinstance(states, torch.Tensor):
            states = [states] + [None] * (self.num_scales - 1)
        outputs, _ = self.forward_with_cache(states)
        return outputs

    def jacobian_transpose(
        self,
        cache: MacroBlockCache,
        output_adjoints: Sequence[torch.Tensor | None],
    ) -> list[torch.Tensor | None]:
        """Apply the exact transpose Jacobian in reverse forward order."""

        if len(output_adjoints) != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} output adjoints.")
        bars = [
            torch.zeros_like(state) if adjoint is None else adjoint
            for state, adjoint in zip(cache.output_states, output_adjoints)
        ]
        pre_bars = [torch.zeros_like(value) for value in cache.pre_outputs]

        for scale in range(self.num_scales - 1):
            fused_bar = self.post_blocks[scale].jacobian_transpose_from_cache(
                cache.post_caches[scale],
                bars[scale],
            )
            pre_bars[scale] = pre_bars[scale] + fused_bar
            bars[scale + 1] = bars[scale + 1] + self.up_operators[scale](fused_bar)

        working_bars: list[torch.Tensor | None] = [None] * self.num_scales
        working_bars[-1] = self.coarsest_block.jacobian_transpose_from_cache(
            cache.coarsest_cache,
            bars[-1],
        )
        input_bars: list[torch.Tensor | None] = [None] * self.num_scales

        for scale in range(self.num_scales - 2, -1, -1):
            coarse_bar = working_bars[scale + 1]
            assert coarse_bar is not None
            if cache.input_present[scale + 1]:
                input_bars[scale + 1] = coarse_bar
            pre_bars[scale] = pre_bars[scale] + self.down_operators[scale].adjoint(
                coarse_bar,
                cache.pre_outputs[scale].shape,
            )
            working_bars[scale] = self.pre_blocks[scale].jacobian_transpose_from_cache(
                cache.pre_caches[scale],
                pre_bars[scale],
            )

        input_bars[0] = working_bars[0]
        return input_bars
