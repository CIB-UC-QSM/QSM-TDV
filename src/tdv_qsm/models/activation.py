"""Source-faithful smooth TDV nonlinearity."""

from __future__ import annotations

import torch


def student_t_pair(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``phi(x)`` and ``phi'(x)`` for the source's ``alpha=1`` choice.

    Both quantities are intentionally evaluated in float32, including inside
    CUDA autocast.  Callers cast the value only when a following convolution
    requires another dtype.
    """

    x32 = x.float()
    denominator = 1.0 + x32.square()
    value = 0.5 * torch.log(denominator)
    derivative = x32 / denominator
    return value, derivative


def log_student_t(x: torch.Tensor) -> torch.Tensor:
    """Compatibility name for the source-faithful Student-t value."""

    value, _ = student_t_pair(x)
    return value
