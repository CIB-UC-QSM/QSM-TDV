"""TDV nonlinearities."""

from __future__ import annotations

import torch


def log_student_t(x: torch.Tensor, nu: float = 9.0) -> torch.Tensor:
    """Smooth log-Student-t TDV activation."""

    if nu <= 0.0:
        raise ValueError("nu must be positive.")
    return torch.log1p(nu * x.square()) / (2.0 * nu)
