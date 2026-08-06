"""CUDA-aware tensor validation helpers."""

from __future__ import annotations

import torch


def _raise_if_false(
    condition: torch.Tensor,
    message: str,
    *,
    error_type: type[Exception] = ValueError,
) -> None:
    """Raise without synchronizing a CUDA stream on successful checks."""

    if condition.is_cuda:
        torch._assert_async(condition, message)
        return
    if not bool(condition):
        raise error_type(message)


def require_finite(
    value: torch.Tensor,
    message: str,
    *,
    error_type: type[Exception] = ValueError,
) -> None:
    _raise_if_false(
        torch.isfinite(value).all(),
        message,
        error_type=error_type,
    )


def require_nonnegative(
    value: torch.Tensor,
    message: str,
    *,
    error_type: type[Exception] = ValueError,
) -> None:
    _raise_if_false(
        torch.all(value >= 0),
        message,
        error_type=error_type,
    )
