"""Differentiable reconstruction metrics."""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array


def masked_mse(estimate: Array, ground_truth: Array, mask: Array) -> Array:
    """Mean squared error over an explicit non-negative support mask."""

    return jnp.sum(mask * (estimate - ground_truth) ** 2) / jnp.maximum(jnp.sum(mask), 1.0)


def nrmse(
    estimate: Array,
    ground_truth: Array,
    mask: Array | None = None,
    *,
    epsilon: float = 1e-12,
) -> Array:
    """Compute ``||x_true - x_gt||₂ / ||x_gt||₂``.

    When a mask is supplied it explicitly defines the support of both norms;
    this excludes air/no-signal voxels without changing the underlying QSM
    forward operator.
    """

    difference = estimate - ground_truth
    target = ground_truth
    if mask is not None:
        difference = difference * mask
        target = target * mask
    return jnp.linalg.norm(difference.reshape(-1)) / jnp.maximum(jnp.linalg.norm(target.reshape(-1)), epsilon)
