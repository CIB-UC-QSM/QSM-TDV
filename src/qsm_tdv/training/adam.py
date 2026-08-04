"""Small pure-JAX Adam implementation for TDV parameter pytrees."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array
PyTree = Any


class AdamState(NamedTuple):
    mean: PyTree
    variance: PyTree
    step: Array


def adam_init(parameters: PyTree) -> AdamState:
    return AdamState(
        mean=jax.tree.map(jnp.zeros_like, parameters),
        variance=jax.tree.map(jnp.zeros_like, parameters),
        step=jnp.asarray(0, dtype=jnp.int32),
    )


def adam_update(
    parameters: PyTree,
    gradients: PyTree,
    state: AdamState,
    *,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> tuple[PyTree, AdamState]:
    """Apply an Adam update without mutating a parameter pytree."""

    step = state.step + 1
    mean = jax.tree.map(lambda old, grad: beta1 * old + (1.0 - beta1) * grad, state.mean, gradients)
    variance = jax.tree.map(
        lambda old, grad: beta2 * old + (1.0 - beta2) * grad**2, state.variance, gradients
    )
    correction1 = 1.0 - beta1**step.astype(jnp.float32)
    correction2 = 1.0 - beta2**step.astype(jnp.float32)
    updated = jax.tree.map(
        lambda value, first, second: value
        - learning_rate * (first / correction1) / (jnp.sqrt(second / correction2) + epsilon),
        parameters,
        mean,
        variance,
    )
    return updated, AdamState(mean, variance, step)


def all_finite(tree: PyTree) -> Array:
    """Return whether every pytree leaf has only finite values."""

    leaves = jax.tree.leaves(tree)
    if not leaves:
        return jnp.asarray(True)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))


def global_norm(tree: PyTree) -> Array:
    """Overflow-safe Euclidean norm of a pytree.

    Squaring a finite float32 gradient can overflow before clipping. Scaling by
    the largest absolute leaf value avoids that failure mode. A non-finite leaf
    is reported as an infinite norm so callers can safely reject its update.
    """

    leaves = jax.tree.leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    finite = all_finite(tree)
    safe_leaves = [jnp.where(jnp.isfinite(leaf), leaf, jnp.zeros_like(leaf)) for leaf in leaves]
    max_abs = jnp.max(jnp.stack([jnp.max(jnp.abs(leaf)) for leaf in safe_leaves]))
    scale = jnp.where(max_abs > 0, max_abs, jnp.asarray(1.0, dtype=max_abs.dtype))
    scaled_squared = jnp.sum(
        jnp.stack([jnp.sum(jnp.square(leaf / scale)) for leaf in safe_leaves])
    )
    norm = max_abs * jnp.sqrt(scaled_squared)
    return jnp.where(finite, norm, jnp.asarray(jnp.inf, dtype=norm.dtype))


def clip_by_global_norm(
    gradients: PyTree,
    max_norm: float,
    *,
    epsilon: float = 1e-12,
) -> tuple[PyTree, Array, Array]:
    """Clip a gradient pytree to a maximum global Euclidean norm.

    The norm computation is overflow-safe. Non-finite leaves are replaced with
    zero in the clipped update, while the reported original norm is infinite.
    This prevents ``inf * 0`` from introducing NaNs into Adam's moments.
    """

    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    gradient_norm = global_norm(gradients)
    clip_scale = jnp.minimum(1.0, jnp.asarray(max_norm, dtype=gradient_norm.dtype) / (gradient_norm + epsilon))
    clipped = jax.tree.map(
        lambda gradient: jnp.where(jnp.isfinite(gradient), gradient * clip_scale, jnp.zeros_like(gradient)),
        gradients,
    )
    return clipped, gradient_norm, clip_scale


def clip_parameter_update(
    parameters: PyTree,
    proposed_parameters: PyTree,
    max_norm: float | None,
    *,
    epsilon: float = 1e-12,
) -> tuple[PyTree, Array, Array]:
    """Bound the global norm of an already-preconditioned parameter update.

    Adam's coordinate-wise normalization means clipping its *input gradients*
    does not, by itself, bound the resulting parameter displacement. This
    helper constrains that displacement after the Adam update while leaving the
    optimizer moments untouched. ``None`` preserves the legacy behaviour.
    """

    if max_norm is not None and max_norm <= 0:
        raise ValueError("max_norm must be positive when supplied")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    update = jax.tree.map(lambda proposed, current: proposed - current, proposed_parameters, parameters)
    update_norm = global_norm(update)
    if max_norm is None:
        return proposed_parameters, update_norm, jnp.asarray(1.0, dtype=update_norm.dtype)
    scale = jnp.minimum(1.0, jnp.asarray(max_norm, dtype=update_norm.dtype) / (update_norm + epsilon))
    bounded_parameters = jax.tree.map(
        lambda current, delta: current + scale * delta,
        parameters,
        update,
    )
    return bounded_parameters, update_norm, scale
