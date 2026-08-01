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


def global_norm(tree: PyTree) -> Array:
    """Euclidean norm of a pytree, useful for finite-gradient checks."""

    squared = [jnp.sum(jnp.square(leaf)) for leaf in jax.tree.leaves(tree)]
    return jnp.sqrt(jnp.sum(jnp.stack(squared)))
