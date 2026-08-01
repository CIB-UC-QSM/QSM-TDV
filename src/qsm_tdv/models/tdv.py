"""A scalar, smooth Total Deep Variation energy in JAX.

This module deliberately contains no field-to-susceptibility prediction path.
For an NDHWC susceptibility input, it returns a local energy map and one scalar
energy per batch element.  Its force is obtained only by differentiating that
scalar energy with respect to susceptibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

Array = jax.Array
Params = dict[str, Any]


@dataclass(frozen=True)
class TDVConfig:
    """Architecture settings for a TDV^L regularizer.

    Each macro block has three resolutions and exactly five residual
    micro-blocks: one at the input scale, one at scale 2, one at scale 3, and
    one after each skip-connected upsampling path.
    """

    features: int = 12
    macro_blocks: int = 1
    kernel_size: int = 3
    nu: float = 0.1
    init_scale: float = 0.02

    def __post_init__(self) -> None:
        if self.features < 1 or self.macro_blocks < 1:
            raise ValueError("features and macro_blocks must be positive")
        if self.kernel_size < 1 or self.kernel_size % 2 != 1:
            raise ValueError("kernel_size must be a positive odd integer")
        if self.nu <= 0:
            raise ValueError("nu must be positive")


def student_t_activation(a: Array, nu: float) -> Array:
    """Smooth log-Student-t activation ``log(1 + nu*a²)/(2*nu)``."""

    return jnp.log1p(nu * a**2) / (2.0 * nu)


def student_t_prime(a: Array, nu: float) -> Array:
    """First derivative of :func:`student_t_activation`."""

    return a / (1.0 + nu * a**2)


def student_t_second(a: Array, nu: float) -> Array:
    """Second derivative of :func:`student_t_activation`."""

    return (1.0 - nu * a**2) / (1.0 + nu * a**2) ** 2


def _kernel(key: Array, kernel_size: int, in_channels: int, out_channels: int, scale: float) -> Array:
    shape = (kernel_size, kernel_size, kernel_size, in_channels, out_channels)
    fan_in = float(kernel_size**3 * in_channels)
    return jax.random.normal(key, shape, dtype=jnp.float32) * (scale / jnp.sqrt(fan_in))


def _conv(x: Array, kernel: Array, *, strides: tuple[int, int, int] = (1, 1, 1)) -> Array:
    """Bias-free NDHWC 3D convolution with zero convolution boundary padding."""

    return lax.conv_general_dilated(
        x,
        kernel,
        window_strides=strides,
        padding="SAME",
        dimension_numbers=("NDHWC", "DHWIO", "NDHWC"),
    )


def _micro_init(key: Array, channels: int, config: TDVConfig) -> Params:
    key_1, key_2 = jax.random.split(key)
    return {
        "k1": _kernel(key_1, config.kernel_size, channels, channels, config.init_scale),
        "k2": _kernel(key_2, config.kernel_size, channels, channels, config.init_scale),
    }


def _micro_apply(params: Params, x: Array, config: TDVConfig) -> Array:
    return x + _conv(student_t_activation(_conv(x, params["k1"]), config.nu), params["k2"])


def _macro_init(key: Array, channels: int, config: TDVConfig) -> Params:
    """Initialise a three-scale U-Net macro-block with five micro-blocks."""

    keys = jax.random.split(key, 9)
    c2, c4 = 2 * channels, 4 * channels
    return {
        "pre": _micro_init(keys[0], channels, config),
        "down_1": _kernel(keys[1], config.kernel_size, channels, c2, config.init_scale),
        "scale_2": _micro_init(keys[2], c2, config),
        "down_2": _kernel(keys[3], config.kernel_size, c2, c4, config.init_scale),
        "scale_3": _micro_init(keys[4], c4, config),
        "up_1": _kernel(keys[5], config.kernel_size, c4, c2, config.init_scale),
        "post_up_1": _micro_init(keys[6], c2, config),
        "up_0": _kernel(keys[7], config.kernel_size, c2, channels, config.init_scale),
        "post_up_0": _micro_init(keys[8], channels, config),
    }


def _upsample_to(x: Array, reference: Array) -> Array:
    """Nearest-neighbour upsampling and static crop for SAME-padded odd shapes."""

    output = jnp.repeat(jnp.repeat(jnp.repeat(x, 2, axis=1), 2, axis=2), 2, axis=3)
    return output[:, : reference.shape[1], : reference.shape[2], : reference.shape[3], :]


def _macro_apply(params: Params, x: Array, config: TDVConfig) -> Array:
    x0 = _micro_apply(params["pre"], x, config)
    x1 = _micro_apply(params["scale_2"], _conv(x0, params["down_1"], strides=(2, 2, 2)), config)
    x2 = _micro_apply(params["scale_3"], _conv(x1, params["down_2"], strides=(2, 2, 2)), config)
    y1 = _conv(_upsample_to(x2, x1), params["up_1"]) + x1
    y1 = _micro_apply(params["post_up_1"], y1, config)
    y0 = _conv(_upsample_to(y1, x0), params["up_0"]) + x0
    return _micro_apply(params["post_up_0"], y0, config)


def project_analysis_kernel(params: Params) -> Params:
    """Project every analysis output filter to zero mean, functionally.

    The ``D,H,W,input`` axes are averaged independently for every output
    channel, which is exactly the zero-mean condition imposed on ``K_a``.
    """

    analysis = params["analysis_kernel"]
    projected = analysis - jnp.mean(analysis, axis=(0, 1, 2, 3), keepdims=True)
    return {**params, "analysis_kernel": projected}


def analysis_filter_means(params: Params) -> Array:
    """Return one filter mean per analysis output channel for diagnostics."""

    return jnp.mean(params["analysis_kernel"], axis=(0, 1, 2, 3))


def init_tdv_parameters(key: Array, config: TDVConfig, *, input_channels: int = 1) -> Params:
    """Initialise and immediately project the learned TDV analysis operator."""

    if input_channels < 1:
        raise ValueError("input_channels must be positive")
    keys = jax.random.split(key, config.macro_blocks + 2)
    params: Params = {
        "analysis_kernel": _kernel(
            keys[0], config.kernel_size, input_channels, config.features, config.init_scale
        ),
        "macro_blocks": tuple(
            _macro_init(keys[index + 1], config.features, config)
            for index in range(config.macro_blocks)
        ),
        "readout_kernel": _kernel(keys[-1], 1, config.features, 1, config.init_scale),
    }
    return project_analysis_kernel(params)


def tdv_local_energy(params: Params, chi: Array, config: TDVConfig) -> Array:
    """Compute local energy density with shape ``[B,Z,Y,X,1]``."""

    if chi.ndim != 5:
        raise ValueError(f"TDV requires NDHWC input, got {chi.shape}")
    x = _conv(chi.astype(jnp.float32), params["analysis_kernel"])
    for block in params["macro_blocks"]:
        x = _macro_apply(block, x, config)
    return _conv(x, params["readout_kernel"])


def tdv_energy(
    params: Params,
    chi: Array,
    config: TDVConfig,
    mask: Array | None = None,
) -> Array:
    """Return one scalar TDV energy for each batch element.

    A mask is an explicit voxel integration weight, never an implicit part of
    the dipole operator.  It must have either one channel or the local-energy
    channel count (one).
    """

    local_energy = tdv_local_energy(params, chi, config)
    if mask is not None:
        if mask.shape != local_energy.shape:
            raise ValueError(
                f"TDV mask must match local energy shape {local_energy.shape}, got {mask.shape}"
            )
        local_energy = local_energy * mask.astype(local_energy.dtype)
    return jnp.sum(local_energy, axis=(1, 2, 3, 4))


def tdv_force(
    params: Params,
    chi: Array,
    config: TDVConfig,
    mask: Array | None = None,
) -> Array:
    """Evaluate ``∇_chi sum_b R_theta(chi_b)`` without detaching it."""

    def total_energy(image: Array) -> Array:
        return jnp.sum(tdv_energy(params, image, config, mask))

    return jax.grad(total_energy)(chi)


def tdv_hessian_vector_product(
    params: Params,
    chi: Array,
    vector: Array,
    config: TDVConfig,
    mask: Array | None = None,
) -> Array:
    """Compute ``∇²R(chi) @ vector`` by JVP; no Hessian is materialised."""

    force = lambda image: tdv_force(params, image, config, mask)
    _, hessian_vector = jax.jvp(force, (chi,), (vector,))
    return hessian_vector
