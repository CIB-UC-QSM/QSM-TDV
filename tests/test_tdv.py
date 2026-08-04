import jax
import jax.numpy as jnp

from qsm_tdv.models.tdv import (
    TDVConfig,
    analysis_filter_means,
    init_tdv_parameters,
    project_analysis_kernel,
    tdv_energy,
    tdv_force,
    tdv_hessian_vector_product,
)


def test_tdv_energy_force_and_directional_finite_difference():
    key_params, key_x, key_direction = jax.random.split(jax.random.key(1), 3)
    config = TDVConfig(features=2, macro_blocks=1)
    params = init_tdv_parameters(key_params, config)
    x = jax.random.normal(key_x, (2, 8, 8, 8, 1))
    direction = jax.random.normal(key_direction, x.shape)
    direction /= jnp.sqrt(jnp.sum(direction**2))
    energy = tdv_energy(params, x, config)
    force = tdv_force(params, x, config)
    hessian_vector = tdv_hessian_vector_product(params, x, direction, config)
    assert energy.shape == (2,)
    assert force.shape == x.shape
    assert hessian_vector.shape == x.shape

    objective = lambda image: jnp.sum(tdv_energy(params, image, config))
    epsilon = 2e-3
    finite_difference = (objective(x + epsilon * direction) - objective(x - epsilon * direction)) / (2.0 * epsilon)
    autodiff_directional = jnp.sum(force * direction)
    assert float(jnp.abs(finite_difference - autodiff_directional)) < 2e-4


def test_tdv_mask_is_applied_to_chi_before_the_energy_network():
    key_params, key_x, key_direction = jax.random.split(jax.random.key(11), 3)
    config = TDVConfig(features=2, macro_blocks=1)
    params = init_tdv_parameters(key_params, config)
    chi = jax.random.normal(key_x, (1, 8, 8, 8, 1), dtype=jnp.float32)
    direction = jax.random.normal(key_direction, chi.shape, dtype=jnp.float32)
    mask = jnp.zeros_like(chi).at[:, :5, 1:7, :6, :].set(1.0)

    masked_energy = tdv_energy(params, chi, config, mask)
    masked_force = tdv_force(params, chi, config, mask)
    masked_hessian_vector = tdv_hessian_vector_product(params, chi, direction, config, mask)

    expected_energy = tdv_energy(params, mask * chi, config)
    expected_force = mask * tdv_force(params, mask * chi, config)
    expected_hessian_vector = mask * tdv_hessian_vector_product(
        params, mask * chi, mask * direction, config
    )
    assert float(jnp.max(jnp.abs(masked_energy - expected_energy))) < 1e-6
    assert float(jnp.max(jnp.abs(masked_force - expected_force))) < 1e-6
    assert float(jnp.max(jnp.abs(masked_hessian_vector - expected_hessian_vector))) < 1e-6
    assert float(jnp.max(jnp.abs((1.0 - mask) * masked_force))) == 0.0

    objective = lambda image: jnp.sum(tdv_energy(params, image, config, mask))
    epsilon = 2e-3
    finite_difference = (objective(chi + epsilon * direction) - objective(chi - epsilon * direction)) / (2.0 * epsilon)
    autodiff_directional = jnp.sum(masked_force * direction)
    assert float(jnp.abs(finite_difference - autodiff_directional)) < 2e-4


def test_analysis_projection_is_zero_mean_per_output_filter():
    config = TDVConfig(features=3)
    params = init_tdv_parameters(jax.random.key(5), config)
    shifted = {**params, "analysis_kernel": params["analysis_kernel"] + 0.7}
    projected = project_analysis_kernel(shifted)
    assert float(jnp.max(jnp.abs(analysis_filter_means(projected)))) < 2e-7
