import jax
import jax.numpy as jnp

from qsm_tdv.models.tdv import TDVConfig, init_tdv_parameters, tdv_force
from qsm_tdv.physics.dipole import apply_dipole, dipole_kernel
from qsm_tdv.physics.reconstruction import (
    ReconstructionConfig,
    cg_residuals_within_tolerance,
    data_consistency,
    data_rhs,
    data_normal,
    fixed_cg,
    reconstruct_trajectory,
    semi_implicit_step,
)


def test_fixed_cg_agrees_with_dense_solve_and_rhs_gradient():
    matrix = jnp.asarray([[2.2, 0.3], [0.3, 1.4]], dtype=jnp.float32)
    rhs = jnp.asarray([[[[[0.4, -0.8]]]], [[[[1.1, 0.2]]]]], dtype=jnp.float32)

    def operator(x):
        return jnp.einsum("ij,bzyxj->bzyxi", matrix, x)

    solution, relative_residual = fixed_cg(operator, rhs, iterations=6)
    dense = jnp.linalg.solve(matrix, rhs[..., 0, 0, 0, :].T).T[:, None, None, None, :]
    assert float(jnp.max(jnp.abs(solution - dense))) < 2e-5
    assert float(jnp.max(relative_residual)) < 1e-5

    objective = lambda value: jnp.sum(fixed_cg(operator, value, iterations=6)[0] ** 2)
    direction = jnp.asarray([[[[[0.2, -0.1]]]], [[[[0.05, 0.1]]]]], dtype=jnp.float32)
    gradient = jax.grad(objective)(rhs)
    epsilon = 1e-3
    finite_difference = (objective(rhs + epsilon * direction) - objective(rhs - epsilon * direction)) / (2 * epsilon)
    autodiff_directional = jnp.sum(gradient * direction)
    assert float(jnp.abs(finite_difference - autodiff_directional)) < 2e-4


def test_semi_implicit_equation_residual_is_small():
    shape = (8, 8, 8)
    key_params, key_chi = jax.random.split(jax.random.key(8))
    tdv_config = TDVConfig(features=1)
    reconstruction_config = ReconstructionConfig(
        steps=1, cg_iterations=20, max_time=0.2, cg_relative_tolerance=2e-5
    )
    params = init_tdv_parameters(key_params, tdv_config)
    chi = jax.random.normal(key_chi, (1, *shape, 1)) * 0.01
    regularizer_mask = jnp.zeros_like(chi).at[:, :6, :, :, :].set(1.0)
    kernel = dipole_kernel(shape, (0.8, 1.1, 1.4), (0.1, 0.4, 0.9))
    local_field = apply_dipole(chi, kernel)
    tau = jnp.asarray(0.03, dtype=jnp.float32)
    next_chi, diagnostics = semi_implicit_step(
        params,
        chi,
        local_field,
        kernel,
        tdv_config,
        reconstruction_config,
        tau,
        regularizer_mask=regularizer_mask,
    )
    rhs = chi + tau * (
        data_rhs(local_field, kernel) - tdv_force(params, chi, tdv_config, regularizer_mask)
    )
    residual = next_chi + tau * data_normal(next_chi, kernel) - rhs
    relative = jnp.linalg.norm(residual) / (jnp.linalg.norm(rhs) + 1e-8)
    assert float(relative) < 2e-5
    assert bool(cg_residuals_within_tolerance(diagnostics, reconstruction_config))


def test_data_consistency_is_the_weighted_residual_norm():
    shape = (4, 4, 4)
    chi = jnp.arange(64, dtype=jnp.float32).reshape(1, *shape, 1) / 100.0
    local_field = jnp.full_like(chi, 0.02)
    mask = jnp.zeros_like(chi).at[:, :2, :, :, :].set(1.0)
    magnitude = jnp.linspace(0.25, 1.0, chi.size, dtype=jnp.float32).reshape(chi.shape)
    weight = mask * magnitude
    kernel = dipole_kernel(shape)

    expected = jnp.linalg.norm((weight * (apply_dipole(chi, kernel) - local_field)).reshape(1, -1), axis=1)
    actual = data_consistency(chi, local_field, kernel, statistical_weight=weight)

    assert float(jnp.max(jnp.abs(actual - expected))) < 1e-7


def test_reconstruction_trajectory_retains_initial_and_each_solver_state():
    shape = (8, 8, 8)
    key_params, key_chi = jax.random.split(jax.random.key(19))
    tdv_config = TDVConfig(features=1)
    reconstruction_config = ReconstructionConfig(steps=2, cg_iterations=8, max_time=0.2)
    params = init_tdv_parameters(key_params, tdv_config)
    chi_init = jax.random.normal(key_chi, (1, *shape, 1), dtype=jnp.float32) * 0.01
    kernel = dipole_kernel(shape)
    local_field = apply_dipole(chi_init, kernel)
    trajectory, diagnostics = reconstruct_trajectory(
        params,
        jnp.asarray(0.0, dtype=jnp.float32),
        chi_init,
        local_field,
        kernel,
        tdv_config,
        reconstruction_config,
    )
    assert trajectory.shape == (reconstruction_config.steps + 1, 1, *shape, 1)
    assert float(jnp.max(jnp.abs(trajectory[0] - chi_init))) == 0.0
    assert diagnostics.relative_residuals.shape == (reconstruction_config.steps, 1)
