import jax
import jax.numpy as jnp
import pytest

from qsm_tdv.physics.dipole import apply_dipole, dipole_adjoint, dipole_kernel


@pytest.mark.parametrize(
    ("shape", "voxel_size", "b0"),
    [
        ((8, 7, 6), (0.8, 1.2, 2.1), (0.2, -0.3, 0.9)),
        ((7, 6, 5), (1.0, 0.9, 1.5), (-0.4, 0.5, 0.2)),
    ],
)
def test_dipole_adjoint_for_anisotropic_arbitrary_b0(shape, voxel_size, b0):
    key_x, key_y = jax.random.split(jax.random.key(4))
    x = jax.random.normal(key_x, (2, *shape, 1))
    y = jax.random.normal(key_y, (2, *shape, 1))
    kernel = dipole_kernel(shape, voxel_size, b0)
    lhs = jnp.sum(apply_dipole(x, kernel) * y)
    rhs = jnp.sum(x * dipole_adjoint(y, kernel))
    relative_error = jnp.abs(lhs - rhs) / (jnp.abs(lhs) + jnp.abs(rhs) + 1e-8)
    assert float(relative_error) < 1e-5


def test_dipole_dc_is_zero_and_kernel_has_no_nans():
    kernel = dipole_kernel((7, 8, 9), (0.7, 1.1, 2.0), (1.0, 2.0, -1.0))
    assert float(kernel[0, 0, 0]) == 0.0
    assert bool(jnp.all(jnp.isfinite(kernel)))
