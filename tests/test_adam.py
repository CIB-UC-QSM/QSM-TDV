import jax
import jax.numpy as jnp
import pytest

from qsm_tdv.training.adam import all_finite, clip_by_global_norm, clip_parameter_update, global_norm


def test_clip_by_global_norm_scales_the_entire_parameter_pytree():
    gradients = {"first": jnp.asarray([3.0, 4.0]), "second": jnp.asarray([12.0])}

    clipped, original_norm, scale = clip_by_global_norm(gradients, max_norm=1.0)

    assert float(original_norm) == pytest.approx(13.0)
    assert float(scale) == pytest.approx(1.0 / 13.0)
    assert float(global_norm(clipped)) == pytest.approx(1.0)
    assert jnp.array_equal(gradients["first"], jnp.asarray([3.0, 4.0]))
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(clipped))


def test_clip_by_global_norm_leaves_small_gradients_unchanged():
    gradients = {"value": jnp.asarray([0.3, 0.4])}

    clipped, original_norm, scale = clip_by_global_norm(gradients, max_norm=1.0)

    assert float(original_norm) == pytest.approx(0.5)
    assert float(scale) == pytest.approx(1.0)
    assert jnp.array_equal(clipped["value"], gradients["value"])


@pytest.mark.parametrize("max_norm", (0.0, -1.0))
def test_clip_by_global_norm_requires_a_positive_threshold(max_norm):
    with pytest.raises(ValueError, match="max_norm must be positive"):
        clip_by_global_norm({"value": jnp.asarray([1.0])}, max_norm=max_norm)


def test_clip_parameter_update_bounds_adam_displacement_without_mutating_inputs():
    parameters = {"value": jnp.asarray([1.0, -1.0])}
    proposed = {"value": jnp.asarray([4.0, 3.0])}

    bounded, update_norm, scale = clip_parameter_update(parameters, proposed, max_norm=2.5)

    assert float(update_norm) == pytest.approx(5.0)
    assert float(scale) == pytest.approx(0.5)
    actual_update = jax.tree.map(lambda new, old: new - old, bounded, parameters)
    assert float(global_norm(actual_update)) == pytest.approx(2.5)
    assert jnp.array_equal(parameters["value"], jnp.asarray([1.0, -1.0]))
    assert jnp.array_equal(proposed["value"], jnp.asarray([4.0, 3.0]))


def test_clip_parameter_update_preserves_unbounded_legacy_mode():
    parameters = {"value": jnp.asarray([1.0])}
    proposed = {"value": jnp.asarray([3.0])}

    bounded, update_norm, scale = clip_parameter_update(parameters, proposed, max_norm=None)

    assert float(update_norm) == pytest.approx(2.0)
    assert float(scale) == pytest.approx(1.0)
    assert jnp.array_equal(bounded["value"], proposed["value"])



def test_global_norm_avoids_float32_overflow_for_finite_gradients():
    gradients = {"value": jnp.asarray([3e30, 4e30], dtype=jnp.float32)}

    clipped, norm, scale = clip_by_global_norm(gradients, max_norm=1.0)

    assert bool(jnp.isfinite(norm))
    assert float(norm) == pytest.approx(5e30, rel=1e-5)
    assert float(scale) == pytest.approx(2e-31, rel=1e-5)
    assert float(global_norm(clipped)) == pytest.approx(1.0, rel=1e-5)


def test_clip_by_global_norm_neutralizes_nonfinite_gradient_leaves():
    gradients = {"value": jnp.asarray([jnp.inf, jnp.nan, 3.0])}

    clipped, norm, scale = clip_by_global_norm(gradients, max_norm=1.0)

    assert bool(jnp.isinf(norm))
    assert float(scale) == pytest.approx(0.0)
    assert bool(all_finite(clipped))
    assert jnp.array_equal(clipped["value"], jnp.zeros(3))
