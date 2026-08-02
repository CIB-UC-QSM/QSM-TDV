import jax
import jax.numpy as jnp
import pytest

from qsm_tdv.training.adam import clip_by_global_norm, global_norm


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
