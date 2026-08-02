import jax
import jax.numpy as jnp

from qsm_tdv.data.contract import load_single_sample
from qsm_tdv.data.synthetic import create_synthetic_sample
from qsm_tdv.models.tdv import TDVConfig, analysis_filter_means
from qsm_tdv.physics.reconstruction import ReconstructionConfig
from qsm_tdv.training.metrics import nrmse
from qsm_tdv.training.overfit import OverfitConfig, train_single_sample


def test_tiny_synthetic_sample_overfits_with_finite_time_and_zero_mean_kernel(tmp_path):
    dataset_path, _ = create_synthetic_sample(tmp_path / "tiny.npz", shape_zyx=(8, 8, 8), noise_std=0.001)
    sample = load_single_sample(dataset_path)
    result = train_single_sample(
        sample,
        TDVConfig(features=1),
        ReconstructionConfig(steps=1, cg_iterations=8, max_time=0.25),
        OverfitConfig(iterations=8, learning_rate=1e-3, log_every=8),
    )
    assert float(result.validation_mse) < float(result.baseline_mse)
    assert bool(jnp.isfinite(result.stopping_time))
    assert float(jnp.max(jnp.abs(analysis_filter_means(result.parameters["tdv"])))) < 2e-7
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(result.parameters))
    metrics = result.history[-1]
    for name in (
        "analysis_gradient_norm",
        "macro_blocks_gradient_norm",
        "readout_gradient_norm",
        "raw_time_gradient",
        "gradient_clip_scale",
        "clipped_gradient_norm",
    ):
        assert jnp.isfinite(metrics[name])
        assert abs(metrics[name]) > 0.0
    assert metrics["clipped_gradient_norm"] <= 1.0 + 1e-6


def test_nrmse_matches_its_documented_l2_definition():
    estimate = jnp.asarray([[[[[3.0], [1.0]]]]])
    ground_truth = jnp.asarray([[[[[1.0], [2.0]]]]])
    expected = jnp.linalg.norm((estimate - ground_truth).reshape(-1)) / jnp.linalg.norm(ground_truth.reshape(-1))
    assert float(jnp.abs(nrmse(estimate, ground_truth) - expected)) < 1e-7
