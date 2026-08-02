"""Single-sample TDV-QSM training and same-sample overfit validation.

This module is intentionally a diagnostic, not a claim of generalisation: the
validation pass uses the training sample to verify that all higher-order
gradients, shared TDV steps, CG solves, and the stopping time can optimize.
Future subject-level training must provide disjoint subjects to the analogous
multi-sample pipeline.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from qsm_tdv.data.contract import QSMSample, effective_data_weight, load_single_sample
from qsm_tdv.evaluation.convergence import save_convergence_artifacts
from qsm_tdv.models.tdv import TDVConfig, init_tdv_parameters, project_analysis_kernel
from qsm_tdv.physics.dipole import dipole_kernel
from qsm_tdv.physics.reconstruction import (
    ReconstructionConfig,
    data_consistency,
    reconstruct,
    require_cg_residuals_within_tolerance,
)
from qsm_tdv.training.adam import AdamState, adam_init, adam_update, clip_by_global_norm, global_norm
from qsm_tdv.training.metrics import masked_mse, nrmse

Array = jax.Array


@dataclass(frozen=True)
class OverfitConfig:
    iterations: int = 100
    learning_rate: float = 1e-3
    max_gradient_norm: float = 1.0
    data_consistency_weight: float = 0.0
    seed: int = 0
    log_every: int = 10
    supervised_metric: str = "mse"
    epoch_chunk_size: int = 2
    include_magnitude_in_weight: bool = True

    def __post_init__(self) -> None:
        if self.iterations < 1 or self.log_every < 1 or self.epoch_chunk_size < 1:
            raise ValueError("iterations, log_every, and epoch_chunk_size must be positive")
        if self.learning_rate <= 0 or self.max_gradient_norm <= 0 or self.data_consistency_weight < 0:
            raise ValueError(
                "learning_rate and max_gradient_norm must be positive and data consistency weight non-negative"
            )
        if self.supervised_metric not in {"mse", "nrmse"}:
            raise ValueError("supervised_metric must be either 'mse' or 'nrmse'")


class TrainingResult(NamedTuple):
    parameters: dict[str, Any]
    optimizer_state: AdamState
    reconstruction: Array
    baseline_mse: Array
    validation_mse: Array
    baseline_nrmse: Array
    validation_nrmse: Array
    validation_data_consistency: Array
    stopping_time: Array
    history: tuple[dict[str, float], ...]


def _single_sample_batch(sample: QSMSample) -> dict[str, Array | None]:
    batch = sample.as_batch()
    return {name: value for name, value in batch.items()}


def train_single_sample(
    sample: QSMSample,
    tdv_config: TDVConfig,
    reconstruction_config: ReconstructionConfig,
    overfit_config: OverfitConfig,
    progress_callback: Callable[[dict[str, float]], None] | None = None,
    *,
    initial_parameters: dict[str, Any] | None = None,
    initial_optimizer_state: AdamState | None = None,
    epoch_offset: int = 0,
) -> TrainingResult:
    """Optimize TDV parameters and T on one sample, then validate on it.

    The returned validation measures optimisation reachability only.  It is
    explicitly not a held-out estimate because this is a single-dataset
    overfitting test.
    """

    batch = _single_sample_batch(sample)
    local_field = batch["local_field"]
    reference = batch["susceptibility"]
    brain_mask = batch["brain_mask"]
    if local_field is None or reference is None or brain_mask is None:
        raise RuntimeError("Validated QSMSample unexpectedly has a missing required array")
    reference_mask = batch["reference_mask"] if batch["reference_mask"] is not None else brain_mask
    data_weight = effective_data_weight(
        brain_mask,
        batch["magnitude"],
        include_magnitude=overfit_config.include_magnitude_in_weight,
    )
    chi_init = data_weight * local_field
    kernel = dipole_kernel(
        tuple(local_field.shape[1:4]),
        sample.metadata.voxel_size_zyx,
        sample.metadata.b0_direction_zyx,
    )
    if (initial_parameters is None) != (initial_optimizer_state is None):
        raise ValueError("initial_parameters and initial_optimizer_state must be supplied together")
    if epoch_offset < 0:
        raise ValueError("epoch_offset must be non-negative")
    if initial_parameters is None:
        initial_parameters = {
            "tdv": init_tdv_parameters(jax.random.key(overfit_config.seed), tdv_config),
            "raw_time": jnp.asarray(0.0, dtype=jnp.float32),
        }
        optimizer_state = adam_init(initial_parameters)
    else:
        optimizer_state = initial_optimizer_state

    def objective(parameters: dict[str, Any]) -> tuple[Array, dict[str, Array]]:
        reconstruction, diagnostics = reconstruct(
            parameters["tdv"],
            parameters["raw_time"],
            chi_init,
            local_field,
            kernel,
            tdv_config,
            reconstruction_config,
            regularizer_mask=brain_mask,
            statistical_weight=data_weight,
        )
        terminal_mse = masked_mse(reconstruction, reference, reference_mask)
        terminal_nrmse = nrmse(reconstruction, reference, reference_mask)
        field_consistency = jnp.mean(
            data_consistency(
                reconstruction,
                local_field,
                kernel,
                statistical_weight=data_weight,
            )
        )
        supervised_error = terminal_mse if overfit_config.supervised_metric == "mse" else terminal_nrmse
        loss = supervised_error + overfit_config.data_consistency_weight * field_consistency
        metrics = {
            "loss": loss,
            "terminal_mse": terminal_mse,
            "terminal_nrmse": terminal_nrmse,
            "data_consistency": field_consistency,
            "time": diagnostics.time,
            "tau": diagnostics.tau,
            "max_cg_relative_residual": jnp.max(diagnostics.relative_residuals),
            "max_force_norm": jnp.max(diagnostics.force_norms),
        }
        return loss, metrics

    def train_step(parameters: dict[str, Any], state: AdamState) -> tuple[dict[str, Any], AdamState, dict[str, Array]]:
        (loss, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(parameters)
        clipped_gradients, gradient_norm, clip_scale = clip_by_global_norm(
            gradients,
            overfit_config.max_gradient_norm,
        )
        updated_parameters, updated_state = adam_update(
            parameters,
            clipped_gradients,
            state,
            learning_rate=overfit_config.learning_rate,
        )
        updated_parameters = {
            **updated_parameters,
            "tdv": project_analysis_kernel(updated_parameters["tdv"]),
        }
        metrics = {
            **metrics,
            "gradient_norm": gradient_norm,
            "clipped_gradient_norm": global_norm(clipped_gradients),
            "gradient_clip_scale": clip_scale,
            "analysis_gradient_norm": global_norm(gradients["tdv"]["analysis_kernel"]),
            "macro_blocks_gradient_norm": global_norm(gradients["tdv"]["macro_blocks"]),
            "readout_gradient_norm": global_norm(gradients["tdv"]["readout_kernel"]),
            "raw_time_gradient": gradients["raw_time"],
            "loss": loss,
        }
        return updated_parameters, updated_state, metrics

    evaluate = jax.jit(objective)

    def build_train_chunk(
        chunk_length: int,
        parameters: dict[str, Any], state: AdamState
    ) -> tuple[dict[str, Any], AdamState, dict[str, Array]]:
        """Build a static, bounded scan of full-volume updates."""

        def one_update(
            carry: tuple[dict[str, Any], AdamState], _: None
        ) -> tuple[tuple[dict[str, Any], AdamState], dict[str, Array]]:
            current_parameters, current_state = carry
            next_parameters, next_state, metrics = train_step(current_parameters, current_state)
            return (next_parameters, next_state), metrics

        (final_parameters, final_state), metric_history = jax.lax.scan(
            one_update,
            (parameters, state),
            xs=None,
            length=chunk_length,
        )
        return final_parameters, final_state, metric_history

    baseline_metrics = evaluate(initial_parameters)[1]
    baseline_mse = baseline_metrics["terminal_mse"]
    baseline_nrmse = baseline_metrics["terminal_nrmse"]
    parameters = initial_parameters
    history: list[dict[str, float]] = []
    chunk_runners: dict[int, Any] = {}
    for first_epoch in range(0, overfit_config.iterations, overfit_config.epoch_chunk_size):
        chunk_length = min(overfit_config.epoch_chunk_size, overfit_config.iterations - first_epoch)
        if chunk_length not in chunk_runners:
            chunk_runners[chunk_length] = jax.jit(
                lambda current_parameters, current_state, length=chunk_length: build_train_chunk(
                    length, current_parameters, current_state
                ),
                donate_argnums=(0, 1),
            )
        parameters, optimizer_state, chunk_metrics = chunk_runners[chunk_length](parameters, optimizer_state)
        # Synchronizing once per bounded chunk prevents a full-volume
        # higher-order buffer queue from accumulating on accelerator backends.
        parameters, optimizer_state, chunk_metrics = jax.block_until_ready(
            (parameters, optimizer_state, chunk_metrics)
        )
        for local_epoch in range(chunk_length):
            iteration = epoch_offset + first_epoch + local_epoch + 1
            if iteration == 1 or iteration % overfit_config.log_every == 0 or iteration == overfit_config.iterations:
                report = {"iteration": float(iteration)}
                report.update({name: float(value[local_epoch]) for name, value in chunk_metrics.items()})
                history.append(report)
                if progress_callback is not None:
                    progress_callback(report)

    (_, validation_metrics) = evaluate(parameters)
    reconstruction, diagnostics = reconstruct(
        parameters["tdv"],
        parameters["raw_time"],
        chi_init,
        local_field,
        kernel,
        tdv_config,
        reconstruction_config,
        regularizer_mask=brain_mask,
        statistical_weight=data_weight,
    )
    require_cg_residuals_within_tolerance(diagnostics, reconstruction_config)
    return TrainingResult(
        parameters=parameters,
        optimizer_state=optimizer_state,
        reconstruction=reconstruction,
        baseline_mse=baseline_mse,
        validation_mse=validation_metrics["terminal_mse"],
        baseline_nrmse=baseline_nrmse,
        validation_nrmse=validation_metrics["terminal_nrmse"],
        validation_data_consistency=validation_metrics["data_consistency"],
        stopping_time=diagnostics.time,
        history=tuple(history),
    )


def save_training_result(
    result: TrainingResult,
    sample: QSMSample,
    tdv_config: TDVConfig,
    reconstruction_config: ReconstructionConfig,
    overfit_config: OverfitConfig,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "reconstruction.npy", np.asarray(result.reconstruction[0]))
    with (output_dir / "checkpoint.pkl").open("wb") as handle:
        pickle.dump(
            {
                "parameters": jax.device_get(result.parameters),
                "optimizer_state": jax.device_get(result.optimizer_state),
                "tdv_config": asdict(tdv_config),
                "reconstruction_config": asdict(reconstruction_config),
                "metadata": asdict(sample.metadata),
            },
            handle,
        )
    convergence_csv, convergence_figure = save_convergence_artifacts(result.history, output_dir)
    report = {
        "validation_mode": "same_sample_overfit_diagnostic",
        "baseline_mse": float(result.baseline_mse),
        "validation_mse": float(result.validation_mse),
        "baseline_nrmse": float(result.baseline_nrmse),
        "validation_nrmse": float(result.validation_nrmse),
        "validation_data_consistency": float(result.validation_data_consistency),
        "stopping_time": float(result.stopping_time),
        "sample_manifest": sample.manifest,
        "tdv_config": asdict(tdv_config),
        "reconstruction_config": asdict(reconstruction_config),
        "overfit_config": asdict(overfit_config),
        "history": list(result.history),
        "convergence_csv": str(convergence_csv),
        "convergence_figure": str(convergence_figure),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and validate TDV-QSM on one sample as an overfit diagnostic.")
    parser.add_argument("--dataset", type=Path, required=True, help="Single-sample .npz conforming to qsm_tdv.data.contract")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/overfit"))
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--max-gradient-norm",
        type=float,
        default=1.0,
        help="Global gradient-norm clipping threshold applied before Adam (default: 1.0)",
    )
    parser.add_argument("--data-consistency-weight", type=float, default=0.0)
    parser.add_argument("--features", type=int, default=4)
    parser.add_argument("--macro-blocks", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--cg-iterations", type=int, default=6)
    parser.add_argument("--max-time", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--supervised-metric", choices=("mse", "nrmse"), default="mse")
    parser.add_argument("--epoch-chunk-size", type=int, default=2)
    parser.add_argument("--remat-force", action="store_true")
    parser.add_argument(
        "--include-magnitude-in-weight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use W = brain_mask * magnitude when the sample provides magnitude (default: enabled)",
    )
    arguments = parser.parse_args()

    sample = load_single_sample(arguments.dataset)
    tdv_config = TDVConfig(features=arguments.features, macro_blocks=arguments.macro_blocks)
    reconstruction_config = ReconstructionConfig(
        steps=arguments.steps,
        cg_iterations=arguments.cg_iterations,
        max_time=arguments.max_time,
        remat_force=arguments.remat_force,
    )
    overfit_config = OverfitConfig(
        iterations=arguments.iterations,
        learning_rate=arguments.learning_rate,
        max_gradient_norm=arguments.max_gradient_norm,
        data_consistency_weight=arguments.data_consistency_weight,
        seed=arguments.seed,
        log_every=arguments.log_every,
        supervised_metric=arguments.supervised_metric,
        epoch_chunk_size=arguments.epoch_chunk_size,
        include_magnitude_in_weight=arguments.include_magnitude_in_weight,
    )
    result = train_single_sample(sample, tdv_config, reconstruction_config, overfit_config)
    save_training_result(result, sample, tdv_config, reconstruction_config, overfit_config, arguments.output_dir)
    print(f"Baseline MSE:   {float(result.baseline_mse):.6e}")
    print(f"Overfit MSE:    {float(result.validation_mse):.6e}")
    print(f"Baseline NRMSE: {float(result.baseline_nrmse):.6e}")
    print(f"Overfit NRMSE:  {float(result.validation_nrmse):.6e}")
    print(f"Stopping time:  {float(result.stopping_time):.6e}")
    print(f"Wrote {arguments.output_dir}")


if __name__ == "__main__":
    main()
