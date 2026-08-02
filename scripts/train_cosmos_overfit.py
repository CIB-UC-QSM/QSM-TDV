#!/usr/bin/env python3
"""Run the complete noisy-signal COSMOS TDV-QSM overfit diagnostic."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import jax
from qsm_tdv.data.contract import effective_data_weight, load_single_sample
from qsm_tdv.data.cosmos import CosmosSimulationConfig, prepare_cosmos_overfit_sample
from qsm_tdv.evaluation.slices import save_orthogonal_reconstruction_slices
from qsm_tdv.models.tdv import TDVConfig
from qsm_tdv.physics.reconstruction import ReconstructionConfig
from qsm_tdv.training.overfit import OverfitConfig, save_training_result, train_single_sample


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate noisy complex COSMOS data and overfit the energy-based TDV-QSM reconstruction."
    )
    parser.add_argument("--cosmos-data", type=Path, default=Path("/cosmos_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/cosmos-snr70-100epochs"))
    parser.add_argument("--snr", type=float, default=70.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--voxel-size", type=float, nargs=3, default=(1.0, 1.0, 1.0), metavar=("VZ", "VY", "VX"))
    parser.add_argument("--b0-direction", type=float, nargs=3, default=(0.0, 0.0, 1.0), metavar=("BZ", "BY", "BX"))
    parser.add_argument("--phase-scale", type=float, default=1.0, help="Radians per source susceptibility unit")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--total-epochs", type=int, default=100, help="Total target used in progress output when resuming")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--max-gradient-norm",
        type=float,
        default=1.0,
        help="Global gradient-norm clipping threshold applied before Adam (default: 1.0)",
    )
    parser.add_argument("--data-consistency-weight", type=float, default=0.0)
    parser.add_argument("--features", type=int, default=1)
    parser.add_argument("--macro-blocks", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--cg-iterations", type=int, default=12)
    parser.add_argument("--max-time", type=float, default=0.25)
    parser.add_argument("--log-every", type=int, default=1, help="Persist every epoch in convergence.csv")
    parser.add_argument("--epoch-chunk-size", type=int, default=2, help="Static JAX updates per synchronized chunk")
    parser.add_argument("--remat-force", action="store_true")
    parser.add_argument(
        "--include-magnitude-in-weight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use W = mask * magnitude in the COSMOS data term (default: enabled)",
    )
    parser.add_argument("--resume", action="store_true", help="Resume TDV parameters and Adam state from output-dir/checkpoint.pkl")
    arguments = parser.parse_args()

    prepared_data_path = arguments.output_dir / "prepared_data" / f"cosmos_snr{arguments.snr:g}.npz"
    checkpoint_path = arguments.output_dir / "checkpoint.pkl"
    report_path = arguments.output_dir / "report.json"
    if arguments.resume:
        if not prepared_data_path.is_file() or not checkpoint_path.is_file() or not report_path.is_file():
            raise FileNotFoundError("--resume requires prepared data, checkpoint.pkl, and report.json in output-dir")
        dataset_path = prepared_data_path
        manifest_path = prepared_data_path.with_suffix(".json")
        with checkpoint_path.open("rb") as handle:
            checkpoint = pickle.load(handle)
        if "optimizer_state" not in checkpoint:
            raise ValueError("Checkpoint does not include optimizer state and cannot be resumed")
        previous_report = json.loads(report_path.read_text(encoding="utf-8"))
        previous_history = previous_report.get("history", [])
        original_input_nrmse = float(
            previous_report.get(
                "initial_input_nrmse",
                previous_history[0]["terminal_nrmse"] if previous_history else previous_report["input_nrmse"],
            )
        )
        initial_parameters = jax.device_put(checkpoint["parameters"])
        initial_optimizer_state = jax.device_put(checkpoint["optimizer_state"])
    else:
        dataset_path, manifest_path = prepare_cosmos_overfit_sample(
            arguments.cosmos_data,
            prepared_data_path,
            CosmosSimulationConfig(
                snr=arguments.snr,
                seed=arguments.seed,
                voxel_size_zyx=tuple(arguments.voxel_size),
                b0_direction_zyx=tuple(arguments.b0_direction),
                phase_scale_radians_per_susceptibility_unit=arguments.phase_scale,
            ),
        )
        previous_history = []
        original_input_nrmse = None
        initial_parameters = None
        initial_optimizer_state = None
    sample = load_single_sample(dataset_path)
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
        supervised_metric="nrmse",
        epoch_chunk_size=arguments.epoch_chunk_size,
        include_magnitude_in_weight=arguments.include_magnitude_in_weight,
    )
    def show_progress(record: dict[str, float]) -> None:
        print(
            f"Epoch {int(record['iteration']):03d}/{arguments.total_epochs}: "
            f"NRMSE={record['terminal_nrmse']:.6e}, "
            f"loss={record['loss']:.6e}, "
            f"CG={record['max_cg_relative_residual']:.2e}",
            flush=True,
        )

    result = train_single_sample(
        sample,
        tdv_config,
        reconstruction_config,
        overfit_config,
        progress_callback=show_progress,
        initial_parameters=initial_parameters,
        initial_optimizer_state=initial_optimizer_state,
        epoch_offset=len(previous_history),
    )
    result = result._replace(history=tuple(previous_history) + result.history)
    if original_input_nrmse is None:
        original_input_nrmse = float(result.baseline_nrmse)
    save_training_result(result, sample, tdv_config, reconstruction_config, overfit_config, arguments.output_dir)
    figure_weight = effective_data_weight(
        sample.brain_mask,
        sample.magnitude,
        include_magnitude=arguments.include_magnitude_in_weight,
    )
    figure_path = save_orthogonal_reconstruction_slices(
        figure_weight * sample.local_field,
        result.reconstruction,
        sample.susceptibility,
        sample.brain_mask,
        arguments.output_dir / "orthogonal_slices.png",
        nrmse_value=float(result.validation_nrmse),
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update(
        {
            "metric": "masked_nrmse = ||chi_recon - chi_gt||_2 / ||chi_gt||_2",
            "initial_input_nrmse": original_input_nrmse,
            "input_nrmse": original_input_nrmse,
            "resume_input_nrmse": float(result.baseline_nrmse),
            "output_nrmse": float(result.validation_nrmse),
            "prepared_dataset": str(dataset_path),
            "prepared_manifest": str(manifest_path),
            "orthogonal_slice_figure": str(figure_path),
        }
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Input NRMSE:  {float(result.baseline_nrmse):.6e}")
    print(f"Output NRMSE: {float(result.validation_nrmse):.6e}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
