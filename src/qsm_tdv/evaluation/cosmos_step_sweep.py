"""Evaluate one TDV-QSM checkpoint on COSMOS at fixed reconstruction lengths.

The checkpoint parameters and learned stopping time are held fixed. Only the
number of semi-implicit steps changes, so every reconstruction uses ``tau=T/S``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qsm_tdv.evaluation.input_set import evaluate_checkpoint


COSMOS_STEP_COUNTS = (2, 10, 50, 100)


def resolve_cosmos_evaluation_directory(requested: str | Path) -> Path:
    """Resolve a COSMOS directory from its mount path or workspace fallback."""

    candidate = Path(requested).expanduser()
    workspace_fallback = Path.cwd() / "cosmos_data"
    for directory in (candidate, workspace_fallback):
        if (directory / "msk.mat").is_file() and any(
            (directory / name).is_file() for name in ("chi_cosmos.mat", "chi.mat")
        ):
            return directory.resolve()
    raise FileNotFoundError(
        f"Expected msk.mat plus chi_cosmos.mat or chi.mat in {candidate}; "
        f"also checked workspace fallback {workspace_fallback}."
    )


def _optional_float(value: str | None) -> float | None:
    return None if value is None or value == "" else float(value)


def load_iteration_metrics(path: str | Path) -> list[dict[str, float | int | None]]:
    """Read iteration metrics produced by :func:`evaluate_checkpoint`."""

    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"iteration", "rmse_to_gt", "tol_update"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{source} is missing metric columns: {sorted(missing)}")
        rows = [
            {
                "iteration": int(record["iteration"]),
                "rmse_to_gt": _optional_float(record["rmse_to_gt"]),
                "tol_update": _optional_float(record["tol_update"]),
            }
            for record in reader
        ]
    if not rows:
        raise ValueError(f"{source} contains no iteration metrics")
    return rows


def _series(
    rows: Sequence[Mapping[str, float | int | None]], metric: str
) -> tuple[list[int], list[float]]:
    iterations: list[int] = []
    values: list[float] = []
    for row in rows:
        value = row[metric]
        if value is not None:
            iterations.append(int(row["iteration"]))
            values.append(float(value))
    return iterations, values


def _plot_metrics(axes: Sequence[Any], metrics_by_steps: Mapping[int, Sequence[Mapping[str, float | int | None]]]) -> None:
    rmse_axis, update_axis = axes
    for steps, rows in metrics_by_steps.items():
        rmse_iterations, rmse_values = _series(rows, "rmse_to_gt")
        update_iterations, update_values = _series(rows, "tol_update")
        label = f"S={steps}"
        rmse_axis.plot(rmse_iterations, rmse_values, marker="o", markersize=3, label=label)
        update_axis.plot(update_iterations, update_values, marker="o", markersize=3, label=label)
    rmse_axis.set(
        xlabel="Semi-implicit iteration",
        ylabel="Masked RMSE to COSMOS",
        title="Iteration-wise reconstruction error",
    )
    update_axis.set(
        xlabel="Semi-implicit iteration",
        ylabel="tol_update = ||x_now - x_prev||_2 / ||x_prev||_2",
        title="Relative reconstruction update",
    )
    for axis in (rmse_axis, update_axis):
        axis.grid(alpha=0.25)
        axis.legend()


def save_iteration_metric_figure(
    rows: Sequence[Mapping[str, float | int | None]], output_path: str | Path, *, steps: int
) -> Path:
    """Save RMSE and ``tol_update`` curves for one reconstruction length."""

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    _plot_metrics(axes, {steps: rows})
    figure.suptitle(f"COSMOS TDV-QSM iteration diagnostics (S={steps})")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination


def save_step_sweep_metric_figure(
    metrics_by_steps: Mapping[int, Sequence[Mapping[str, float | int | None]]], output_path: str | Path
) -> Path:
    """Save an RMSE/tolerance overlay for all requested reconstruction lengths."""

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    _plot_metrics(axes, metrics_by_steps)
    figure.suptitle("COSMOS TDV-QSM step-count comparison")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination


def _write_summary(path: Path, reports: Mapping[int, Mapping[str, Any]]) -> None:
    fields = (
        "steps",
        "stopping_time",
        "tau",
        "final_rmse_to_gt",
        "final_nrmse_to_gt",
        "final_tol_update",
        "max_cg_relative_residual",
        "cg_tolerance_met",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for steps, report in reports.items():
            writer.writerow({"steps": steps, **{name: report.get(name) for name in fields[1:]}})


def _print_iteration_metrics(steps: int, rows: Sequence[Mapping[str, float | int | None]]) -> None:
    print(f"\nS={steps}: iteration-wise metrics", flush=True)
    print("iteration,rmse_to_gt,tol_update", flush=True)
    for row in rows:
        rmse = "" if row["rmse_to_gt"] is None else f"{float(row['rmse_to_gt']):.8e}"
        update = "" if row["tol_update"] is None else f"{float(row['tol_update']):.8e}"
        print(f"{int(row['iteration'])},{rmse},{update}", flush=True)


def evaluate_cosmos_step_sweep(
    checkpoint_path: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    cg_iterations: int | None = None,
    voxel_size_zyx: tuple[float, float, float] | None = None,
    b0_direction_zyx: tuple[float, float, float] | None = None,
    include_magnitude_in_weight: bool = True,
) -> dict[str, Any]:
    """Evaluate the same regularizer at ``S=2,10,50,100`` on COSMOS."""

    input_directory = resolve_cosmos_evaluation_directory(data_dir)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    reports: dict[int, Mapping[str, Any]] = {}
    metrics_by_steps: dict[int, list[dict[str, float | int | None]]] = {}
    for steps in COSMOS_STEP_COUNTS:
        step_dir = destination / f"steps-{steps}"
        report = evaluate_checkpoint(
            checkpoint_path,
            input_directory,
            step_dir,
            steps=steps,
            cg_iterations=cg_iterations,
            voxel_size_zyx=voxel_size_zyx,
            b0_direction_zyx=b0_direction_zyx,
            include_magnitude_in_weight=include_magnitude_in_weight,
        )
        rows = load_iteration_metrics(step_dir / "iteration_metrics.csv")
        if not any(row["rmse_to_gt"] is not None for row in rows):
            raise ValueError("COSMOS step sweep requires chi_cosmos.mat or chi.mat for iteration-wise RMSE")
        save_iteration_metric_figure(rows, step_dir / "iteration_metrics.png", steps=steps)
        reports[steps] = report
        metrics_by_steps[steps] = rows
        _print_iteration_metrics(steps, rows)

    summary_path = destination / "step_sweep_summary.csv"
    _write_summary(summary_path, reports)
    comparison_figure = save_step_sweep_metric_figure(
        metrics_by_steps, destination / "step_sweep_iteration_metrics.png"
    )
    sweep_report = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "input_directory": str(input_directory.resolve()),
        "step_counts": list(COSMOS_STEP_COUNTS),
        "rmse_definition": "masked sqrt(mean((chi_s - chi_gt)^2)) over brain_mask > 0",
        "tol_update_definition": "||x_now - x_prev||_2 / ||x_prev||_2 over the full reconstruction state",
        "summary_csv": str(summary_path),
        "iteration_metrics_figure": str(comparison_figure),
        "step_reports": {str(steps): report for steps, report in reports.items()},
    }
    report_path = destination / "step_sweep_report.json"
    report_path.write_text(json.dumps(sweep_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sweep_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained TDV-QSM regularizer on COSMOS with S=2, 10, 50, and 100."
    )
    parser.add_argument(
        "--cosmos-data",
        "--input-dir",
        dest="input_dir",
        type=Path,
        required=True,
        help="COSMOS directory containing msk.mat and chi_cosmos.mat/chi.mat; phase_in.mat is optional",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trusted local TDV-QSM checkpoint.pkl")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/cosmos-step-evaluation"))
    parser.add_argument(
        "--cg-iterations",
        type=int,
        default=None,
        help="Fixed CG iterations per TDV-QSM step; defaults to the checkpoint configuration",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        nargs=3,
        default=None,
        metavar=("VZ", "VY", "VX"),
        help="Voxel size in z,y,x order; defaults to checkpoint metadata",
    )
    parser.add_argument(
        "--b0-direction",
        type=float,
        nargs=3,
        default=None,
        metavar=("BZ", "BY", "BX"),
        help="B0 direction in z,y,x order; defaults to checkpoint metadata",
    )
    parser.add_argument(
        "--include-magnitude-in-weight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use W = mask * magnitude when magnitude is present (default: enabled)",
    )
    arguments = parser.parse_args()
    report = evaluate_cosmos_step_sweep(
        arguments.checkpoint,
        arguments.input_dir,
        arguments.output_dir,
        cg_iterations=arguments.cg_iterations,
        voxel_size_zyx=None if arguments.voxel_size is None else tuple(arguments.voxel_size),
        b0_direction_zyx=None if arguments.b0_direction is None else tuple(arguments.b0_direction),
        include_magnitude_in_weight=arguments.include_magnitude_in_weight,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
