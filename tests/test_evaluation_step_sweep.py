import csv

import numpy as np
from qsm_tdv.evaluation import cosmos_step_sweep

from qsm_tdv.evaluation.cosmos_step_sweep import (
    load_iteration_metrics,
    save_iteration_metric_figure,
    save_step_sweep_metric_figure,
)
from qsm_tdv.evaluation.input_set import _rmse, _tol_update


def test_evaluation_metrics_and_step_sweep_figures(tmp_path):
    previous = np.array([[[1.0], [2.0]]], dtype=np.float32)
    current = np.array([[[2.0], [2.0]]], dtype=np.float32)
    reference = np.array([[[1.0], [5.0]]], dtype=np.float32)
    mask = np.array([[[1.0], [0.0]]], dtype=np.float32)

    assert np.isclose(_rmse(current, reference, mask), 1.0)
    assert np.isclose(
        _tol_update(current, previous),
        np.linalg.norm((current - previous).ravel()) / np.linalg.norm(previous.ravel()),
    )

    metrics_path = tmp_path / "iteration_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("iteration", "rmse_to_gt", "tol_update"))
        writer.writeheader()
        writer.writerows(
            (
                {"iteration": 0, "rmse_to_gt": 0.5, "tol_update": None},
                {"iteration": 1, "rmse_to_gt": 0.25, "tol_update": 0.1},
            )
        )
    rows = load_iteration_metrics(metrics_path)
    assert rows[0]["tol_update"] is None
    assert rows[1]["rmse_to_gt"] == 0.25

    single_figure = save_iteration_metric_figure(rows, tmp_path / "steps-2" / "iteration_metrics.png", steps=2)
    sweep_figure = save_step_sweep_metric_figure({2: rows, 10: rows}, tmp_path / "step_sweep_metrics.png")
    assert single_figure.is_file() and single_figure.stat().st_size > 1_000
    assert sweep_figure.is_file() and sweep_figure.stat().st_size > 1_000


def test_cosmos_step_sweep_runs_all_requested_lengths(monkeypatch, tmp_path):
    requested_steps: list[int] = []

    def fake_evaluate_checkpoint(_checkpoint, _input, output_dir, *, steps, **_kwargs):
        requested_steps.append(steps)
        destination = output_dir
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / "iteration_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("iteration", "rmse_to_gt", "tol_update"))
            writer.writeheader()
            writer.writerows(
                (
                    {"iteration": 0, "rmse_to_gt": 0.5, "tol_update": None},
                    {"iteration": steps, "rmse_to_gt": 0.25, "tol_update": 0.1},
                )
            )
        return {
            "stopping_time": 0.2,
            "tau": 0.2 / steps,
            "final_rmse_to_gt": 0.25,
            "final_nrmse_to_gt": 0.5,
            "final_tol_update": 0.1,
            "max_cg_relative_residual": 1e-6,
            "cg_tolerance_met": True,
        }

    monkeypatch.setattr(cosmos_step_sweep, "evaluate_checkpoint", fake_evaluate_checkpoint)
    cosmos_dir = tmp_path / "cosmos"
    cosmos_dir.mkdir()
    (cosmos_dir / "chi_cosmos.mat").touch()
    (cosmos_dir / "msk.mat").touch()
    report = cosmos_step_sweep.evaluate_cosmos_step_sweep(
        tmp_path / "checkpoint.pkl", tmp_path / "cosmos", tmp_path / "evaluation"
    )

    assert requested_steps == [2, 10, 50, 100]
    assert report["step_counts"] == [2, 10, 50, 100]
    assert (tmp_path / "evaluation" / "step_sweep_summary.csv").is_file()
    assert (tmp_path / "evaluation" / "step_sweep_iteration_metrics.png").is_file()
    for steps in report["step_counts"]:
        assert (tmp_path / "evaluation" / f"steps-{steps}" / "iteration_metrics.png").is_file()



def test_cosmos_step_sweep_resolves_workspace_fallback(monkeypatch, tmp_path):
    workspace_cosmos = tmp_path / "cosmos_data"
    workspace_cosmos.mkdir()
    (workspace_cosmos / "chi_cosmos.mat").touch()
    (workspace_cosmos / "msk.mat").touch()
    monkeypatch.chdir(tmp_path)

    assert cosmos_step_sweep.resolve_cosmos_evaluation_directory("/cosmos_data") == workspace_cosmos
