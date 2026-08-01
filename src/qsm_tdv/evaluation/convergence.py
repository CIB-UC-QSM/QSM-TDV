"""Persistent numerical and graphical training-convergence artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_convergence_artifacts(
    history: Sequence[Mapping[str, float]], output_dir: str | Path
) -> tuple[Path, Path]:
    """Save complete epoch logs as CSV plus a compact convergence figure."""

    if not history:
        raise ValueError("Cannot save convergence artifacts for an empty history")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for record in history for field in record})
    csv_path = destination / "convergence.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)

    epochs = np.asarray([record["iteration"] for record in history], dtype=np.float32)
    nrmse_values = np.asarray([record["terminal_nrmse"] for record in history], dtype=np.float32)
    loss_values = np.asarray([record["loss"] for record in history], dtype=np.float32)
    data_values = np.asarray([record["data_consistency"] for record in history], dtype=np.float32)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    axes[0].plot(epochs, nrmse_values, marker="o", markersize=3, label="NRMSE")
    axes[0].plot(epochs, loss_values, marker="o", markersize=3, label="training loss", alpha=0.75)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Error")
    axes[0].set_title("Supervised convergence")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(epochs, data_values, marker="o", markersize=3, color="tab:orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(r"$\|M(A\chi-b)\|_2$")
    axes[1].set_title("Data consistency")
    axes[1].grid(alpha=0.25)
    figure_path = destination / "convergence.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return csv_path, figure_path
