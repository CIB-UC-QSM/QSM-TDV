"""Publication-ready orthogonal susceptibility-slice visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _unbatch(volume: np.ndarray) -> np.ndarray:
    array = np.asarray(volume, dtype=np.float32)
    if array.ndim == 5:
        if array.shape[0] != 1 or array.shape[-1] != 1:
            raise ValueError(f"Expected a single NDHWC volume, got {array.shape}")
        array = array[0, ..., 0]
    elif array.ndim == 4:
        if array.shape[-1] != 1:
            raise ValueError(f"Expected [Z,Y,X,1], got {array.shape}")
        array = array[..., 0]
    if array.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {array.shape}")
    return array


def _slice_indices(mask: np.ndarray) -> tuple[int, int, int]:
    support = np.argwhere(mask > 0)
    if support.size == 0:
        return tuple(size // 2 for size in mask.shape)
    centre = np.rint(np.mean(support, axis=0)).astype(int)
    return tuple(int(value) for value in centre)


def _plane(volume: np.ndarray, orientation: str, index: int) -> tuple[np.ndarray, str, str]:
    if orientation == "sagittal":
        return volume[:, :, index].T, "Z", "Y"
    if orientation == "coronal":
        return volume[:, index, :].T, "Z", "X"
    if orientation == "axial":
        return volume[index, :, :], "Y", "X"
    raise ValueError(f"Unknown orientation: {orientation}")


def save_orthogonal_reconstruction_slices(
    chi_input: np.ndarray,
    chi_output: np.ndarray,
    chi_ground_truth: np.ndarray,
    brain_mask: np.ndarray,
    output_path: str | Path,
    *,
    nrmse_value: float,
) -> Path:
    """Save sagittal, coronal, and axial input/output reconstruction slices.

    The figure also includes COSMOS ground truth and absolute error so the
    requested input/output comparison has a common quantitative reference.
    Input, output, and ground truth share one symmetric susceptibility scale.
    """

    return save_orthogonal_evaluation_slices(
        chi_input,
        chi_output,
        brain_mask,
        output_path,
        chi_ground_truth=chi_ground_truth,
        nrmse_value=nrmse_value,
        title_prefix="COSMOS TDV-QSM overfit diagnostic",
    )


def save_orthogonal_evaluation_slices(
    chi_input: np.ndarray,
    chi_output: np.ndarray,
    brain_mask: np.ndarray,
    output_path: str | Path,
    *,
    chi_ground_truth: np.ndarray | None = None,
    nrmse_value: float | None = None,
    title_prefix: str = "TDV-QSM evaluation",
) -> Path:
    """Save sagittal, coronal, and axial input/prediction/optional-GT slices.

    The input is the zero initialisation ``chi_0=0`` and the prediction is the
    final semi-implicit TDV-QSM state.  Ground truth is optional so the
    same report works for in-vivo or challenge inputs that have no ``chi``.
    """

    input_volume = _unbatch(chi_input)
    output_volume = _unbatch(chi_output)
    mask = _unbatch(brain_mask)
    ground_truth = None if chi_ground_truth is None else _unbatch(chi_ground_truth)
    if input_volume.shape != output_volume.shape or input_volume.shape != mask.shape:
        raise ValueError("Input, output, and mask must share the same spatial shape")
    if ground_truth is not None and ground_truth.shape != input_volume.shape:
        raise ValueError("Ground truth must share the input spatial shape")
    indices = _slice_indices(mask)
    support = mask > 0
    displayed_volumes = [input_volume[support], output_volume[support]]
    if ground_truth is not None:
        displayed_volumes.append(ground_truth[support])
    data_limit = float(np.max(np.abs(np.concatenate(displayed_volumes))))
    data_limit = max(data_limit, np.finfo(np.float32).eps)
    if ground_truth is None:
        error = None
        error_limit = None
        columns = (
            ("Input $\\chi_0=0$", input_volume, "susceptibility"),
            ("TDV-QSM prediction $\\chi_S$", output_volume, "susceptibility"),
        )
    else:
        error = np.abs(output_volume - ground_truth) * support
        error_limit = max(float(np.max(error[support])), np.finfo(np.float32).eps)
        columns = (
            ("Input $\\chi_0=0$", input_volume, "susceptibility"),
            ("TDV-QSM prediction $\\chi_S$", output_volume, "susceptibility"),
            ("Ground truth $\\chi_{gt}$", ground_truth, "susceptibility"),
            ("Absolute error", error, "error"),
        )

    figure, axes = plt.subplots(3, len(columns), figsize=(3.3 * len(columns), 10), constrained_layout=True)
    axes = np.asarray(axes).reshape(3, len(columns))
    orientations = (("sagittal", indices[2]), ("coronal", indices[1]), ("axial", indices[0]))
    susceptibility_images = []
    error_images = []
    for row, (orientation, index) in enumerate(orientations):
        for column, (title, volume, kind) in enumerate(columns):
            plane, horizontal_axis, vertical_axis = _plane(volume, orientation, index)
            axis = axes[row, column]
            if kind == "susceptibility":
                image = axis.imshow(plane, cmap="gray", vmin=-data_limit, vmax=data_limit, origin="lower")
                susceptibility_images.append(image)
            else:
                image = axis.imshow(plane, cmap="magma", vmin=0.0, vmax=error_limit, origin="lower")
                error_images.append(image)
            if row == 0:
                axis.set_title(title)
            if column == 0:
                axis.set_ylabel(f"{orientation.title()}\n{vertical_axis}")
            axis.set_xlabel(horizontal_axis)
            axis.set_xticks([])
            axis.set_yticks([])
    susceptibility_columns = 3 if ground_truth is not None else 2
    figure.colorbar(
        susceptibility_images[0],
        ax=axes[:, :susceptibility_columns],
        shrink=0.82,
        label="Susceptibility (source units)",
    )
    if error_images:
        figure.colorbar(error_images[0], ax=axes[:, -1], shrink=0.82, label="Absolute error")
    title = title_prefix
    if nrmse_value is not None:
        title += f" — masked NRMSE = {nrmse_value:.5f}"
    figure.suptitle(title)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination
