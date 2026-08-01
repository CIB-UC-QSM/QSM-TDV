"""Publication-ready orthogonal susceptibility-slice visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SUSCEPTIBILITY_DISPLAY_RANGE = (-0.1, 0.1)
ABSOLUTE_ERROR_DISPLAY_RANGE = (0.0, 0.5)
COSMOS_SLICE_ROTATIONS_CLOCKWISE = {"sagittal": 180, "axial": 90}


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


def _rotate_plane_clockwise(
    plane: np.ndarray,
    horizontal_axis: str,
    vertical_axis: str,
    degrees: int,
) -> tuple[np.ndarray, str, str]:
    """Rotate a displayed plane by an integer multiple of 90° clockwise."""

    if degrees % 90 != 0:
        raise ValueError("Slice rotations must be integer multiples of 90 degrees")
    quarter_turns = (degrees // 90) % 4
    rotated = np.rot90(plane, k=-quarter_turns)
    if quarter_turns % 2:
        horizontal_axis, vertical_axis = vertical_axis, horizontal_axis
    return rotated, horizontal_axis, vertical_axis


def _resolved_slice_rotations(rotations: Mapping[str, int] | None) -> dict[str, int]:
    """Validate and fill per-orientation clockwise display rotations."""

    resolved = {"sagittal": 0, "coronal": 0, "axial": 0}
    if rotations is None:
        return resolved
    unexpected = set(rotations).difference(resolved)
    if unexpected:
        raise ValueError(f"Unknown slice orientations for rotation: {sorted(unexpected)}")
    for orientation, degrees in rotations.items():
        if degrees % 90 != 0:
            raise ValueError("Slice rotations must be integer multiples of 90 degrees")
        resolved[orientation] = degrees
    return resolved


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
    Input, output, and ground truth use the fixed susceptibility range
    [-0.1, 0.1].
    """

    return save_orthogonal_evaluation_slices(
        chi_input,
        chi_output,
        brain_mask,
        output_path,
        chi_ground_truth=chi_ground_truth,
        nrmse_value=nrmse_value,
        title_prefix="COSMOS TDV-QSM overfit diagnostic",
        slice_rotations_clockwise=COSMOS_SLICE_ROTATIONS_CLOCKWISE,
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
    slice_rotations_clockwise: Mapping[str, int] | None = None,
) -> Path:
    """Save sagittal, coronal, and axial input/prediction/optional-GT slices.

    The input is the weighted field initialisation ``chi_0=W b`` and the
    prediction is the final semi-implicit TDV-QSM state.  Ground truth is optional so the
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
    rotations = _resolved_slice_rotations(slice_rotations_clockwise)
    indices = _slice_indices(mask)
    support = mask > 0
    if ground_truth is None:
        error = None
        columns = (
            ("Input $\\chi_0=Wb$", input_volume, "susceptibility"),
            ("TDV-QSM prediction $\\chi_S$", output_volume, "susceptibility"),
        )
    else:
        error = np.abs(output_volume - ground_truth) * support
        columns = (
            ("Input $\\chi_0=Wb$", input_volume, "susceptibility"),
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
            plane, horizontal_axis, vertical_axis = _rotate_plane_clockwise(
                plane,
                horizontal_axis,
                vertical_axis,
                rotations[orientation],
            )
            axis = axes[row, column]
            if kind == "susceptibility":
                image = axis.imshow(
                    plane,
                    cmap="gray",
                    vmin=SUSCEPTIBILITY_DISPLAY_RANGE[0],
                    vmax=SUSCEPTIBILITY_DISPLAY_RANGE[1],
                    origin="lower",
                )
                susceptibility_images.append(image)
            else:
                image = axis.imshow(
                    plane,
                    cmap="magma",
                    vmin=ABSOLUTE_ERROR_DISPLAY_RANGE[0],
                    vmax=ABSOLUTE_ERROR_DISPLAY_RANGE[1],
                    origin="lower",
                )
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
