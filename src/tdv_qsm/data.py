"""Validated medical-volume sample contract for the QSM project extension."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QSMSample:
    """One unbatched, anonymized sample in canonical ``[1,Z,Y,X]`` layout."""

    local_field: torch.Tensor
    susceptibility: torch.Tensor
    brain_mask: torch.Tensor
    magnitude_weight: torch.Tensor
    initial: torch.Tensor
    voxel_size_zyx: torch.Tensor
    b0_direction_zyx: torch.Tensor
    subject_id: str
    field_unit: str
    susceptibility_unit: str
    reference_convention: str
    processing_version: str
    spatial_affine: torch.Tensor | None = None
    component_affines: Mapping[str, torch.Tensor] | None = None


def validate_qsm_sample(sample: QSMSample, *, b0_tolerance: float = 1e-5) -> None:
    """Reject samples that violate tensor, metadata, or affine invariants."""

    image_names = (
        "local_field",
        "susceptibility",
        "brain_mask",
        "magnitude_weight",
        "initial",
    )
    images = {name: getattr(sample, name) for name in image_names}
    reference_shape: tuple[int, ...] | None = None
    for name, value in images.items():
        if value.dtype != torch.float32:
            raise ValueError(f"{name} must be float32.")
        if value.ndim != 4 or value.shape[0] != 1:
            raise ValueError(f"{name} must have shape [1, Z, Y, X].")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values.")
        if reference_shape is None:
            reference_shape = tuple(value.shape)
        elif tuple(value.shape) != reference_shape:
            raise ValueError("All image arrays must have identical shapes.")
    if torch.any(sample.magnitude_weight < 0.0):
        raise ValueError("magnitude_weight must be nonnegative.")
    if torch.any(sample.brain_mask < 0.0) or not torch.any(sample.brain_mask > 0.0):
        raise ValueError("brain_mask must be nonnegative and nonempty.")

    for name, value in (
        ("voxel_size_zyx", sample.voxel_size_zyx),
        ("b0_direction_zyx", sample.b0_direction_zyx),
    ):
        if value.dtype != torch.float32 or value.shape != (3,):
            raise ValueError(f"{name} must be float32 with shape [3].")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values.")
    if torch.any(sample.voxel_size_zyx <= 0.0):
        raise ValueError("voxel_size_zyx must be strictly positive.")
    b0_norm = torch.linalg.vector_norm(sample.b0_direction_zyx)
    if not torch.isclose(
        b0_norm,
        torch.ones_like(b0_norm),
        atol=b0_tolerance,
        rtol=b0_tolerance,
    ):
        raise ValueError("b0_direction_zyx must be normalized for every sample.")

    for name in (
        "subject_id",
        "field_unit",
        "susceptibility_unit",
        "reference_convention",
        "processing_version",
    ):
        value = getattr(sample, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be an explicit nonempty string.")
    if any(marker in sample.subject_id for marker in ("@", "/", "\\")):
        raise ValueError("subject_id must be anonymized and must not contain contact/path data.")

    if sample.spatial_affine is not None:
        affine = sample.spatial_affine
        if affine.shape != (4, 4) or not torch.isfinite(affine).all():
            raise ValueError("spatial_affine must be a finite [4,4] tensor.")
        if sample.component_affines is not None:
            for name, component_affine in sample.component_affines.items():
                if component_affine.shape != (4, 4) or not torch.isfinite(component_affine).all():
                    raise ValueError(f"Affine for {name} must be finite with shape [4,4].")
                if not torch.allclose(component_affine, affine, atol=1e-5, rtol=1e-5):
                    raise ValueError(f"Affine for {name} is inconsistent with spatial_affine.")
    elif sample.component_affines:
        raise ValueError("component_affines require a reference spatial_affine.")


def validate_subject_splits(splits: Mapping[str, Iterable[QSMSample]]) -> None:
    """Validate all samples and enforce subject-level split isolation."""

    subject_split: dict[str, str] = {}
    for split_name, samples in splits.items():
        if not split_name:
            raise ValueError("Split names must be nonempty.")
        for sample in samples:
            validate_qsm_sample(sample)
            previous = subject_split.get(sample.subject_id)
            if previous is not None and previous != split_name:
                raise ValueError(
                    f"Subject {sample.subject_id!r} appears in multiple splits: "
                    f"{previous!r} and {split_name!r}."
                )
            subject_split[sample.subject_id] = split_name
