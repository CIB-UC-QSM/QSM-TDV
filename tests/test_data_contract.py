from __future__ import annotations

import pytest
import torch

from tdv_qsm.data import QSMSample, validate_qsm_sample, validate_subject_splits


def _sample(subject_id: str = "sub-001") -> QSMSample:
    shape = (1, 4, 5, 6)
    image = torch.zeros(shape, dtype=torch.float32)
    return QSMSample(
        local_field=image,
        susceptibility=image,
        brain_mask=torch.ones(shape, dtype=torch.float32),
        magnitude_weight=torch.ones(shape, dtype=torch.float32),
        initial=image,
        voxel_size_zyx=torch.tensor([1.0, 1.2, 2.0]),
        b0_direction_zyx=torch.tensor([0.0, 0.0, 1.0]),
        subject_id=subject_id,
        field_unit="ppm",
        susceptibility_unit="ppm",
        reference_convention="already_referenced",
        processing_version="qsm-pipeline-v1",
    )


def test_complete_qsm_sample_contract_validates() -> None:
    validate_qsm_sample(_sample())


def test_sample_rejects_bad_weight_mask_voxel_and_b0_metadata() -> None:
    sample = _sample()
    sample.magnitude_weight[..., 0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        validate_qsm_sample(sample)
    sample = _sample()
    sample.brain_mask.zero_()
    with pytest.raises(ValueError, match="nonempty"):
        validate_qsm_sample(sample)
    sample = _sample()
    sample.voxel_size_zyx[0] = 0.0
    with pytest.raises(ValueError, match="positive"):
        validate_qsm_sample(sample)
    sample = _sample()
    sample.b0_direction_zyx[:] = torch.tensor([0.0, 0.0, 2.0])
    with pytest.raises(ValueError, match="normalized"):
        validate_qsm_sample(sample)


def test_subject_splits_reject_duplicates_across_splits() -> None:
    with pytest.raises(ValueError, match="multiple splits"):
        validate_subject_splits({"train": [_sample("sub-a")], "test": [_sample("sub-a")]})
