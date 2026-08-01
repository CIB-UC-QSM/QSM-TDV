import numpy as np

from qsm_tdv.data.contract import effective_data_weight, load_single_sample
from qsm_tdv.data.synthetic import create_synthetic_sample


def test_legacy_chi_init_is_ignored_in_favour_of_weighted_local_field(tmp_path):
    dataset_path, _ = create_synthetic_sample(tmp_path / "sample.npz", shape_zyx=(8, 8, 8))
    with np.load(dataset_path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["chi_init"] = np.ones_like(arrays["chi_init"], dtype=np.float32)
    np.savez_compressed(dataset_path, **arrays)

    sample = load_single_sample(dataset_path)
    batch = sample.as_batch()
    expected = sample.brain_mask * sample.local_field
    assert np.max(np.abs(np.asarray(batch["chi_init"])[0] - expected)) == 0.0


def test_effective_data_weight_uses_magnitude_only_when_enabled_and_available():
    mask = np.asarray([[[[1.0], [0.0]]]], dtype=np.float32)
    magnitude = np.asarray([[[[0.25], [0.75]]]], dtype=np.float32)

    assert np.array_equal(effective_data_weight(mask, magnitude, include_magnitude=True), mask * magnitude)
    assert np.array_equal(effective_data_weight(mask, magnitude, include_magnitude=False), mask)
    assert np.array_equal(effective_data_weight(mask, None, include_magnitude=True), mask)
