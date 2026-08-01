import numpy as np

from qsm_tdv.data.contract import load_single_sample
from qsm_tdv.data.synthetic import create_synthetic_sample


def test_legacy_nonzero_chi_init_is_ignored_in_favour_of_zero_initialisation(tmp_path):
    dataset_path, _ = create_synthetic_sample(tmp_path / "sample.npz", shape_zyx=(8, 8, 8))
    with np.load(dataset_path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["chi_init"] = np.ones_like(arrays["chi_init"], dtype=np.float32)
    np.savez_compressed(dataset_path, **arrays)

    sample = load_single_sample(dataset_path)
    batch = sample.as_batch()
    assert np.max(np.abs(np.asarray(batch["chi_init"]))) == 0.0
