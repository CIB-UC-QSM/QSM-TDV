import json

import numpy as np
from scipy.io import savemat

from qsm_tdv.data.contract import load_single_sample
from qsm_tdv.data.cosmos import CosmosSimulationConfig, prepare_cosmos_overfit_sample
from qsm_tdv.evaluation.input_set import load_evaluation_input
from qsm_tdv.evaluation.slices import (
    save_orthogonal_evaluation_slices,
    save_orthogonal_reconstruction_slices,
)


def test_cosmos_complex_signal_simulation_snr_and_slice_figure(tmp_path):
    cosmos_dir = tmp_path / "cosmos"
    cosmos_dir.mkdir()
    shape = (8, 8, 8)
    z, y, x = np.meshgrid(
        np.linspace(-1, 1, shape[0]), np.linspace(-1, 1, shape[1]), np.linspace(-1, 1, shape[2]), indexing="ij"
    )
    mask = ((z**2 + y**2 + x**2) < 0.9).astype(np.uint8)
    chi = (0.2 * np.exp(-4 * ((z + 0.2) ** 2 + y**2 + x**2)) * mask).astype(np.float32)
    magnitude = (0.2 + 0.8 * mask).astype(np.float32)
    savemat(cosmos_dir / "chi_cosmos.mat", {"chi_cosmos": chi})
    savemat(cosmos_dir / "magn.mat", {"magn": magnitude})
    savemat(cosmos_dir / "msk.mat", {"msk": mask})

    dataset_path, manifest_path = prepare_cosmos_overfit_sample(
        cosmos_dir,
        tmp_path / "cosmos_snr70.npz",
        CosmosSimulationConfig(snr=70.0, seed=3),
    )
    sample = load_single_sample(dataset_path)
    manifest = json.loads(manifest_path.read_text())
    with np.load(dataset_path) as archive:
        clean_signal = archive["clean_signal"][..., 0]
        noisy_signal = archive["noisy_signal"][..., 0]
    realised_snr = np.linalg.norm((clean_signal * mask).ravel()) / np.linalg.norm(((noisy_signal - clean_signal) * mask).ravel())
    assert sample.susceptibility.shape == (8, 8, 8, 1)
    assert float(np.max(np.abs(sample.chi_init))) == 0.0
    assert abs(realised_snr - 70.0) < 2e-4
    assert abs(manifest["realised_snr"] - 70.0) < 2e-4
    assert manifest["noise_model"].startswith("circular complex Gaussian")

    figure = save_orthogonal_reconstruction_slices(
        sample.chi_init,
        sample.chi_init,
        sample.susceptibility,
        sample.brain_mask,
        tmp_path / "orthogonal_slices.png",
        nrmse_value=0.25,
    )
    assert figure.is_file()
    assert figure.stat().st_size > 1_000


def test_evaluation_input_prefers_phase_and_simulates_when_only_chi_is_available(tmp_path):
    shape = (8, 8, 8)
    chi = np.zeros(shape, dtype=np.float32)
    chi[3:5, 3:5, 3:5] = 0.1
    mask = np.ones(shape, dtype=np.uint8)
    simulated_dir = tmp_path / "simulated"
    simulated_dir.mkdir()
    savemat(simulated_dir / "chi.mat", {"chi": chi})
    savemat(simulated_dir / "msk.mat", {"msk": mask})
    simulated = load_evaluation_input(
        simulated_dir,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        b0_direction_zyx=(0.0, 0.0, 1.0),
    )
    assert simulated.input_mode == "simulated_from_chi"
    assert simulated.susceptibility is not None
    assert simulated.local_field.shape == shape

    phase_dir = tmp_path / "phase"
    phase_dir.mkdir()
    phase = np.full(shape, 0.25, dtype=np.float32)
    savemat(phase_dir / "phase_in.mat", {"phase_in": phase})
    savemat(phase_dir / "msk.mat", {"msk": mask})
    phase_input = load_evaluation_input(
        phase_dir,
        voxel_size_zyx=(1.0, 1.0, 1.0),
        b0_direction_zyx=(0.0, 0.0, 1.0),
    )
    assert phase_input.input_mode == "phase_in"
    assert phase_input.susceptibility is None
    assert np.max(np.abs(phase_input.local_field - phase)) == 0.0
    figure = save_orthogonal_evaluation_slices(
        phase,
        phase * 0.5,
        mask,
        tmp_path / "phase_only_slices.png",
    )
    assert figure.is_file()
    assert figure.stat().st_size > 1_000
