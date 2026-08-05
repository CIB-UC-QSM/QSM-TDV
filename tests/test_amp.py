from __future__ import annotations

import copy

import pytest
import torch

from tdv_qsm.models.energy import TDVEnergy3D
from tdv_qsm.models.explicit_tdv import ExplicitTDVQSM3D
from tdv_qsm.operators.dipole import QSMOperator, build_dipole_kernel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for float16 AMP")
def test_cuda_amp_matches_float32_and_preserves_force_state_dtypes() -> None:
    torch.manual_seed(21)
    device = torch.device("cuda")
    shape = (5, 6, 7)
    regularizer_fp32 = TDVEnergy3D(
        num_features=2,
        num_macro_blocks=1,
        use_amp=False,
    ).to(device)
    regularizer_amp = copy.deepcopy(regularizer_fp32)
    regularizer_amp.use_amp = True
    kernel = build_dipole_kernel(shape, device=device)
    reference = ExplicitTDVQSM3D(
        regularizer_fp32,
        QSMOperator(kernel),
        num_steps=2,
        fixed_T=0.02,
        fixed_lambda=0.02,
    ).to(device)
    mixed = ExplicitTDVQSM3D(
        regularizer_amp,
        QSMOperator(kernel),
        num_steps=2,
        fixed_T=0.02,
        fixed_lambda=0.02,
    ).to(device)
    initial = torch.randn(1, 1, *shape, device=device)
    field = torch.randn_like(initial)
    mask = torch.ones_like(initial)
    weight = torch.rand_like(initial)
    reference_output = reference(field, mask, None, weight, initial).susceptibility
    mixed_output = mixed(field, mask, None, weight, initial).susceptibility
    assert mixed.regularizer.force(initial).dtype == torch.float32
    assert mixed_output.dtype == torch.float32
    torch.testing.assert_close(mixed_output, reference_output, atol=5e-3, rtol=5e-2)
