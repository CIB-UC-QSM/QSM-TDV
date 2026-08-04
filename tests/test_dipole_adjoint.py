from __future__ import annotations

import pytest
import torch

from tdv_qsm.operators.dipole import DipoleOperator3D, build_dipole_kernel


@pytest.mark.parametrize("shape", [(5, 6, 7), (6, 8, 10)])
def test_dipole_adjoint_identity_for_anisotropic_arbitrary_b0(shape: tuple[int, int, int]) -> None:
    kernel = build_dipole_kernel(
        shape,
        voxel_size_zyx=(0.8, 1.2, 2.3),
        b0_direction_zyx=(1.0, -2.0, 0.5),
    )
    operator = DipoleOperator3D(kernel)
    x = torch.randn(2, 1, *shape)
    y = torch.randn_like(x)
    left = torch.sum(operator.forward(x) * y)
    right = torch.sum(x * operator.adjoint(y))
    relative_error = torch.abs(left - right) / (torch.abs(left) + torch.abs(right) + 1e-8)
    assert relative_error < 1e-5
    assert kernel[0, 0, 0].item() == 0.0


def test_dipole_rejects_zero_b0_direction() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        build_dipole_kernel((5, 5, 5), b0_direction_zyx=(0.0, 0.0, 0.0))
